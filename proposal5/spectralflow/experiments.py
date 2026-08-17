"""Q1 stage-ladder runner for SpectralFlow.

The critical experiment (docs/ARCHITECTURE.md Stage 1 vs Stage 2) shares ONE
trained score network; only the inference-time switches change.  This module
makes that comparison runnable in a single script:

    python -c "import spectralflow.experiments as ex; ex.main()"

Because the projection lives in the sampler and never sees training, toggling
it does not require re-training - which is exactly why the comparison is cheap
enough to run many times and why a failure here is attributable to the
projection, not to a different optimisation schedule.

Stages (matching docs/ARCHITECTURE.md):
    0. Bicubic / GSA / HySure          protocol audit (run in hsifusion)
    1. Plain DDIM (use_projection=False)   does the prior learn the manifold?
    2. Stage 1 + per-step null-space projection   does the projection help?
    3. Stage 2, HR-MSI guide zeroed     is the guide load-bearing at inference?
    4. Stage 3 + blind operator refinement (refine_rounds>0)  does it help?

Reported per scene: PSNR, SSIM, SAM, ERGAS, LR-consistency (the headline for
the structural claim: ~1e-6 with the projection, ~1e-2 without).
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .engine import SamplingModel, build_model, evaluate_dataset, load_checkpoint

HEADLINE = ("sam", "ergas", "psnr", "ssim")
LOWER_IS_BETTER = {"sam": True, "ergas": True, "lr_consistency": True,
                   "psnr": False, "ssim": False}

# (name, overrides) evaluated on the SAME checkpoint.  Stage 4 re-enables the
# optional per-scene operator refinement and therefore re-samples each scene.
STAGE_LADDER: List[Tuple[str, Dict]] = [
    ("Stage 1: plain DDIM (no projection)",   {"use_projection": False}),
    ("Stage 2: null-space projected DDIM",    {"use_projection": True}),
    ("Stage 3: projected, no MSI guide",      {"use_projection": True,
                                               "use_msi_guide": False}),
]


# ---------------------------------------------------------------- statistics
def _normal_sf(z: float) -> float:
    import math
    return 0.5 * math.erfc(z / (2 ** 0.5))


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test with a normal approximation."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    d = d[d != 0]
    n = d.size
    if n < 2:
        return {"n": float(n), "p": float("nan")}
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    absd = np.abs(d)[order]
    i = 0
    while i < n:                                   # average ranks in ties
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    sigma = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    z = (w - mu) / sigma if sigma > 0 else 0.0
    return {"n": float(n), "p": float(min(2 * _normal_sf(abs(z)), 1.0))}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else float("inf"))


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000,
                 seed: int = 0) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_rows(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    out = {}
    for m in ("psnr", "ssim", "sam", "ergas", "lr_consistency"):
        v = [r[m] for r in rows]
        lo, hi = bootstrap_ci(v)
        out[m] = {"mean": float(np.mean(v)),
                  "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                  "ci_lo": lo, "ci_hi": hi, "n": len(v)}
    return out


# ------------------------------------------------------------------- ladder
def run_stage_ladder(model: SamplingModel, cfg: Config, root: str,
                     split: str = "Test", device: str = "cuda",
                     overrides: Optional[Sequence[Tuple[str, Dict]]] = None,
                     tile_hr: int = 256, verbose: bool = True) -> List[Dict]:
    """Evaluate ONE checkpoint under each ladder configuration.

    The model is re-used, so a difference between Stage 1 and Stage 2 is
    attributable to the projection alone - the identical network, weights,
    seeds and scenes.
    """
    overrides = list(overrides if overrides is not None else STAGE_LADDER)
    results = []
    for name, ov in overrides:
        if verbose:
            print(f"\n=== {name} ===")
        model.set_projection(ov.get("use_projection", cfg.use_projection))
        model.set_msi_guide(ov.get("use_msi_guide", cfg.use_msi_guide))
        rounds = ov.get("refine_rounds", cfg.refine_rounds)
        mean, rows = evaluate_dataset(model, root, cfg, split, device,
                                      tile_hr=tile_hr, verbose=verbose,
                                      return_rows=True, refine_rounds=rounds)
        results.append({"stage": name, "mean": mean, "rows": rows})
        if verbose:
            print(f"  MEAN  SAM={mean['sam']:.3f}  ERGAS={mean['ergas']:.3f}  "
                  f"PSNR={mean['psnr']:.3f}  LRcons={mean['lr_consistency']:.2e}")
    return results


def stage_comparison(a: Dict, b: Dict) -> Dict:
    """Paired comparison between two ladder rows over shared scenes."""
    by_b = {r["scene"]: r for r in b["rows"]}
    shared = [r for r in a["rows"] if r["scene"] in by_b]
    out = {"name_a": a["stage"], "name_b": b["stage"], "n_scenes": len(shared)}
    for m in ("psnr", "ssim", "sam", "ergas", "lr_consistency"):
        av = [r[m] for r in shared]
        bv = [by_b[r["scene"]][m] for r in shared]
        delta = float(np.mean(av) - np.mean(bv))
        improved = delta < 0 if LOWER_IS_BETTER.get(m, False) else delta > 0
        out[m] = {"mean_a": float(np.mean(av)), "mean_b": float(np.mean(bv)),
                  "delta": delta, "improved": bool(improved),
                  "p": wilcoxon_signed_rank(av, bv)["p"],
                  "d": cohens_d(av, bv),
                  "ci_a": bootstrap_ci(av), "ci_b": bootstrap_ci(bv)}
    return out


# -------------------------------------------------------------------- tables
def ladder_table(results: List[Dict]) -> str:
    headers = ["Stage", "PSNR up", "SSIM up", "SAM (deg) down",
               "ERGAS down", "LR-consistency down"]
    rows = []
    for r in results:
        m = r["mean"]
        rows.append([r["stage"], f"{m['psnr']:.3f}", f"{m['ssim']:.4f}",
                     f"{m['sam']:.3f}", f"{m['ergas']:.3f}",
                     f"{m['lr_consistency']:.2e}"])
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def significance_table(comparisons: List[Dict], metric: str = "sam") -> str:
    headers = ["Comparison", f"{metric} (A)", f"{metric} (B)", "delta",
               "Wilcoxon p", "Cohen d", "n"]
    rows = []
    for c in comparisons:
        e = c[metric]
        star = "***" if e["p"] < 0.001 else "**" if e["p"] < 0.01 else \
               "*" if e["p"] < 0.05 else "n.s."
        rows.append([f"{c['name_b']} vs {c['name_a']}",
                     f"{e['mean_a']:.4f}", f"{e['mean_b']:.4f}",
                     f"{e['delta']:+.4f}", f"{e['p']:.4f} {star}",
                     f"{e['d']:.2f}", int(c["n_scenes"])])
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def main(device: str = "cuda") -> None:
    """Run the full Stage 1 vs Stage 2 ladder on the best checkpoint."""
    import os
    ckpt_dir = "./spectralflow_out"
    cfg = Config(out_dir=ckpt_dir).resolve(verbose=False)
    cfg.out_dir = ckpt_dir
    model = load_checkpoint(cfg, device)
    results = run_stage_ladder(model, cfg, cfg.source_root, device=device)
    print("\n" + ladder_table(results))
    if len(results) >= 2:
        comp = [stage_comparison(results[i], results[i + 1])
                for i in range(len(results) - 1)]
        print("\nSAM (the metric that fails under domain shift):\n"
              + significance_table(comp, "sam"))
        print("\nLR-consistency (the structural claim):\n"
              + significance_table(comp, "lr_consistency"))


if __name__ == "__main__":
    main()
