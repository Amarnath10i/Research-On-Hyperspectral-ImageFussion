"""SLT experiment ladder.

The four mechanism stages share one architecture and differ only by two
switches (``manifold``, ``geodesic``, ``use_msi_guide``), so each added idea is
isolated.  Report per-scene PSNR/SSIM/SAM/ERGAS + LR-consistency; the roadmap
decision gate is Stage 3 vs Stage 2 (does the *exact* geodesic geometry beat a
naive chordal renormalisation) and Stage 4 vs Stage 3 (does the MSI guide earn
its channel).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from proposal1.daetf.experiments import (comparison_table, markdown_table,
                                         save_results, significance_table)

from .config import Config
from .engine import evaluate_dataset, train


def _stage_cfg(base: Config, **overrides) -> Config:
    values = dict(base.to_dict())
    values.update(overrides)
    return Config(**values)


STAGE_LADDER: List[Dict] = [
    {"name": "s1_euclid_residual", "desc": "Euclidean residual (regression)",
     "overrides": {"manifold": False, "use_msi_guide": False}},
    {"name": "s2_naive_chordal", "desc": "Naive manifold: add+renormalise",
     "overrides": {"manifold": True, "geodesic": False, "use_msi_guide": False}},
    {"name": "s3_geodesic_exp", "desc": "Fisher-sphere exponential map",
     "overrides": {"manifold": True, "geodesic": True, "use_msi_guide": False}},
    {"name": "s4_msi_guided", "desc": "MSI-guided geodesic transport",
     "overrides": {"manifold": True, "geodesic": True, "use_msi_guide": True}},
]


def run_slt_ladder(base_cfg: Config, device: str = "cuda",
                   iters: Optional[int] = None,
                   seeds: tuple = (0, 1, 2)) -> Dict:
    """Train + evaluate every ladder stage; returns a comparison payload.

    Each stage is trained from a fresh seed with its own out_dir, so the stages
    are comparable (no shared checkpoint).  ``iters`` overrides the training
    budget when supplied (used for smoke runs).
    """
    results: Dict[str, Dict] = {}
    for stage in STAGE_LADDER:
        name = stage["name"]
        cfg = _stage_cfg(base_cfg, out_dir=f"./slt_ladder/{name}",
                         name=f"slt_{name}", **stage["overrides"])
        if iters is not None:
            cfg.iters = iters
        rows_by_seed: List[List] = []
        for seed in seeds:
            cfg.seed = seed
            model, _ = train(cfg, device, log_fn=print)
            mean, rows = evaluate_dataset(model, cfg.source_root, cfg, "Test",
                                          device, return_rows=True)
            rows_by_seed.append([r for r in rows])
            print(f"[ladder] {name} seed={seed} "
                  f"SAM={mean['sam']:.3f} ERGAS={mean['ergas']:.3f}")
        results[name] = {"mean": {k: float(v) for k, v in mean.items()},
                         "stage": stage["desc"], "rows": rows_by_seed}
    return results


def ladder_report(results: Dict) -> str:
    """Markdown table of stage means with the paired significance tests."""
    header = ["stage", "PSNR", "SSIM", "SAM", "ERGAS", "LRcons"]
    rows = []
    for name, r in results.items():
        m = r["mean"]
        rows.append([name, f"{m['psnr']:.3f}", f"{m['ssim']:.4f}",
                     f"{m['sam']:.3f}", f"{m['ergas']:.3f}",
                     f"{m['lr_consistency']:.2e}"])
    out = markdown_table(header, rows)
    comparisons = [{"name": f"{b} vs {a}",
                    "metric": "sam",
                    "rows": results[a]["rows"], "baseline_rows": results[b]["rows"]}
                   for a, b in (("s2_naive_chordal", "s1_euclid_residual"),
                                ("s3_geodesic_exp", "s2_naive_chordal"),
                                ("s4_msi_guided", "s3_geodesic_exp"))]
    out += "\n\n" + significance_table(comparisons, metric="sam")
    return out


def main(base_cfg: Optional[Config] = None, device: str = "cuda") -> None:
    cfg = base_cfg or Config().resolve()
    results = run_slt_ladder(cfg, device)
    report = ladder_report(results)
    save_results("./slt_ladder/slt_ladder_results.json",
                 {"cfg": cfg.to_dict(), "results": results, "report": report})
    print(report)


if __name__ == "__main__":
    main()