#!/usr/bin/env python
"""Stage-0 baseline audit: run bicubic, GSA, and subspace-LS under one protocol.

This is the prerequisite for every paper in the program.  Until these numbers
look sane (bicubic PSNR in a reasonable range, GSA improving over bicubic),
no learned method can be trusted.

Usage:
    python tools/run_baseline_audit.py                     # defaults
    python tools/run_baseline_audit.py --source /path/to/cave
    python tools/run_baseline_audit.py --scale 8 --limit 5  # quick check
    python tools/run_baseline_audit.py --bootstrap 200       # tighter CIs

The output is a markdown table saved to results/BASELINE_AUDIT.md alongside
the raw per-scene numbers.  The protocol follows PROTOCOL_AUDIT.md:
  - PSNR with fixed data_range=1.0 (NOT per-image max)
  - SSIM averaged over bands
  - SAM in degrees
  - ERGAS with the true scale factor
  - Paired Wilcoxon between each method and the next row
  - Bootstrap 95% CIs for mean metrics (unless --no-bootstrap)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
for d in [str(ROOT / "common"), str(ROOT / "proposal1"), str(ROOT / "proposal2"),
          str(ROOT / "proposal3"), str(ROOT / "proposal4"), str(ROOT / "proposal5")]:
    if d not in sys.path:
        sys.path.insert(0, d)


def _bootstrap_ci(values, n_boot=200, ci=0.95, seed=0):
    """Non-parametric bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    means = [arr[rng.randint(len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(arr)), float(lo), float(hi)


def _wilcoxon_p(a, b):
    """Paired Wilcoxon signed-rank test (two-sided), returns p-value or NaN."""
    try:
        from scipy.stats import wilcoxon
        d = np.array(a) - np.array(b)
        d = d[np.abs(d) > 1e-12]
        if len(d) < 5:
            return float("nan")
        _, p = wilcoxon(d)
        return float(p)
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Stage-0 baseline audit")
    parser.add_argument("--source", default=None, help="Path to CAVE dataset")
    parser.add_argument("--target", default=None, help="Path to Harvard dataset")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Max scenes per dataset")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap samples for CIs")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    import torch
    from common.hsifusion.config import BaseConfig
    from common.hsifusion.baselines import evaluate_all_baselines, BASELINES
    from common.hsifusion.io_utils import discover_dataset

    # ── resolve dataset paths ──────────────────────────────────────────────
    source = args.source or discover_dataset(["cave"], verbose=True)
    target = args.target or discover_dataset(["harvard"], required=False, verbose=True)

    out_dir = Path(args.out) if args.out else ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset_name, root in [("CAVE", source), ("Harvard", target)]:
        if root is None:
            print(f"\n[skip] {dataset_name} dataset not found")
            continue

        print(f"\n{'='*60}\n  {dataset_name}  (root={root})\n{'='*60}")

        cfg = BaseConfig(scale=args.scale).resolve(
            source_hints=["cave"] if dataset_name == "CAVE" else ["harvard"],
            verbose=True
        )

        # use a fixed SRF (no jitter in the audit — protocol-equal)
        srf = np.eye(cfg.bands, cfg.msi_bands) if cfg.msi_bands == cfg.bands \
            else np.ones((cfg.bands, cfg.msi_bands)) / cfg.msi_bands

        device = args.device if torch.cuda.is_available() else "cpu"
        results = evaluate_all_baselines(root, cfg, srf, device=device,
                                         limit=args.limit, verbose=True)
        all_results[dataset_name] = results

    # ── write markdown table ───────────────────────────────────────────────
    lines = ["# Baseline Audit (Stage-0)\n",
             "Protocol: PROTOCOL_AUDIT.md (fixed data_range=1.0, Gaussian SSIM, "
             "degrees SAM, true-scale ERGAS).\n"]
    lines.append(f"Scale: ×{args.scale}  |  Scenes per dataset: "
                 f"{'all' if args.limit is None else args.limit}\n\n")

    for dataset_name, results in all_results.items():
        lines.append(f"## {dataset_name}\n\n")
        lines.append("| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Wilcoxon p (vs prev) |\n")
        lines.append("|---|---|---|---|---|---|\n")

        prev_means = None
        method_order = list(results.keys())
        for name in method_order:
            data = results[name]
            mean = data["mean"]
            row_vals = data["rows"]

            if not args.no_bootstrap:
                psnr_m, psnr_lo, psnr_hi = _bootstrap_ci(
                    [r["psnr"] for r in row_vals], args.bootstrap)
                ssim_m, ssim_lo, ssim_hi = _bootstrap_ci(
                    [r["ssim"] for r in row_vals], args.bootstrap)
                sam_m, sam_lo, sam_hi = _bootstrap_ci(
                    [r["sam"] for r in row_vals], args.bootstrap)
                ergas_m, ergas_lo, ergas_hi = _bootstrap_ci(
                    [r["ergas"] for r in row_vals], args.bootstrap)
                psnr_str = f"{psnr_m:.3f} [{psnr_lo:.3f}, {psnr_hi:.3f}]"
                ssim_str = f"{ssim_m:.4f} [{ssim_lo:.4f}, {ssim_hi:.4f}]"
                sam_str = f"{sam_m:.3f} [{sam_lo:.3f}, {sam_hi:.3f}]"
                ergas_str = f"{ergas_m:.3f} [{ergas_lo:.3f}, {ergas_hi:.3f}]"
            else:
                psnr_str = f"{mean['psnr']:.3f}"
                ssim_str = f"{mean['ssim']:.4f}"
                sam_str = f"{mean['sam']:.3f}"
                ergas_str = f"{mean['ergas']:.3f}"

            # paired Wilcoxon against previous method
            if prev_means is not None:
                p = _wilcoxon_p(
                    [r["psnr"] for r in prev_means],
                    [r["psnr"] for r in row_vals]
                )
                p_str = f"{p:.4f}" if np.isfinite(p) else "N/A"
            else:
                p_str = "—"

            lines.append(f"| {name} | {psnr_str} | {ssim_str} | {sam_str} | {ergas_str} | {p_str} |\n")
            prev_means = row_vals

        # per-scene detail
        lines.append(f"\n### Per-scene detail\n\n")
        for name in method_order:
            lines.append(f"#### {name}\n\n")
            lines.append("| Scene | PSNR | SSIM | SAM | ERGAS |\n")
            lines.append("|---|---|---|---|---|\n")
            for r in results[name]["rows"]:
                lines.append(f"| {r['scene']} | {r['psnr']:.3f} | {r['ssim']:.4f} | "
                             f"{r['sam']:.3f} | {r['ergas']:.3f} |\n")
            lines.append("\n")

    out_path = out_dir / "BASELINE_AUDIT.md"
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"\n[done] wrote {out_path}")

    # also save raw JSON
    import json
    json_path = out_dir / "baseline_audit.json"
    # convert numpy to python types for JSON
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_convert)
    print(f"[done] wrote {json_path}")


if __name__ == "__main__":
    main()