"""Evaluation tables and significance tests (protocol-compliant).

Wraps the shared statistical helpers from proposal1.daetf.experiments so the
KrylovNet reports use the same paired-Wilcoxon / bootstrap protocol as P1/P5.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from proposal1.daetf.experiments import (bootstrap_ci, comparison_table,
                                         markdown_table, significance_table,
                                         wilcoxon_signed_rank)


def run_comparison(results: Dict[str, Dict[str, np.ndarray]],
                   reference: Sequence[str] = ("unfold", "slt")) -> str:
    """results: method -> {'psnr': (n,), 'ssim': (n,), 'sam': (n,), 'ergas': (n,)}
    One row per method against the first reference method."""
    methods = list(results.keys())
    ref = reference[0] if reference[0] in results else methods[0]
    rows = []
    for m in methods:
        if m == ref:
            rows.append([m, *[f"{results[m][k].mean():.4f}" for k in
                              ("psnr", "ssim", "sam", "ergas")], "-"])
            continue
        cells = [m]
        for k in ("psnr", "ssim", "sam", "ergas"):
            d = results[m][k] - results[ref][k]
            cells.append(f"{results[m][k].mean():.4f}")
        p, w = wilcoxon_signed_rank(results[m]["sam"],
                                    results[ref]["sam"])
        sig = "sig" if p < 0.05 else "n.s."
        cells.append(f"p={p:.3f} ({sig})")
        rows.append(cells)
    return markdown_table(
        ["method", "psnr", "ssim", "sam", "ergas", "sam p"],
        rows)


def pair_sam_ci(results: Dict[str, Dict[str, np.ndarray]],
                a: str, b: str, n_boot: int = 2000) -> str:
    d = results[a]["sam"] - results[b]["sam"]
    lo, hi = bootstrap_ci(d, n_boot=n_boot)
    return f"SAM {a}-{b}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"


__all__ = ["run_comparison", "pair_sam_ci", "comparison_table",
           "significance_table", "markdown_table", "bootstrap_ci",
           "wilcoxon_signed_rank"]