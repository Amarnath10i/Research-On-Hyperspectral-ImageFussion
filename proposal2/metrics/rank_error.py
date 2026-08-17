"""Rank-error metric for P2.

Primary headline metric: |r_hat - r_id|, the deviation of the estimated
observation-identifiable rank from the ground-truth identifiable rank.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rank_error(r_hat, r_true) -> float:
    return float(abs(int(r_hat) - int(r_true)))


def summarize(estimates, truths, tol: int = 1) -> dict:
    """Summarize rank estimates against ground truths across a grid.

    Returns dict with per-sample errors plus aggregate stats and the number of
    samples within +/-tol (the paper's headline: "recovers r_id within +-1").
    """
    est = np.asarray([int(e) for e in estimates])
    tru = np.asarray([int(t) for t in truths])
    err = np.abs(est - tru)
    return {
        "n": int(err.size),
        "mean_abs_err": float(err.mean()),
        "max_abs_err": float(err.max()) if err.size else 0.0,
        "within_tol": int((err <= tol).sum()),
        "within_tol_frac": float((err <= tol).mean()) if err.size else 0.0,
    }


def fmt(summary: dict) -> str:
    return (f"n={summary['n']} mean|err|={summary['mean_abs_err']:.3f} "
            f"max|err|={summary['max_abs_err']} "
            f"within+/-1={summary['within_tol']}/{summary['n']} "
            f"({100 * summary['within_tol_frac']:.0f}%)")