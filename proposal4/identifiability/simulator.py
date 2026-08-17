"""P4 (capstone) simulator: scene/sensor identifiability phase diagram.

Controls the identifiability knobs (scene spectral rank, MSI band count, SRF
overlap, noise SNR, spatial scale) and reports two complementary measures of
how identifiable the fusion problem is:

    score      = r_id_hat / r   (spectral DOF the observations pin down, P2)
    null_frac  = ||P_N X||/||X||  (fraction of the scene the observations
                                   leave ambiguous, P1 combined projector)

Regime labels:  I (Identifiable), W (Weakly), N (Non-identifiable).  The
phase diagram maps (M, SNR) and (SRF width, SNR) onto these regimes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from proposal1.ambiguity.operator import CombinedOperator
from proposal2.metrics.identifiable_rank import estimate_ranks
from proposal2.rankest.generator import RankScene, make_srf


def regime(score: float) -> str:
    """score = r_id_hat / r.  I >= 0.75, N < 0.25, W in between."""
    if score >= 0.75:
        return "I"
    if score >= 0.25:
        return "W"
    return "N"


def simulate(rank: int = 8, msi_bands: int = 8, srf_width: float = 0.02,
             snr_db: Optional[float] = None, bands: int = 31, hw: int = 48,
             scale: int = 4, seed: int = 0) -> Dict:
    scene = RankScene(bands, msi_bands, scale, rank, hw, hw, seed,
                      srf_width=srf_width)
    P = CombinedOperator(scale, bands, msi_bands,
                         srf=make_srf(bands, msi_bands, srf_width),
                         cg_steps=120, ridge=1e-6)
    yH = P.D(scene.X[None])[0]                       # (B, h, w)
    yM = P.R(scene.X[None])[0]                       # (M, H, W)
    sigma = 0.0
    if snr_db is not None and not math.isinf(snr_db):
        sig = float(yM.std().clamp_min(1e-12))
        sigma = sig / (10 ** (snr_db / 20.0))
        g = torch.Generator().manual_seed(seed + 2)
        yH = yH + sigma * torch.randn_like(yH, generator=g)
        yM = yM + sigma * torch.randn_like(yM, generator=g)
    r_hat, r_id_hat, sigma_hat = estimate_ranks(yH, yM)
    x_amb = P.project_null(scene.X[None])
    null_frac = float(x_amb.norm().item() / scene.X.norm().clamp_min(1e-12).item())
    score = r_id_hat / rank
    return {"rank": rank, "msi_bands": msi_bands, "srf_width": srf_width,
            "snr_db": snr_db, "r_hat": r_hat, "r_id_hat": r_id_hat,
            "r_id_true": scene.r_id_true, "sigma": sigma,
            "sigma_hat": sigma_hat, "null_frac": null_frac,
            "score": score, "regime": regime(score)}