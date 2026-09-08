"""Observation-identifiable spectral rank estimators (non-neural).

Core object of P2:
    r_id = number of spectral degrees of freedom of the HR-HSI recoverable
           through the combined observation operator A = [D; R].

Estimators (all derived from observations only, no oracle):
  estimate_noise              trailing-singular-value noise std from the LR-HSI
  estimate_hsi_rank           hard-threshold rank of the LR-HSI spectral space
  estimate_identifiable_rank  hard-threshold count of MSI spectral singular
                              values (Gavish-Donoho optimal threshold on the
                              noise level), capped by the MSI band count
  estimate_ranks              full pipeline returning (r_hat, r_id_hat, sigma_hat)

The hard threshold is
    tau = omega(beta) * sigma * sqrt(n),
with the noise level recovered from the data; in the noise-free regime the
threshold degrades to a small relative floor so the structurally significant
directions are counted exactly.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def energy_rank(mat: torch.Tensor, frac: float = 0.995) -> int:
    s = torch.linalg.svdvals(mat.float())
    e = s * s
    total = e.sum().clamp_min(1e-30)
    k = int((e.cumsum(0) <= frac * total).sum().item()) + 1
    return max(1, min(k, s.numel()))


def estimate_noise(mat: torch.Tensor, frac: float = 0.9) -> float:
    """Noise std from the trailing singular values of an observation matrix.

    For a low-rank-plus-noise matrix (D x N), the singular values beyond the
    energy-frac cap belong to the noise bulk, whose median is ~ sigma*sqrt(N)
    (Marchenko-Pastur).  In the noise-free case the tail is ~0.
    """
    mat = mat.reshape(mat.shape[0], -1).float()
    D, N = mat.shape
    if D >= N:
        mat = mat.t()
        D, N = mat.shape
    s = torch.linalg.svdvals(mat)
    r_cap = energy_rank(mat, frac)
    tail = s[r_cap:]
    if tail.numel() == 0:
        return 0.0
    return float(tail.median() / math.sqrt(N))


def gavish_donoho_beta(beta: float) -> float:
    """omega(beta) for the optimal hard threshold, beta = m/n in (0, 1]."""
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.43 * beta + 1.43


def hard_threshold_count(s: torch.Tensor, sigma: float,
                         n: int, floor_rel: float = 1e-6) -> int:
    """Count singular values above max(tau, floor_rel * s_max)."""
    s = s.clamp_min(1e-30)
    if sigma is not None and sigma > 0:
        tau = gavish_donoho_beta(1.0) * sigma * math.sqrt(n)
    else:
        tau = 0.0
    thresh = max(tau, floor_rel * s[0].clamp_min(1e-12))
    return int((s >= thresh).sum().item())


def estimate_hsi_rank(lr_hsi: torch.Tensor,
                      sigma: Optional[float] = None) -> int:
    """Intrinsic spectral rank estimate from the LR-HSI (B, h, w)."""
    Xm = lr_hsi.reshape(lr_hsi.shape[0], -1).float()
    if sigma is None:
        sigma = estimate_noise(Xm)
    s = torch.linalg.svdvals(Xm)
    return hard_threshold_count(s, sigma, Xm.shape[1])


def estimate_identifiable_rank(msi: torch.Tensor,
                               sigma: Optional[float] = None,
                               floor_rel: float = 1e-6) -> int:
    """Observation-identifiable spectral rank from the HR-MSI (M, H, W).

    Uses the optimal hard threshold on the singular values of the MSI's
    spectral matrix (M x HW).  sigma is the shared observation noise level
    (estimated from the LR-HSI by estimate_ranks); if None/0 the relative
    floor is used (noise-free case).
    """
    M, H, W = msi.shape
    n = H * W
    Xm = msi.reshape(M, n).t().float()  # (n, M), n >= M
    n_, m = Xm.shape
    s = torch.linalg.svdvals(Xm)
    if sigma is not None and sigma > 0:
        beta = min(m, n_) / max(m, n_)
        tau = gavish_donoho_beta(beta) * sigma * math.sqrt(n_)
    else:
        tau = 0.0
    thresh = max(tau, floor_rel * s[0].clamp_min(1e-12))
    return int((s >= thresh).sum().item())


def estimate_ranks(lr_hsi: torch.Tensor, msi: torch.Tensor
                   ) -> Tuple[int, int, float]:
    """Full observation-only pipeline: (r_hat, r_id_hat, sigma_hat)."""
    sigma_hat = estimate_noise(lr_hsi.reshape(lr_hsi.shape[0], -1))
    r_hat = estimate_hsi_rank(lr_hsi, sigma=sigma_hat)
    r_id_hat = estimate_identifiable_rank(msi, sigma=sigma_hat)
    return r_hat, r_id_hat, sigma_hat