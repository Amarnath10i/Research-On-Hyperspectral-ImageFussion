"""Observable/ambiguous decomposition, hallucination metric, uncertainty map.

Everything here is a function of the operator A=[D;R] only (no network, no
learned prior), so it is the falsifiable scaffold for P1's empirical claims:
the ambiguity map predicts where a null-space-ignorant reconstruction fails,
and null-space perturbations never touch the observable component.
"""

from __future__ import annotations

from typing import Tuple

import torch


def decompose(x: torch.Tensor, P) -> Tuple[torch.Tensor, torch.Tensor]:
    """X = X_obs + X_amb with X_obs = A^T(A A^T)^-1 A X, X_amb = P_N X."""
    return P.project_range(x), P.project_null(x)


def obs_amb_split(x: torch.Tensor, x_hat: torch.Tensor, P):
    """Per-reconstruction observable and ambiguous components."""
    x_obs, x_amb = decompose(x, P)
    xh_obs, xh_amb = decompose(x_hat, P)
    return x_obs, x_amb, xh_obs, xh_amb


def error_split(x: torch.Tensor, x_hat: torch.Tensor, P, eps: float = 1e-6):
    """E_obs = relative error on the observable component, E_amb on the
    ambiguous component.  Structural identity: E_obs is invariant to any
    perturbation of the null-space content."""
    x_obs, x_amb, xh_obs, xh_amb = obs_amb_split(x, x_hat, P)
    E_obs = (x_obs - xh_obs).norm() / (x_obs.norm() + eps)
    E_amb = (x_amb - xh_amb).norm() / (x_amb.norm() + eps)
    return E_obs, E_amb


def hallucination(x: torch.Tensor, x_hat: torch.Tensor, P, eps: float = 1e-6):
    """H = ||P_N(x_hat - x)|| / (||P_N x|| + eps) - normalized error on the
    genuinely-ambiguous component.  H in [0, ~1]: 0 when the null-space
    content is reproduced, ~1 when it is fabricated or dropped wholesale."""
    _, x_amb, _, xh_amb = obs_amb_split(x, x_hat, P)
    return (xh_amb - x_amb).norm() / (x_amb.norm() + eps)


def uncertainty_map(x: torch.Tensor, P):
    """U(x,y) = ||P_N X|| per pixel, U(lam) = ||P_N X|| per band.

    Both measure how much of the scene content at that location/band is NOT
    pinned by the observations, i.e. where a fusion must guess.
    """
    _, x_amb = decompose(x, P)
    u_px = x_amb.pow(2).sum(-3).sqrt()                      # (..., H, W)
    u_band = x_amb.pow(2).flatten(-2).mean(-1).sqrt()       # (..., B)
    return u_px, u_band


def pixelwise_amb_error(x: torch.Tensor, x_hat: torch.Tensor, P):
    """||P_N(x_hat - x)|| per pixel - where the ambiguous error lands."""
    _, x_amb, _, xh_amb = obs_amb_split(x, x_hat, P)
    return (xh_amb - x_amb).pow(2).sum(-3).sqrt()           # (..., H, W)


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())