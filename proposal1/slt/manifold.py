"""Spectral manifold geometry - the SLT core.

Normalised spectra live on the unit sphere S^{B-1}: after per-pixel
L2-normalisation, the spectral angle SAM(p, q) = arccos(<p, q>) is *exactly*
the geodesic distance on that sphere.  We transport a base point (the
LR-derived direction) by a learned tangent vector through the exponential map,
so the network's only degrees of freedom are geodesic displacements and its
error is measured in the same units the headline metric uses.

Everything here is differentiable and validated numerically by selfcheck.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Per-pixel L2 normalisation over ``dim`` (default: the band axis)."""
    return x / x.norm(2, dim=dim, keepdim=True).clamp_min(eps)


def tangent_projection(v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Project ``v`` onto the tangent space of the unit sphere at ``p``.

    ``p`` must be unit-norm; the returned vector is orthogonal to ``p``.
    """
    c = (v * p).sum(1, keepdim=True)
    return v - c * p


def exp_map(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Spherical exponential map: move ``p`` along the geodesic in direction v.

    ``v`` is projected onto the tangent space at ``p`` first.  The magnitude of
    the transported displacement equals the geodesic (SAM) distance travelled.
    Safe at v = 0 (returns p) via the sinc limit.
    """
    v = tangent_projection(v, p)
    n = v.norm(2, 1, keepdim=True)
    n_c = n.clamp_max(math.pi)
    v = v * (n_c / n.clamp_min(1e-12))          # rescale to the clamped length
    n_s = n_c.clamp_min(1e-8)
    sinc = torch.sin(n_c) / n_s                  # -> 1 as n -> 0
    return torch.cos(n_c) * p + sinc * v


def log_map(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Logarithmic map: the tangent vector at ``p`` whose exponential image is q.

    ``Log_p(q)`` has magnitude equal to the geodesic distance SAM(p, q); it is
    the natural regression target for the transport (an isometry on the sphere,
    so squared MSE in the tangent space bounds the geodesic error).
    """
    c = (p * q).sum(1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(c)
    u = tangent_projection(q, p)                 # q - <p,q> p, orthogonal to p
    un = u.norm(2, 1, keepdim=True)
    unit = u / un.clamp_min(1e-8)
    unit = torch.where(un > 1e-6, unit, torch.zeros_like(unit))
    return unit * theta


def geodesic_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Geodesic distance in radians between unit vectors (== SAM in radians)."""
    c = (p * q).sum(1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(c)


def sam_degrees(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """SAM in degrees between two spectra (no normalisation required)."""
    return torch.rad2deg(geodesic_distance(l2_normalize(p), l2_normalize(q)))