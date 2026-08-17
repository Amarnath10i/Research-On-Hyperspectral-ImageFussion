"""Finite-difference operators for the diffusion term and the tensor net.

``divergence(u, gxy, gz)`` implements  div(D grad u)  with a *cross-spectral*
tensor: ``gxy`` is the per-band in-plane diffusivity and ``gz`` couples adjacent
spectral bands.  Gradients use forward differences and the divergence is the
exact (negative) adjoint with zero-flux (Neumann) boundary, so the operator is
self-adjoint:

    <div(g grad u), v> == <u, div(g grad v)>

which the selfcheck verifies numerically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def divergence(u: torch.Tensor, gxy: torch.Tensor,
               gz: torch.Tensor) -> torch.Tensor:
    """div(D grad u) with face-sampled in-plane (gxy) and band-axis (gz)
    diffusivities.  All inputs (B, C, H, W)."""
    B, C, H, W = u.shape
    gx_face = (gxy[..., 1:, :] + gxy[..., :-1, :]) / 2.0    # (B,C,H-1,W)
    gy_face = (gxy[..., :, 1:] + gxy[..., :, :-1]) / 2.0    # (B,C,H,W-1)
    gz_face = (gz[:, 1:] + gz[:, :-1]) / 2.0                # (B,C-1,H,W)
    gx = gx_face * (u[..., 1:, :] - u[..., :-1, :])
    gy = gy_face * (u[..., :, 1:] - u[..., :, :-1])
    gz = gz_face * (u[:, 1:] - u[:, :-1])

    out = torch.zeros_like(u)
    out[:, :, 0, :] += gx[:, :, 0, :]
    out[:, :, 1:H - 1, :] += gx[:, :, 1:, :] - gx[:, :, :-1, :]
    out[:, :, H - 1, :] += -gx[:, :, -1, :]
    out[:, :, :, 0] += gy[:, :, :, 0]
    out[:, :, :, 1:W - 1] += gy[:, :, :, 1:] - gy[:, :, :, :-1]
    out[:, :, :, W - 1] += -gy[:, :, :, -1]
    out[:, 0] += gz[:, 0]
    out[:, 1:C - 1] += gz[:, 1:] - gz[:, :-1]
    out[:, C - 1] += -gz[:, -1]
    return out


class TensorNet(nn.Module):
    """Learned cross-spectral diffusion tensor.

    Inputs: current iterate u, upsampled LR guide x_up, and the MSI guide m
    (projected to C bands).  Emits per-pixel, per-band in-plane diffusivities
    gxy and band-axis diffusivities gz (positive via softplus)."""

    def __init__(self, bands: int, msi_bands: int, hidden: int = 24,
                 layers: int = 2):
        super().__init__()
        self.proj_m = nn.Conv2d(msi_bands, bands, 1)
        self.in_proj = nn.Conv2d(3 * bands, hidden, 3, padding=1)
        body = []
        for _ in range(layers):
            body += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU()]
        self.body = nn.Sequential(*body)
        self.head_xy = nn.Conv2d(hidden, bands, 3, padding=1)
        self.head_z = nn.Conv2d(hidden, bands, 1)

    def forward(self, u: torch.Tensor, x_up: torch.Tensor,
                m: torch.Tensor) -> tuple:
        feats = torch.cat([u, x_up, self.proj_m(m)], dim=1)
        h = F.silu(self.in_proj(feats))
        h = self.body(h)
        return (F.softplus(self.head_xy(h)),
                F.softplus(self.head_z(h)))