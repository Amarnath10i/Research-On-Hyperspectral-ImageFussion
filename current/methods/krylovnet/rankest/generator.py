"""Controlled synthetic scene generator for P2 (identifiable spectral rank).

Generates HR-HSI X = U_r Z with known intrinsic spectral rank r and known
degradation physics (per-band blur+downsample D, SRF R), so the
*observation-identifiable rank* r_id = rank(R^T U_r) is ground truth.

All quantities needed to validate a rank estimator are exposed:
  .X          HR-HSI (bands, H, W)
  .Y_H        LR-HSI (bands, h, w) = D(X)
  .Y_M        HR-MSI (msi_bands, H, W) = R^T X
  .U          true spectral basis (bands, r)
  .R          true SRF (bands, msi_bands)
  .r          intrinsic rank
  .r_id_true  observation-identifiable rank = rank(R^T U_r)
  .sigma      injected noise std (0 if clean)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def gaussian_kernel(sigma: float, ksize: int) -> torch.Tensor:
    x = torch.arange(ksize).float() - ksize // 2
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum().clamp_min(1e-12)


def degrade_spatial(x: torch.Tensor, sigma: float, scale: int) -> torch.Tensor:
    """Per-band Gaussian blur then strided downsampling (x: B,C,H,W)."""
    B, C, H, W = x.shape
    ksize = 2 * int(3 * sigma) + 1
    k = gaussian_kernel(sigma, ksize)
    w = (k.outer(k)).view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = ksize // 2
    xb = F.conv2d(x, w, groups=C, padding=pad)
    return xb[:, :, ::scale, ::scale]


def make_srf(bands: int, msi_bands: int, width: float = 0.05) -> torch.Tensor:
    """Gaussian-bump SRF columns over a wavelength axis; width controls overlap."""
    lam = torch.linspace(0.0, 1.0, bands)
    centers = torch.linspace(0.15, 0.85, msi_bands)
    R = torch.exp(-((lam[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
    R = R / R.sum(0, keepdim=True).clamp_min(1e-12)
    return R


def _smooth(z: torch.Tensor, sigma: float) -> torch.Tensor:
    ksize = 2 * int(3 * sigma) + 1
    k = gaussian_kernel(sigma, ksize)  # (ksize,)
    z = z.unsqueeze(0).unsqueeze(0)
    z = F.conv2d(z, k.view(1, 1, 1, ksize), padding=(0, ksize // 2))
    z = F.conv2d(z, k.view(1, 1, ksize, 1), padding=(ksize // 2, 0))
    return z.squeeze(0).squeeze(0)


def spatial_modes(H: int, W: int, count: int,
                  max_freq: int = 7) -> torch.Tensor:
    """Orthogonal 2D cosine modes (kx,ky in 1..max_freq), exact rank control.

    Modes use frequencies below the LR Nyquist (scale=4, LR grid H/4) so the
    blur+downsample operator preserves their linear independence.
    """
    pairs = [(kx, ky) for s in range(2, 2 * max_freq + 1)
             for kx in range(1, max_freq + 1)
             for ky in range(1, max_freq + 1) if kx + ky == s][:count]
    assert len(pairs) >= count, "not enough modes"
    x = torch.arange(W).float() / W
    y = torch.arange(H).float() / H
    fields = []
    for kx, ky in pairs:
        f = torch.cos(2 * math.pi * kx * x).view(1, -1) * \
            torch.cos(2 * math.pi * ky * y).view(-1, 1)
        f = f - f.mean()
        f = f / f.std().clamp_min(1e-9)
        fields.append(f)
    return torch.stack(fields, dim=0)  # (count, H, W), unit-variance orthogonal rows


class RankScene:
    def __init__(self, bands: int = 31, msi_bands: int = 16, scale: int = 4,
                 rank: int = 12, H: int = 64, W: int = 64, seed: int = 0,
                 srf_width: float = 0.05, decay: float = 0.99,
                 snr_db: Optional[float] = None, sigma: float = 0.0,
                 spatial_sigma: float = 2.0, blur_sigma: float = 1.2):
        g = torch.Generator().manual_seed(seed)
        assert rank <= bands, "intrinsic rank must not exceed bands"
        self.bands = bands
        self.msi_bands = msi_bands
        self.scale = scale
        self.r = rank
        self.H, self.W = H, W
        self.h, self.w = H // scale, W // scale

        # spectral basis (orthonormal) + orthogonal smooth spatial modes
        U = torch.linalg.qr(torch.randn(bands, rank, generator=g)).Q
        modes = spatial_modes(H, W, rank)
        fields = [modes[k] * (decay ** k) for k in range(rank)]
        Z = torch.stack(fields, dim=0).reshape(rank, H * W)  # (r, HW)
        self.U = U
        self.Z = Z

        X = (U @ Z).reshape(bands, H, W)
        X = X / X.abs().max().clamp_min(1e-12)  # pure scalar scale, rank preserved
        self.X = X

        self.R = make_srf(bands, msi_bands, srf_width)
        self.Y_H = degrade_spatial(X.unsqueeze(0), blur_sigma, scale).squeeze(0)
        self.Y_M = torch.einsum("bhw,bm->mhw", X, self.R)

        if sigma > 0:
            pass
        elif snr_db is not None and math.isinf(snr_db):
            sigma = 0.0
        elif snr_db is not None:
            sig = self.Y_M.std().clamp_min(1e-12)
            sigma = float(sig / (10 ** (snr_db / 20.0)))
        self.sigma = float(sigma)
        if sigma > 0:
            g2 = torch.Generator().manual_seed(seed + 1)
            self.Y_H = self.Y_H + sigma * torch.randn_like(self.Y_H, generator=g2)
            self.Y_M = self.Y_M + sigma * torch.randn_like(self.Y_M, generator=g2)

        self.r_id_true = self.identifiable_rank_true()

    def identifiable_rank_true(self) -> int:
        G = self.R.t() @ self.U  # (msi_bands, r)
        s = torch.linalg.svdvals(G)
        s0 = s[0].clamp_min(1e-12)
        return int((s > 1e-6 * s0).sum().item())

    def __repr__(self) -> str:
        return (f"RankScene(r={self.r}, r_id_true={self.r_id_true}, "
                f"bands={self.bands}, msi={self.msi_bands}, sigma={self.sigma:.2e})")