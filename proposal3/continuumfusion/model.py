"""ContinuumFusion - continuous spectral-spatial representation for HSI-MSI fusion.

WHERE THIS DIFFERS FROM PROPOSALS 1 AND 2
-----------------------------------------
Both earlier proposals produce a fixed-size output grid: the upsampler in
proposal 1 and the decimation operator in proposal 2 are both built for one
scale factor, and changing it means retraining.

ContinuumFusion never represents the image as a grid. It learns a continuous
function

    f(x, y, lambda) -> radiance

conditioned on local features from the observations, and *samples* that function
wherever an output pixel is wanted. The scale factor becomes a query parameter,
not an architectural constant.

WHY THIS IS THE RIGHT GAP TO ATTACK
-----------------------------------
The benchmark in `existing/` is unusable precisely because its ten methods ran
at x4, x8, x16 and x32 and cannot be compared. That is not an accident of
sloppy bookkeeping - it is a property of grid-based architectures, each of
which is welded to the factor it was trained for. A model that handles any
factor from one set of weights makes the comparison well-posed: every method
can be evaluated at every factor.

It also matters physically. A real sensor's resolution ratio is whatever the
optics give you, rarely a power of two.

THE ARCHITECTURE
----------------
1. Encoder: LR-HSI and MSI are encoded to a feature map on the *LR* grid,
   because that is where the hyperspectral information actually lives.
2. Continuous decoder: for an output coordinate (x, y), gather the four nearest
   LR feature vectors, and for each, feed [feature, delta_coord, cell_size] to
   an MLP that predicts the radiance. Blend the four by area weights. This is
   local implicit image function decoding, made spectral.
3. Spectral coordinate: the band index enters as a coordinate too, through a
   learned per-band embedding, so the same MLP produces all bands and the
   representation is continuous along wavelength as well. That is what allows
   querying bands the sensor never sampled.
4. MSI detail is injected at the queried coordinate by bilinear sampling of the
   HR MSI feature map, so high-frequency spatial information is read at full
   resolution rather than upsampled from the LR grid.

WHY IT SHOULD TRANSFER
----------------------
The decoder sees relative coordinates and cell sizes, never absolute image
size, so it is scale-agnostic by construction. Training with randomised scale
factors turns the resolution ratio into just another nuisance variable the
model has learned to be robust to - the same argument as randomising the blur,
applied to geometry.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- encoder
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    """Features on the LR grid, fused with MSI context pooled to that grid.

    The latent lives on the LR grid on purpose: the spectra are only measured
    there, so that is where spectral reasoning belongs. Spatial detail is added
    later, at query time, from the full-resolution MSI.
    """

    def __init__(self, bands: int, msi_bands: int, width: int, depth: int):
        super().__init__()
        self.hsi_stem = nn.Conv2d(bands, width, 3, 1, 1)
        self.msi_stem = nn.Conv2d(msi_bands, width, 3, 1, 1)
        self.fuse = nn.Conv2d(width * 2, width, 1)
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(depth)])

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        h = self.hsi_stem(lr_hsi)
        m = self.msi_stem(msi)
        m_lr = F.adaptive_avg_pool2d(m, lr_hsi.shape[-2:])
        z = self.fuse(torch.cat([h, m_lr], dim=1))
        return self.body(z)


class DetailEncoder(nn.Module):
    """High-resolution MSI features, sampled at query coordinates."""

    def __init__(self, msi_bands: int, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(msi_bands, width, 3, 1, 1), nn.ReLU(inplace=True),
            ResBlock(width), nn.Conv2d(width, width, 3, 1, 1))

    def forward(self, msi: torch.Tensor) -> torch.Tensor:
        return self.body(msi)


# ------------------------------------------------------------ continuous decoder
class SpectralCoordMLP(nn.Module):
    """Decodes (latent, relative coordinate, cell size, band embedding) -> radiance.

    The band index is a *coordinate*, not an output channel. One MLP therefore
    serves every band, the parameter count does not grow with band count, and
    the representation is continuous along wavelength - so a band the sensor
    never sampled can be queried by interpolating its embedding.
    """

    def __init__(self, latent: int, detail: int, bands: int, width: int,
                 depth: int, band_dim: int = 24):
        super().__init__()
        self.band_emb = nn.Parameter(torch.randn(bands, band_dim) * 0.02)
        in_dim = latent + detail + 2 + 2 + band_dim   # +delta_xy +cell_hw +band
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(inplace=True)]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, detail: torch.Tensor,
                delta: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """latent [N,Cl] detail [N,Cd] delta [N,2] cell [N,2] -> [N,bands]."""
        n, bands = latent.shape[0], self.band_emb.shape[0]
        base = torch.cat([latent, detail, delta, cell], dim=-1)     # [N,D]
        base = base[:, None, :].expand(n, bands, base.shape[-1])
        emb = self.band_emb[None].expand(n, bands, -1).to(base.dtype)
        return self.net(torch.cat([base, emb], dim=-1)).squeeze(-1)  # [N,bands]


def make_coord(shape: Tuple[int, int], device, flatten: bool = True
               ) -> torch.Tensor:
    """Pixel-centre coordinates in [-1, 1]. Centres, not corners: using corners
    biases every interpolation by half a pixel."""
    coords = []
    for n in shape:
        r = 1.0 / n
        coords.append(torch.arange(n, device=device).float() * 2 * r - 1 + r)
    grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
    return grid.reshape(-1, 2) if flatten else grid


class ContinuumFusion(nn.Module):
    """HSI-MSI fusion as a continuous field, queried at arbitrary resolution."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.bands, cfg.msi_bands, cfg.width, cfg.enc_depth)
        self.detail = DetailEncoder(cfg.msi_bands, cfg.detail_width)
        # feature unfolding: a 3x3 neighbourhood gives the decoder local context
        latent = cfg.width * 9 if cfg.unfold else cfg.width
        self.decoder = SpectralCoordMLP(latent, cfg.detail_width, cfg.bands,
                                        cfg.mlp_width, cfg.mlp_depth,
                                        cfg.band_dim)
        self.unfold = cfg.unfold

    # ------------------------------------------------------------------ query
    def query(self, feat: torch.Tensor, detail: torch.Tensor,
              coord: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Sample the continuous field at `coord`.

        Local ensemble: each output point is decoded from its four neighbouring
        latent cells and blended by area weights. Decoding from the single
        nearest cell leaves visible blocking at the cell boundaries.
        """
        b, c, h, w = feat.shape
        if self.unfold:
            feat = F.unfold(feat, 3, padding=1).view(b, c * 9, h, w)

        feat_coord = make_coord((h, w), feat.device, flatten=False)
        feat_coord = feat_coord.permute(2, 0, 1)[None].expand(b, 2, h, w)

        rx, ry = 1.0 / h, 1.0 / w
        preds, areas = [], []
        for dx in (-1, 1):
            for dy in (-1, 1):
                c_ = coord.clone()
                c_[..., 0] += dx * rx + 1e-6
                c_[..., 1] += dy * ry + 1e-6
                c_.clamp_(-1 + 1e-6, 1 - 1e-6)
                grid = c_.flip(-1)[:, :, None, :]                  # [B,N,1,2]
                q_feat = F.grid_sample(feat, grid, mode="nearest",
                                       align_corners=False)[:, :, :, 0]
                q_coord = F.grid_sample(feat_coord, grid, mode="nearest",
                                        align_corners=False)[:, :, :, 0]
                q_feat = q_feat.permute(0, 2, 1)                   # [B,N,C]
                q_coord = q_coord.permute(0, 2, 1)                 # [B,N,2]
                rel = coord - q_coord
                rel[..., 0] *= h
                rel[..., 1] *= w
                # detail read at full resolution, not upsampled from the LR grid
                q_det = F.grid_sample(detail, coord.flip(-1)[:, :, None, :],
                                      mode="bilinear", align_corners=False
                                      )[:, :, :, 0].permute(0, 2, 1)
                n = coord.shape[1]
                pred = self.decoder(q_feat.reshape(b * n, -1),
                                    q_det.reshape(b * n, -1),
                                    rel.reshape(b * n, 2),
                                    cell.reshape(b * n, 2))
                preds.append(pred.view(b, n, -1))
                areas.append(torch.abs(rel[..., 0] * rel[..., 1]) + 1e-9)

        # diagonal swap: weight each corner by the area of the opposite rectangle
        areas = [areas[3], areas[2], areas[1], areas[0]]
        tot = sum(areas)
        out = sum(p * (a / tot)[..., None] for p, a in zip(preds, areas))
        return out                                                # [B,N,bands]

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor,
                out_hw: Optional[Tuple[int, int]] = None
                ) -> Dict[str, torch.Tensor]:
        """`out_hw` defaults to the MSI grid, but any resolution can be asked
        for - that is the whole point of the representation."""
        b = lr_hsi.shape[0]
        if out_hw is None:
            out_hw = (msi.shape[-2], msi.shape[-1])
        h, w = out_hw

        feat = self.encoder(lr_hsi, msi)
        detail = self.detail(msi)

        coord = make_coord((h, w), lr_hsi.device)[None].expand(b, h * w, 2)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2.0 / h
        cell[..., 1] *= 2.0 / w

        # the residual over a cheap interpolation keeps the MLP predicting a
        # correction rather than the full radiance, which trains far faster
        base = F.interpolate(lr_hsi, size=out_hw, mode="bicubic",
                             align_corners=False)
        pred = self.query(feat, detail, coord, cell)               # [B,N,bands]
        pred = pred.permute(0, 2, 1).reshape(b, -1, h, w)
        out = (base + pred).clamp(0, 1)
        return {"out": out, "residual": pred, "feat": feat.mean(dim=(2, 3))}

    def features(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        return self.encoder(lr_hsi, msi).mean(dim=(2, 3))


def build_model(cfg) -> ContinuumFusion:
    return ContinuumFusion(cfg)
