"""DAETF-Net: the assembled network.

Flow:
    (LR-HSI, MSI) -> degradation code -----------------------------+
    LR-HSI -> back-projection upsampler -> coarse HR estimate y0   |
    y0  -> equivariant feature extractor -> FiLM <-----------------+
    MSI -> encoder ----------------------> FiLM <------------------+
    (f_hsi, f_msi) -> Tucker interaction -> region-aware MoE -> wavelet refinement
    -> residual reconstruction, added to y0

Every module can be swapped for a matched control arm through the Config
ablation switches, so each contribution is measured against a like-for-like
baseline rather than against its own absence.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import Config
from .modules import (BackProjectionUpsampler, BicubicUpsampler, DegradationEncoder,
                      EquivariantFeatureExtractor, FiLM, FrequencyDomainRefinement,
                      PlainFeatureExtractor, RegionAwareMoE,
                      TensorSpectralSpatialEncoder)


class DAETFNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.bands is None or cfg.msi_bands is None:
            raise ValueError("Config.bands/msi_bands are unset - call cfg.resolve() "
                             "or pass them explicitly before building the model")
        self.cfg = cfg
        c, b, m = cfg.width, cfg.bands, cfg.msi_bands

        self.upsampler = (
            BackProjectionUpsampler(b, cfg.scale, width=c, iters=cfg.bp_iters)
            if cfg.use_backprojection else BicubicUpsampler(b, cfg.scale, width=c)
        )
        self.deg = DegradationEncoder(b, m, code=cfg.code_dim)
        self.efe = (
            EquivariantFeatureExtractor(b, cfg.equi_width, c, depth=cfg.equi_depth)
            if cfg.use_equivariant
            else PlainFeatureExtractor(b, cfg.equi_width, c, depth=cfg.equi_depth)
        )
        self.msi_enc = nn.Sequential(
            nn.Conv2d(m, c, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, c, 3, 1, 1),
        )
        self.film_h = FiLM(cfg.code_dim, c)
        self.film_m = FiLM(cfg.code_dim, c)

        self.tsse = (TensorSpectralSpatialEncoder(c, c, c, rank=cfg.rank)
                     if cfg.use_tsse else None)
        self.concat_fuse = None if cfg.use_tsse else nn.Sequential(
            nn.Conv2d(c * 2, c, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, c, 3, 1, 1)
        )
        self.moe = (RegionAwareMoE(c, experts=cfg.experts, topk=cfg.topk)
                    if cfg.use_moe else None)
        self.plain_block = None if cfg.use_moe else nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, c, 3, 1, 1)
        )
        self.fdrm = FrequencyDomainRefinement(c) if cfg.use_fdrm else None
        self.recon = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, b, 3, 1, 1)
        )

    def _trunk(self, lr_hsi: torch.Tensor, msi: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code, deg_params = self.deg(lr_hsi, msi)
        if not self.cfg.use_degradation_code:
            code = torch.zeros_like(code)
        y0 = self.upsampler(lr_hsi)
        fh = self.film_h(self.efe(y0), code)
        fm = self.film_m(self.msi_enc(msi), code)
        z = (self.tsse(fh, fm) if self.tsse is not None
             else self.concat_fuse(torch.cat([fh, fm], dim=1)))
        return y0, z, deg_params

    def features(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        """Pooled bottleneck features, used by the MMD domain-alignment term."""
        return self._trunk(lr_hsi, msi)[1].mean(dim=(2, 3))

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> Dict[str, torch.Tensor]:
        y0, z, deg_params = self._trunk(lr_hsi, msi)
        z = self.moe(z) if self.moe is not None else self.plain_block(z)
        if self.fdrm is not None:
            z = self.fdrm(z)
        out = y0 + self.recon(z)
        return {"out": out, "coarse": y0, "deg": deg_params, "feat": z.mean(dim=(2, 3))}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
