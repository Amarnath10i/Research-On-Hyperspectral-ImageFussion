"""DAETF-Net v3: Degradation-Conditioned Spectral-Spatial Routing Network.

Architecture identity: Adaptive Spectral-Causal Routing (ASCR)

Flow:
    (LR-HSI, MSI) -> degradation code d -------------------------+
    LR-HSI -> back-projection upsampler -> coarse HR y0           |
    y0  -> equivariant feature extractor -> FiLM <----------------+
    MSI -> encoder ----------------------> FiLM <----------------+
    SpectralDisagreementField(f_hsi, f_msi) -> delta, D           |
    TSSE(f_hsi, f_msi) -> z_fused -> channel attention            |
    DegradationConditionedMoE(z, d, D) -> routed expert output    |
    -> wavelet refinement -> RDB reconstruction                   |
    -> residual on y0 -> Ŷ                                        |

What v3 adds over v2:
  1. SpectralDisagreementField: detects WHERE modalities disagree
  2. DegradationConditionedMoE: routes using disagreement + degradation code
  3. Semantic experts: spectral / edge / texture / correction (not generic conv)

Every module can be swapped for a matched control arm through Config ablation
switches, so each contribution is measured against a like-for-like baseline.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .modules import (BackProjectionUpsampler, BicubicUpsampler,
                      ChannelAttention, DegradationEncoder,
                      DegradationConditionedMoE,
                      EquivariantFeatureExtractor, FiLM,
                      FrequencyDomainRefinement,
                      PlainFeatureExtractor,
                      ResidualDenseBlock,
                      SpectralDisagreementField,
                      TensorSpectralSpatialEncoder)
from .nullspace import RangeNullProjector, kernel_from_params


class DAETFNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.bands is None or cfg.msi_bands is None:
            raise ValueError("Config.bands/msi_bands are unset - call cfg.resolve() "
                             "or pass them explicitly before building the model")
        self.cfg = cfg
        c, b, m = cfg.width, cfg.bands, cfg.msi_bands

        # --- range/null decomposition (v4) --------------------------------
        # When enabled this REPLACES the upsampler on the output path: the
        # pseudo-inverse is already the data-optimal estimate, so a learned
        # upsampler would only be re-deriving what the projector gives exactly.
        self.projector = (
            RangeNullProjector(cfg.scale, ksize=cfg.blur_ksize,
                               sigma=cfg.eval_sigma, cg_steps=cfg.cg_steps,
                               ridge=cfg.ridge)
            if cfg.use_nullspace else None
        )
        # Only built when the projector is off, otherwise its weights would sit
        # in the parameter count receiving no gradient - inflating the reported
        # model size and making the ablation compare unequal budgets.
        self.upsampler = None if cfg.use_nullspace else (
            BackProjectionUpsampler(b, cfg.scale, width=c, iters=cfg.bp_iters)
            if cfg.use_backprojection else BicubicUpsampler(b, cfg.scale, width=c)
        )

        # --- degradation encoder ---
        self.deg = DegradationEncoder(b, m, code=cfg.code_dim)

        # --- equivariant HSI feature extractor ---
        self.efe = (
            EquivariantFeatureExtractor(b, cfg.equi_width, c, depth=cfg.equi_depth)
            if cfg.use_equivariant
            else PlainFeatureExtractor(b, cfg.equi_width, c, depth=cfg.equi_depth)
        )

        # --- MSI encoder ---
        self.msi_enc = nn.Sequential(
            nn.Conv2d(m, c, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, c, 3, 1, 1),
        )

        # --- degradation conditioning (FiLM) ---
        self.film_h = FiLM(cfg.code_dim, c)
        self.film_m = FiLM(cfg.code_dim, c)

        # --- cross-modal disagreement field (NEW v3) ---
        self.disagree = (SpectralDisagreementField(c)
                         if cfg.use_disagreement else None)

        # --- Tucker spectral-spatial fusion ---
        self.tsse = (TensorSpectralSpatialEncoder(c, c, c, rank=cfg.rank)
                     if cfg.use_tsse else None)
        self.concat_fuse = None if cfg.use_tsse else nn.Sequential(
            nn.Conv2d(c * 2, c, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, c, 3, 1, 1)
        )

        # --- channel attention after Tucker fusion ---
        self.channel_attn = ChannelAttention(c, reduction=8) if cfg.use_tsse else None

        # --- degradation-conditioned MoE (NEW v3, replaces RegionAwareMoE) ---
        if cfg.use_moe:
            self.moe = DegradationConditionedMoE(
                channels=c,
                code_dim=cfg.code_dim,
                disagree_ch=3,       # fixed: D is 3-channel
                topk=cfg.topk,
            )
        else:
            self.moe = None
        self.plain_block = None if cfg.use_moe else nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, c, 3, 1, 1)
        )

        # --- wavelet frequency refinement ---
        self.fdrm = FrequencyDomainRefinement(c) if cfg.use_fdrm else None

        # --- reconstruction head ---
        self.rdb = ResidualDenseBlock(c, growth=32, n_layers=3)
        self.recon = nn.Conv2d(c, b, 3, 1, 1)

    # ------------------------------------------------------------------
    def _trunk(self, lr_hsi: torch.Tensor, msi: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                          torch.Tensor, Optional[torch.Tensor]]:
        """Core forward pass. Returns (y0, z_fused, deg_params, d_code, D_field)."""
        code, deg_params = self.deg(lr_hsi, msi)
        if not self.cfg.use_degradation_code:
            code = torch.zeros_like(code)

        # Coarse estimate. With the decomposition on, this is the range
        # component D_pinv(X) - the part of the solution the observation
        # determines - computed in closed form rather than learned.
        if self.projector is not None:
            hw = (msi.shape[-2], msi.shape[-1])
            kernel = kernel_from_params(deg_params, self.cfg.blur_ksize)
            y0 = self.projector.pinv(lr_hsi, hw, kernel)
        else:
            kernel = None
            y0 = self.upsampler(lr_hsi)

        # Feature extraction with degradation conditioning
        fh = self.film_h(self.efe(y0), code)      # [B, C, H, W]
        fm = self.film_m(self.msi_enc(msi), code)  # [B, C, H, W]

        # Spectral disagreement field (v3 new)
        if self.disagree is not None:
            delta, D = self.disagree(fh, fm)       # D: [B, 3, H, W]
        else:
            delta, D = None, torch.zeros(fh.shape[0], 3, fh.shape[2], fh.shape[3],
                                         device=fh.device, dtype=fh.dtype)

        # Tucker cross-modal fusion
        z = (self.tsse(fh, fm) if self.tsse is not None
             else self.concat_fuse(torch.cat([fh, fm], dim=1)))

        # Channel attention
        if self.channel_attn is not None:
            z = self.channel_attn(z)

        return y0, z, deg_params, code, D, kernel

    def features(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        """Pooled bottleneck features for MMD domain-alignment term."""
        return self._trunk(lr_hsi, msi)[1].mean(dim=(2, 3))

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> Dict[str, torch.Tensor]:
        y0, z, deg_params, code, D, kernel = self._trunk(lr_hsi, msi)

        # Degradation-conditioned expert routing (v3 new)
        if self.moe is not None:
            z = self.moe(z, code, D)
        else:
            z = self.plain_block(z)

        # Wavelet refinement
        if self.fdrm is not None:
            z = self.fdrm(z)

        # Dense reconstruction head
        z = self.rdb(z)
        residual = self.recon(z)

        if self.projector is not None:
            # Y_hat = D_pinv(X) + P_perp(F_theta(X, M)).
            # Projecting the residual onto the null space of D means the
            # network literally cannot alter the data-determined component:
            # D(Y_hat) = X holds for whatever the network produces, so the
            # observation is satisfied by construction rather than by penalty.
            null_part = self.projector.project_null(residual, kernel)
            out = y0 + null_part
        else:
            null_part = residual
            out = y0 + residual

        result = {
            "out": out,
            "coarse": y0,
            "range": y0,           # data-determined component
            "null": null_part,     # learned component, invisible to D
            "deg": deg_params,
            "kernel": kernel,
            "feat": z.mean(dim=(2, 3)),
        }
        # Attach disagreement field and expert usage for diagnostics/visualisation
        if D is not None:
            result["disagree"] = D
        if self.moe is not None and self.moe.last_gate is not None:
            result["expert_gate"] = self.moe.last_gate.detach()
        return result

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def expert_usage_summary(self) -> Optional[Dict[str, float]]:
        """Per-expert mean activation for diagnostic logging."""
        if self.moe is None:
            return None
        usage = self.moe.expert_usage()
        if usage is None:
            return None
        names = ["spectral", "edge", "texture", "correction"]
        return {n: float(u) for n, u in zip(names, usage)}
