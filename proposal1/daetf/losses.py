"""Spectral-Physical Composite (SPC) loss.

    L = w1 Charbonnier + w2 SAM + w3 gradient + w4 (1 - SSIM)      [fidelity]
      + w5 || Down(Y) - LR ||  + w6 || SRF(Y) - MSI ||             [physics]
      + w7 MoE-balance + w8 ||G||_*  + w9 degradation-regression   [regularisers]
      + w10 MMD(source, target)                                    [domain]

Two properties matter for the research claim:

1. The SAM term optimises the metric on which every benchmarked baseline
   degrades under domain shift (SAM 2-7 deg in-domain vs 8-36 deg cross-domain),
   rather than optimising only the metric that already looks good.

2. The two physics terms need no ground truth. They are computable on any
   unseen scene, which is precisely what makes self-supervised test-time
   adaptation possible on a new sensor or dataset.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .degrade import FixedDegradation
from .metrics import ssim_torch


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Robust L1: differentiable at zero, less outlier-sensitive than L2."""
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean spectral angle in radians."""
    p, t = pred.flatten(2), target.flatten(2)
    num = (p * t).sum(dim=1)
    den = p.norm(dim=1) * t.norm(dim=1)
    cos = (num / den.clamp_min(eps)).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def spectral_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral gradient loss: penalises differences in adjacent-band derivatives.

    While SAM loss measures spectral angle globally, this loss ensures that the
    *shape* of the spectral curve (band-to-band transitions) is preserved. This
    directly combats the spectral distortion that was the worst metric (SAM > 14°).
    """
    # Compute spectral derivatives: difference between adjacent bands
    # pred/target shape: [B, C, H, W] where C is the number of spectral bands
    spec_grad_p = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    spec_grad_t = target[:, 1:, :, :] - target[:, :-1, :, :]
    return F.l1_loss(spec_grad_p, spec_grad_t)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multi-bandwidth RBF maximum mean discrepancy, with the bandwidth set from
    the median pairwise distance so it adapts to the feature scale."""
    z = torch.cat([x, y], dim=0)
    d = torch.cdist(z, z) ** 2
    n = x.shape[0]
    med = d.detach().flatten().median().clamp_min(1e-6)
    k = sum(torch.exp(-d / (med * s)) for s in (0.25, 0.5, 1.0, 2.0, 4.0))
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


class SPCLoss(nn.Module):
    def __init__(self, cfg: Config, srf: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.degrade = FixedDegradation.from_config(cfg)
        self.register_buffer("srf", srf)      # [bands, msi_bands]

    def apply_srf(self, x: torch.Tensor) -> torch.Tensor:
        """Project a hyperspectral cube through the recovered spectral response."""
        w = self.srf.to(x.dtype).t().reshape(self.srf.shape[1], self.srf.shape[0], 1, 1)
        return F.conv2d(x, w)

    def forward(self, out: Dict[str, torch.Tensor], target: torch.Tensor,
                lr_hsi: torch.Tensor, msi: torch.Tensor, model,
                deg_gt: Optional[torch.Tensor] = None,
                kernel: Optional[torch.Tensor] = None,
                tgt_feat: Optional[torch.Tensor] = None,
                supervised: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        pred = out["out"]
        logs: Dict[str, float] = {}
        total = pred.new_zeros(())

        if supervised:
            l_char = charbonnier(pred, target)
            l_sam = sam_loss(pred, target)
            l_grad = gradient_loss(pred, target)
            l_ssim = 1.0 - ssim_torch(pred.clamp(0, 1).float(), target.float())
            l_specgrad = spectral_gradient_loss(pred, target)
            total = (total + cfg.w_char * l_char + cfg.w_sam * l_sam
                     + cfg.w_grad * l_grad + cfg.w_ssim * l_ssim
                     + cfg.w_specgrad * l_specgrad)
            logs.update(char=l_char.item(), sam=l_sam.item(),
                        grad=l_grad.item(), ssim=l_ssim.item(),
                        specgrad=l_specgrad.item())

        # --- projective spectral embedding (v5): train on the metric that
        # fails.  The manifold is intensity-invariant and calibrated so that
        # L2 there approximates spectral angle, so these two terms make the
        # objective measure exactly what SAM measures - without SAM's
        # anti-parallel gradient problem and without any intensity signal.
        embed = getattr(model, "embed", None)
        if supervised and embed is not None:
            e_pred, _ = embed(pred)
            e_gt, _ = embed(target)
            l_embed = F.mse_loss(e_pred, e_gt)
            l_cal = embed.calibration_loss(torch.cat([pred, target], dim=0))
            total = total + cfg.w_embed * l_embed + cfg.w_cal * l_cal
            logs.update(embed=l_embed.item(), cal=l_cal.item())

        # --- physics: valid on any domain, with or without ground truth -------
        if cfg.use_physics or not supervised:
            l_spat = charbonnier(self.degrade(pred, kernel), lr_hsi)
            l_spec = charbonnier(self.apply_srf(pred), msi)
            total = total + cfg.w_spat * l_spat + cfg.w_spec * l_spec
            logs.update(spat=l_spat.item(), spec=l_spec.item())

        # --- regularisers -------------------------------------------------------
        if getattr(model, "moe", None) is not None:
            l_bal = model.moe.balance_loss()
            total = total + cfg.w_bal * l_bal
            logs["bal"] = l_bal.item()

        if supervised:
            if getattr(model, "tsse", None) is not None:
                l_rank = model.tsse.rank_penalty()
                total = total + cfg.w_rank * l_rank
                logs["rank"] = l_rank.item()
            if deg_gt is not None:
                l_deg = F.smooth_l1_loss(out["deg"].float(), deg_gt.float())
                total = total + cfg.w_deg * l_deg
                logs["deg"] = l_deg.item()
            if tgt_feat is not None:
                l_mmd = mmd_rbf(out["feat"].float(), tgt_feat.float())
                total = total + cfg.w_mmd * l_mmd
                logs["mmd"] = l_mmd.item()

        logs["total"] = total.item()
        return total, logs
