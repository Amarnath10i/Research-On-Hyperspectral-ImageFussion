"""Shared loss terms and the common fidelity+physics objective.

Every proposal optimises the same base objective so that architectural
differences are not confounded with differences in what was optimised. A
proposal that needs extra terms (a rank penalty, an expert-balance penalty,
a per-stage supervision) adds them through `extra_terms`, which keeps the
shared part identical and the additions explicit.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .degrade import FixedDegradation
from .metrics import ssim_torch


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Robust L1: differentiable at zero, less outlier-sensitive than L2."""
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean spectral angle in radians - the metric every baseline loses under
    domain shift, optimised directly."""
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


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multi-bandwidth RBF MMD, bandwidth set from the median pairwise distance."""
    z = torch.cat([x, y], dim=0)
    d = torch.cdist(z, z) ** 2
    n = x.shape[0]
    med = d.detach().flatten().median().clamp_min(1e-6)
    k = sum(torch.exp(-d / (med * s)) for s in (0.25, 0.5, 1.0, 2.0, 4.0))
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


class FusionLoss(nn.Module):
    """Fidelity + physics, shared by every proposal.

        L = w_char C + w_sam SAM + w_grad G + w_ssim (1-SSIM)     [fidelity]
          + w_spat ||Down(Y) - LR|| + w_spec ||SRF(Y) - MSI||     [physics]
          + extra_terms(...)                                      [per-proposal]

    The physics terms need no ground truth, so `supervised=False` yields an
    objective that is computable on an unlabelled scene from an unseen sensor.
    That is what makes test-time adaptation and the fully self-supervised
    proposal possible with the same code.
    """

    def __init__(self, cfg, srf: torch.Tensor,
                 extra_terms: Optional[Callable] = None):
        super().__init__()
        self.cfg = cfg
        self.degrade = FixedDegradation(cfg.scale, ksize=cfg.blur_ksize,
                                        sigma=cfg.eval_sigma)
        self.register_buffer("srf", srf)          # [bands, msi_bands]
        self.extra_terms = extra_terms

    def apply_srf(self, x: torch.Tensor) -> torch.Tensor:
        w = self.srf.to(x.dtype).t().reshape(self.srf.shape[1], self.srf.shape[0], 1, 1)
        return F.conv2d(x, w)

    def forward(self, out: Dict[str, torch.Tensor], target: Optional[torch.Tensor],
                lr_hsi: torch.Tensor, msi: torch.Tensor, model: nn.Module,
                kernel: Optional[torch.Tensor] = None,
                supervised: bool = True,
                **kw) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        pred = out["out"]
        logs: Dict[str, float] = {}
        total = pred.new_zeros(())

        if supervised and target is not None:
            l_char = charbonnier(pred, target)
            l_sam = sam_loss(pred, target)
            l_grad = gradient_loss(pred, target)
            l_ssim = 1.0 - ssim_torch(pred.clamp(0, 1).float(), target.float())
            total = (total + cfg.w_char * l_char + cfg.w_sam * l_sam
                     + cfg.w_grad * l_grad + cfg.w_ssim * l_ssim)
            logs.update(char=l_char.item(), sam=l_sam.item(),
                        grad=l_grad.item(), ssim=l_ssim.item())

        if cfg.use_physics or not supervised:
            l_spat = charbonnier(self.degrade(pred, kernel), lr_hsi)
            l_spec = charbonnier(self.apply_srf(pred), msi)
            total = total + cfg.w_spat * l_spat + cfg.w_spec * l_spec
            logs.update(spat=l_spat.item(), spec=l_spec.item())

        if self.extra_terms is not None:
            extra, extra_logs = self.extra_terms(
                out=out, target=target, lr_hsi=lr_hsi, msi=msi, model=model,
                cfg=cfg, supervised=supervised, **kw)
            total = total + extra
            logs.update(extra_logs)

        logs["total"] = total.item()
        return total, logs
