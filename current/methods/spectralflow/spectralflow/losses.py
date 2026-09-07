"""Losses for SpectralFlow.

Training objective is *denoising score matching*: predict the Gaussian noise
that was added to a clean HR hyperspectral patch,

    L_sm = E_{t, eps} || eps_theta( sqrt(a_t) y_0 + sqrt(1-a_t) eps, M, d, t )
                     - eps ||^2 .

This teaches the spectral-spatial manifold prior used at sampling time.  The
null-space projection that guarantees observation consistency is applied in the
sampler, not here - score matching only needs the model to know what a clean
hyperspectral cube looks like given the MSI guide and the degradation.

Two auxiliary terms make the conditioning load-bearing rather than decorative:
  * degradation regression (the code must actually predict blur/noise/SRF), and
  * a spectral-consistency term (SRF(y) should match the MSI guide), so the
    guide carries the spatial detail that the null space must be filled with.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nullspace import decode_degradation_params
from .sampler import LinearNoiseSchedule


def score_matching_loss(score_net: nn.Module, schedule: LinearNoiseSchedule,
                        gt: torch.Tensor, msi: torch.Tensor,
                        code: torch.Tensor,
                        t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and true Gaussian noise."""
    a = schedule.a[t].to(gt.dtype)
    sqrt_a = a.sqrt().reshape(-1, 1, 1, 1)
    sqrt_1ma = (1 - a).sqrt().reshape(-1, 1, 1, 1)
    y_t = sqrt_a * gt + sqrt_1ma * eps
    eps_pred = score_net(y_t, msi, code, t)
    return F.mse_loss(eps_pred, eps)


class ScoreMatchingLoss(nn.Module):
    def __init__(self, num_timesteps: int = 200, beta_start: float = 1e-4,
                 beta_end: float = 0.02, w_deg: float = 0.05,
                 w_spec: float = 0.0):
        super().__init__()
        self.schedule = LinearNoiseSchedule(num_timesteps, beta_start, beta_end)
        self.w_deg = w_deg
        self.w_spec = w_spec

    def forward(self, model, gt: torch.Tensor, msi: torch.Tensor,
                lr_hsi: torch.Tensor, kernel: torch.Tensor,
                deg_gt: torch.Tensor,
                srf: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the score-matching loss plus auxiliary terms.

        Args:
            model:   the trained stack - must expose ``score_net``, ``deg_head``
                     and ``code_dim``.  The degradation head maps (lr, msi) ->
                     (code, raw_params).
            gt:      [B, C, H, W] clean HR hyperspectral patch.
            msi:     [B, m, H, W] HR multispectral guide.
            lr_hsi:  [B, C, h, w] LR hyperspectral observation.
            kernel:  [B, k, k] simulated kernel (informational).
            deg_gt:  [B, 5] true physical degradation parameters.
        """
        self.schedule = self.schedule.to(gt.device)
        b = gt.shape[0]
        t = torch.randint(1, self.schedule.T + 1, (b,), device=gt.device)
        eps = torch.randn_like(gt)

        code, raw = model.deg_head(lr_hsi, msi)
        l_sm = score_matching_loss(model.score_net, self.schedule, gt, msi,
                                   code, t, eps)

        total = l_sm
        logs = {"sm": l_sm.item()}

        # degradation regression: make the code mean something physical
        if deg_gt is not None and self.w_deg > 0:
            l_deg = F.smooth_l1_loss(decode_degradation_params(raw), deg_gt)
            total = total + self.w_deg * l_deg
            logs["deg"] = l_deg.item()

        # spectral consistency of the guide: SRF(y_0) should explain the MSI.
        # A little of this at training time keeps the MSI channel load-bearing.
        if srf is not None and self.w_spec > 0:
            sr = srf.to(gt.dtype).t().reshape(srf.shape[1], srf.shape[0], 1, 1)
            pred_msi = F.conv2d(gt, sr)
            l_spec = F.l1_loss(pred_msi, msi)
            total = total + self.w_spec * l_spec
            logs["spec"] = l_spec.item()

        logs["total"] = total.item()
        return total, logs