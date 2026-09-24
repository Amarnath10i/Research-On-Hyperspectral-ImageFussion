"""ManifoldFlow loss: rectified flow matching + optional straightness regulariser.

L = w_flow E_t ||v_theta(y_t, m, t) - u||^2
  + w_straight E_{t,t'} ||v_theta(y_t, m, t) - v_theta(y_t, m, t')||^2  (same y_t)
  + w_recon L1(y_sampled, gt)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class FlowLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, pred: dict, gt: torch.Tensor, lr: torch.Tensor,
                msi: torch.Tensor, kernel: torch.Tensor, model) -> dict:
        u = pred["target"]
        v = pred["velocity"]
        l_flow = F.mse_loss(v, u)
        l_straight = torch.zeros((), device=v.device)
        if self.cfg.straightness_reg:
            t2 = torch.rand_like(pred["t"])
            v2 = model.velocity_field(pred["y0"]
                                      + pred["t"].unsqueeze(-1).unsqueeze(-1)
                                      .unsqueeze(-1) * u, msi, t2, kernel)
            l_straight = F.mse_loss(v2, v)
        # auxiliary: sampled reconstruction
        out = model.sample(lr, msi, kernel, steps=2)
        l_recon = F.l1_loss(out, gt)
        total = (self.cfg.w_flow * l_flow
                 + self.cfg.w_straight * l_straight
                 + self.cfg.w_recon * l_recon)
        return {"loss": total, "flow": l_flow, "straight": l_straight,
                "recon": l_recon}