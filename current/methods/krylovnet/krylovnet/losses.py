"""KrylovNet loss: physics + spectral + reconstruction + residual regulariser.

The physics and spectral terms use the *same* operator as the model, so they
measure exactly what the unrolled solver is trying to satisfy (A x ~ b), and
converge together with the GMRES residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class KrylovLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, pred: dict, gt: torch.Tensor, lr: torch.Tensor,
                msi: torch.Tensor, kernel: torch.Tensor, model) -> dict:
        out = pred["out"]
        op = model.op
        l_phys = F.mse_loss(op.D(out, kernel), lr)
        l_spec = F.mse_loss(op.S(out, model.srf), msi)
        l_recon = F.l1_loss(out, gt)
        l_res = pred["residuals"][-1].mean()
        total = (self.cfg.w_phys * l_phys + self.cfg.w_spec * l_spec
                 + self.cfg.w_recon * l_recon + self.cfg.w_res * l_res)
        return {"loss": total, "phys": l_phys, "spec": l_spec,
                "recon": l_recon, "res": l_res.detach()}