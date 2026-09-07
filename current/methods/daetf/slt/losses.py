"""SLT losses: the transport objective is metric-aligned by construction."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .manifold import geodesic_distance, l2_normalize, log_map


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dp = torch.abs(pred[..., 1:, :] - pred[..., :-1, :]).mean()
    dq = torch.abs(pred[..., :, 1:] - pred[..., :, :-1]).mean()
    tp = torch.abs(target[..., 1:, :] - target[..., :-1, :]).mean()
    tq = torch.abs(target[..., :, 1:] - target[..., :, :-1]).mean()
    return dp + dq - tp - tq


class SLTLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, pred: Dict[str, torch.Tensor],
                gt: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = pred["out"]
        logs: Dict[str, float] = {}

        if self.cfg.manifold:
            v = pred["v"]
            dir_gt = l2_normalize(gt)
            v_gt = log_map(pred["dir0"], dir_gt)       # isometry: ||v_gt|| = SAM
            l_geo = F.mse_loss(v, v_gt)
            l_sam = geodesic_distance(l2_normalize(out), dir_gt).mean()
            l_recon = F.l1_loss(out, gt)
            total = (self.cfg.w_geo * l_geo
                     + self.cfg.w_sam * l_sam
                     + self.cfg.w_recon * l_recon)
            logs.update({"geo": l_geo.item(), "sam": l_sam.item(),
                         "recon": l_recon.item()})
        else:
            l_char = F.l1_loss(out, gt)
            l_sam = geodesic_distance(l2_normalize(out),
                                      l2_normalize(gt)).mean()
            l_grad = _gradient_loss(out, gt)
            total = (self.cfg.w_recon * l_char
                     + self.cfg.w_sam * l_sam
                     + self.cfg.w_grad * l_grad)
            logs.update({"char": l_char.item(), "sam": l_sam.item(),
                         "grad": l_grad.item()})

        logs["total"] = total.item()
        return total, logs