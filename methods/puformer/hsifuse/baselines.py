"""Training-free baselines used to calibrate our protocol against published tables.

GSA (Aiazzi et al., TGRS 2007), in the HS-MS form of Yokoya's HSMS fusion
toolbox: every HS band is sharpened with the MS band it overlaps most. That band's
low-resolution intensity is synthesised by regressing the (degraded) MS band on
the LR-HSI bands.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .ops import Degradation


@torch.no_grad()
def gsa(yh: torch.Tensor, ym: torch.Tensor, deg: Degradation) -> torch.Tensor:
    """yh (N,B,h,w), ym (N,M,H,W) -> (N,B,H,W)."""
    s = deg.scale
    up = F.interpolate(yh, scale_factor=s, mode="bicubic", align_corners=False)
    group = deg.srf.argmax(1)                         # MS band each HS band belongs to
    out = up.clone()
    for n in range(yh.shape[0]):
        hs_lr = yh[n].flatten(1).double()                        # B x hw
        ms_lr = deg.D(ym[n:n + 1], pad_mode="reflect")[0].flatten(1).double()  # M x hw
        A = torch.cat([hs_lr, torch.ones_like(hs_lr[:1])], 0).T  # hw x (B+1)
        up_n = up[n].flatten(1).double()
        A_up = torch.cat([up_n, torch.ones_like(up_n[:1])], 0).T
        for j in range(ym.shape[1]):
            w = torch.linalg.lstsq(A, ms_lr[j:j + 1].T).solution  # intensity weights
            I = (A_up @ w).T[0]                                   # HR intensity from upsampled HS
            P = ym[n, j].flatten().double()
            P = (P - P.mean()) / P.std() * I.std() + I.mean()     # histogram-match detail source
            idx = (group == j).nonzero().flatten()
            Ic = I - I.mean()
            g = ((up_n[idx] - up_n[idx].mean(1, keepdim=True)) @ Ic) / (Ic @ Ic)
            out[n, idx] = (up_n[idx] + g[:, None] * (P - I)[None]).view(len(idx), *up.shape[-2:]).float()
    return out.clamp_min(0)
