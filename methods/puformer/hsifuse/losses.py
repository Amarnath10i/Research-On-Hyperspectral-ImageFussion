"""Differentiable SAM and SSIM losses on (N, C, H, W) batches, computed in float32.

They target the two metrics where plain L1 training trails the literature:
    SAM  - mean spectral angle in radians
    SSIM - band-wise Gaussian SSIM with data range = each sample's max, the same
           definition as the headline metric (stricter on dark images)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sam_loss(x, gt, eps=1e-8):
    x, gt = x.float(), gt.float()
    cos = (x * gt).sum(1) / (x.norm(dim=1) * gt.norm(dim=1) + eps)
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)).mean()


def ssim_loss(x, gt, win=11, sigma=1.5):
    x, gt = x.float(), gt.float()
    n, c = gt.shape[:2]
    ax = torch.arange(win, dtype=gt.dtype, device=gt.device) - (win - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].expand(c, 1, win, win)
    f = lambda t: F.conv2d(t, k, groups=c)
    L = gt.amax(dim=(1, 2, 3), keepdim=True).clamp_min(0.05)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu1, mu2 = f(gt), f(x)
    s11, s22, s12 = f(gt * gt) - mu1 ** 2, f(x * x) - mu2 ** 2, f(gt * x) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return 1 - m.mean()
