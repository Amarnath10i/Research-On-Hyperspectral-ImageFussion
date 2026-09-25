"""Quality metrics. Inputs are (C, H, W) tensors in [0, 1] (global-max normalised).

Headline PSNR (`PSNR`): 10 log10(1 / MSE) over the whole cube, i.e. peak = the dataset's
global max (data are divided by it). This matches TIP'26 Table IV: its RMSE column is
in raw DN, and 20 log10(max_DN / RMSE_DN) reproduces its PSNR column to within ~1 dB.
Also reported: `PSNR_psrt` (peak = each test image's own max, PSRT metrics.py),
`MPSNR_peak1` (band-wise mean, peak 1) and `SSIM_psrt` (PSRT cal_ssim.py: fixed
C1 = 0.01^2, C2 = 0.03^2, i.e. data range 1, zero padding). Headline `SSIM` uses data
range = the image's own max, which is stricter on dark images.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DN_SCALE = 15133.0  # max DN of the central 2048x2048 crop (our normaliser), for RMSE in DN


def psnr(gt, x):
    return float(10 * torch.log10(1.0 / torch.mean((gt - x) ** 2)))


def psnr_psrt(gt, x):
    mse = torch.mean((gt - x) ** 2)
    return float(10 * torch.log10(gt.max() ** 2 / mse))


def mpsnr_peak1(gt, x):
    mse = torch.mean((gt - x) ** 2, dim=(1, 2)).clamp_min(1e-12)
    return float(torch.mean(10 * torch.log10(1.0 / mse)))


def sam(gt, x, eps=1e-8):
    g, f = gt.flatten(1), x.flatten(1)
    cos = (g * f).sum(0) / (g.norm(dim=0) * f.norm(dim=0) + eps)
    return float(torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean())


def ergas(gt, x, ratio=4):
    rmse = torch.sqrt(torch.mean((gt - x) ** 2, dim=(1, 2)))
    mean = gt.mean(dim=(1, 2)).clamp_min(1e-12)
    return float(100 / ratio * torch.sqrt(torch.mean((rmse / mean) ** 2)))


def rmse_dn(gt, x):
    return float(torch.sqrt(torch.mean((gt - x) ** 2))) * DN_SCALE


def cc(gt, x):
    g = gt.flatten(1) - gt.flatten(1).mean(1, keepdim=True)
    f = x.flatten(1) - x.flatten(1).mean(1, keepdim=True)
    return float(((g * f).sum(1) / (g.norm(dim=1) * f.norm(dim=1) + 1e-12)).mean())


def ssim(gt, x, win=11, sigma=1.5):
    """Band-wise Gaussian SSIM, data range = max(GT), averaged over bands."""
    L = gt.max()
    ax = torch.arange(win, dtype=gt.dtype, device=gt.device) - (win - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].expand(gt.shape[0], 1, win, win)
    f = lambda t: F.conv2d(t[None], k, groups=gt.shape[0])[0]
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu1, mu2 = f(gt), f(x)
    s11, s22, s12 = f(gt * gt) - mu1 ** 2, f(x * x) - mu2 ** 2, f(gt * x) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def ssim_psrt(gt, x, win=11, sigma=1.5):
    """SSIM exactly as PSRT's cal_ssim.py: data range 1, zero padding, mean over bands and pixels."""
    ax = torch.arange(win, dtype=gt.dtype, device=gt.device) - win // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].expand(gt.shape[0], 1, win, win)
    f = lambda t: F.conv2d(t[None], k, padding=win // 2, groups=gt.shape[0])[0]
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = f(gt), f(x)
    s11, s22, s12 = f(gt * gt) - mu1 ** 2, f(x * x) - mu2 ** 2, f(gt * x) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


ALL = dict(PSNR=psnr, PSNR_psrt=psnr_psrt, MPSNR_peak1=mpsnr_peak1, SSIM=ssim, SSIM_psrt=ssim_psrt, SAM=sam,
           ERGAS=ergas, RMSE_DN=rmse_dn, CC=cc)


def evaluate(gts, preds):
    """Mean of every metric over a list/stack of (C, H, W) images (float64 for accuracy)."""
    out = {k: 0.0 for k in ALL}
    for g, p in zip(gts, preds):
        g, p = g.double(), p.double().clamp(0, 1)
        for k, fn in ALL.items():
            out[k] += fn(g, p)
    n = len(gts)
    return {k: v / n for k, v in out.items()}
