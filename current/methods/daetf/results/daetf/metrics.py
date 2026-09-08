"""One metric implementation, shared by every method and both datasets.

The v1 benchmark computed PSNR/SSIM/SAM/ERGAS separately inside each of the 20
notebooks, with different data ranges, different normalisations and ERGAS scale
factors that did not always match the actual downsampling factor. Those numbers
were therefore not comparable across methods. Everything here is fixed:

  * PSNR uses a constant data_range (default 1.0), never the per-image maximum,
    which otherwise inflates scores on dark scenes.
  * SSIM is Gaussian-windowed (11x11, sigma 1.5) and averaged over bands.
  * SAM is reported in degrees, ignoring degenerate zero-spectra pixels.
  * ERGAS receives the true scale factor of the experiment.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def _gauss_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
               size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Gaussian-windowed SSIM, averaged over channels. Differentiable."""
    c = pred.shape[1]
    win = _gauss_window(size, sigma, pred.device, pred.dtype).expand(c, 1, size, size)
    mu1 = F.conv2d(pred, win, padding=size // 2, groups=c)
    mu2 = F.conv2d(target, win, padding=size // 2, groups=c)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, win, padding=size // 2, groups=c) - mu1s
    s2 = F.conv2d(target * target, win, padding=size // 2, groups=c) - mu2s
    s12 = F.conv2d(pred * target, win, padding=size // 2, groups=c) - mu12
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return m.mean()


def _hwc(x: np.ndarray) -> np.ndarray:
    return x if x.shape[-1] <= 64 else np.transpose(x, (1, 2, 0))


def metric_psnr(pred: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - ref) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(data_range ** 2 / mse))


def metric_sam(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    p, r = _hwc(pred).reshape(-1, pred.shape[-1]), _hwc(ref).reshape(-1, ref.shape[-1])
    cos = (p * r).sum(1) / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(r, axis=1), eps)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return float(np.mean(ang[np.isfinite(ang)]))


def metric_ergas(pred: np.ndarray, ref: np.ndarray, scale: int, eps: float = 1e-8) -> float:
    p, r = _hwc(pred), _hwc(ref)
    rmse = np.sqrt(np.mean((p - r) ** 2, axis=(0, 1)))
    mu = np.maximum(np.mean(r, axis=(0, 1)), eps)
    return float(100.0 / scale * np.sqrt(np.mean((rmse / mu) ** 2)))


def metric_ssim(pred: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    p = torch.from_numpy(np.ascontiguousarray(_hwc(pred).transpose(2, 0, 1)))[None].float()
    r = torch.from_numpy(np.ascontiguousarray(_hwc(ref).transpose(2, 0, 1)))[None].float()
    return float(ssim_torch(p, r, data_range=data_range))


def evaluate_arrays(pred: np.ndarray, ref: np.ndarray, scale: int) -> Dict[str, float]:
    """The four reported metrics for one scene."""
    return {
        "psnr": metric_psnr(pred, ref),
        "ssim": metric_ssim(pred, ref),
        "sam": metric_sam(pred, ref),
        "ergas": metric_ergas(pred, ref, scale),
    }
