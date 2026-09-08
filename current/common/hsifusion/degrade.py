"""Forward observation model: blur, decimation and the fixed evaluation operator.

Keeping the degradation differentiable is what allows the spatial-consistency
term of the loss to be back-propagated through, and therefore what allows
self-supervised adaptation on a domain with no ground truth.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:  # pragma: no cover
    from .config import Config


def gaussian_kernel2d(ksize: int, sx: float, sy: float, theta: float) -> torch.Tensor:
    """Rotated anisotropic Gaussian blur kernel, normalised to sum 1."""
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    xr = xx * cos_t + yy * sin_t
    yr = -xx * sin_t + yy * cos_t
    k = torch.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return k / k.sum().clamp_min(1e-12)


def blur_downsample(x: torch.Tensor, kernel: torch.Tensor, scale: int) -> torch.Tensor:
    """Apply a per-sample blur kernel then decimate. Differentiable.

    x      : [B, C, H, W]
    kernel : [k, k] (shared) or [B, k, k] (one kernel per sample)
    """
    b, c, _, _ = x.shape
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).expand(b, -1, -1)
    k = kernel.shape[-1]
    pad = k // 2
    # fold the batch into the channel axis so each sample keeps its own kernel
    w = kernel.to(x.dtype).reshape(b, 1, 1, k, k).expand(b, c, 1, k, k).reshape(b * c, 1, k, k)
    xr = x.reshape(1, b * c, *x.shape[-2:])
    xr = F.pad(xr, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(xr, w, groups=b * c)
    out = out.reshape(b, c, *out.shape[-2:])
    return out[..., ::scale, ::scale].contiguous()


class FixedDegradation(nn.Module):
    """Non-learnable blur+decimate: builds the evaluation LR input and backs the
    spatial-consistency loss."""

    def __init__(self, scale: int, ksize: int = 9, sigma: float = 1.2):
        super().__init__()
        self.scale = scale
        self.register_buffer("kernel", gaussian_kernel2d(ksize, sigma, sigma, 0.0))

    @classmethod
    def from_config(cls, cfg: "Config") -> "FixedDegradation":
        return cls(cfg.scale, ksize=cfg.blur_ksize, sigma=cfg.eval_sigma)

    def forward(self, x: torch.Tensor, kernel: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        k = self.kernel if kernel is None else kernel
        return blur_downsample(x, k, self.scale)
