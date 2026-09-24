"""Degradation model for the Chikusei x4 protocol (TIP'26 Two-Stage Diffusion setting).

    LR-HSI  Y_H = S(K * X)      K: Gaussian 7x7, sigma=2;  S: x4 decimation
    HR-MSI  Y_M = R X           R: WorldView-2 8-band spectral response

The same operators are used for data simulation and inside the network's
data-consistency steps, so D / Dt and R / Rt are exact adjoint pairs
(zero padding inside the network; reflect padding when simulating whole images).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHIKUSEI_WL = np.linspace(363.0, 1018.0, 128)

# WorldView-2 MS bands: 5 %-response lower/upper edges (nm), DigitalGlobe
# "Spectral Response for DigitalGlobe Earth Imaging Instruments", Table 5.
WV2_BANDS = [
    ("coastal", 396, 458), ("blue", 442, 515), ("green", 506, 586), ("yellow", 584, 632),
    ("red", 624, 694), ("red_edge", 699, 749), ("nir1", 765, 901), ("nir2", 856, 1043),
]


def wv2_srf(wl: np.ndarray = CHIKUSEI_WL, rolloff_nm: float = 4.0, shift_nm: float = 0.0) -> np.ndarray:
    """(B, 8) spectral response, columns sum to 1.

    DigitalGlobe publishes the curves only as plots, so each band is a flat top
    with logistic flanks that reach 5 % response exactly at the tabulated edges.
    `shift_nm` moves every band (used to test SRF mismatch).
    """
    a = np.log(0.05 / 0.95)  # logistic offset -> value 0.05 at the edge
    cols = []
    for _, lo, hi in WV2_BANDS:
        lo, hi = lo + shift_nm, hi + shift_nm
        r = 1 / (1 + np.exp(-((wl - lo) / rolloff_nm + a))) / (1 + np.exp(-((hi - wl) / rolloff_nm + a)))
        cols.append(r)
    srf = np.stack(cols, 1).astype(np.float64)
    return (srf / srf.sum(0, keepdims=True)).astype(np.float32)


def gaussian_kernel(size: int = 7, sigma: float = 2.0) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


class Degradation(nn.Module):
    """Differentiable D (blur + decimate), its adjoint Dt, spectral R and Rt."""

    def __init__(self, srf: np.ndarray, scale: int = 4, ksize: int = 7, sigma: float = 2.0):
        super().__init__()
        self.scale, self.ksize = scale, ksize
        self.register_buffer("k", torch.from_numpy(gaussian_kernel(ksize, sigma))[None, None])
        self.register_buffer("srf", torch.from_numpy(srf))  # (B, M)

    def blur(self, x: torch.Tensor, pad_mode: str = "zeros") -> torch.Tensor:
        c, p = x.shape[1], self.ksize // 2
        if pad_mode != "zeros":
            x = F.pad(x, (p, p, p, p), mode=pad_mode)
            p = 0
        return F.conv2d(x, self.k.expand(c, 1, -1, -1).to(x.dtype), padding=p, groups=c)

    def D(self, x: torch.Tensor, pad_mode: str = "zeros") -> torch.Tensor:
        return self.blur(x, pad_mode)[..., :: self.scale, :: self.scale]

    def Dt(self, y: torch.Tensor) -> torch.Tensor:
        n, c, h, w = y.shape
        up = y.new_zeros(n, c, h * self.scale, w * self.scale)
        up[..., :: self.scale, :: self.scale] = y
        return self.blur(up)  # kernel is symmetric, so blur is self-adjoint

    def R(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nbhw,bm->nmhw", x, self.srf.to(x.dtype))

    def Rt(self, m: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nmhw,bm->nbhw", m, self.srf.to(m.dtype))

    @torch.no_grad()
    def simulate(self, x: torch.Tensor):
        """Whole-image simulation (reflect padding at the image border)."""
        return self.D(x, pad_mode="reflect"), self.R(x)
