"""Degradation model (TIP'26 Two-Stage Diffusion setting, extended to x8 and to CAVE / Harvard).

    LR-HSI  Y_H = S(K * X)      K: Gaussian 7x7, sigma=2;  S: x4 or x8 decimation
    HR-MSI  Y_M = R X           R: WorldView-2 8 bands (Chikusei), IKONOS 4 bands (Pavia Centre),
                                   Nikon D700 RGB (CAVE, Harvard)

Data are simulated per image with reflect padding at the image border. The network's
data-consistency steps use the same operators, and D / Dt and R / Rt are exact adjoint
pairs for either border mode:
    pad="reflect"  D is exactly the simulation operator (D X = Y_H for the true X), and
                   Dt is its exact adjoint (reflect padding folded back onto the border)
    pad="zeros"    legacy: zero padding inside the network (used by the first 57.99 dB run)
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


# Pavia Centre (ROSIS): 102 bands over 430-860 nm; bands 11-102 are used (see data.py)
PAVIA_WL = np.linspace(430.0, 860.0, 102)[10:]

# IKONOS MS bands (nm): published spectral ranges, used as the 50 %-response edges
IKONOS_BANDS = [("blue", 445, 516), ("green", 506, 595), ("red", 632, 698), ("nir", 757, 853)]


def band_srf(wl: np.ndarray, bands, rolloff_nm: float = 4.0, shift_nm: float = 0.0, level: float = 0.05) -> np.ndarray:
    """(B, M) spectral response from band edges, columns sum to 1.

    Each band is a flat top with logistic flanks that reach `level` response exactly at the
    given edges. `shift_nm` moves every band (used to test SRF mismatch).
    """
    a = np.log(level / (1 - level))  # logistic offset -> value `level` at the edge
    cols = []
    for _, lo, hi in bands:
        lo, hi = lo + shift_nm, hi + shift_nm
        r = 1 / (1 + np.exp(-((wl - lo) / rolloff_nm + a))) / (1 + np.exp(-((hi - wl) / rolloff_nm + a)))
        cols.append(r)
    srf = np.stack(cols, 1).astype(np.float64)
    return (srf / srf.sum(0, keepdims=True)).astype(np.float32)


def wv2_srf(wl: np.ndarray = CHIKUSEI_WL, rolloff_nm: float = 4.0, shift_nm: float = 0.0) -> np.ndarray:
    """(B, 8) WorldView-2 response. DigitalGlobe publishes the curves only as plots, so the
    flanks reach 5 % exactly at the tabulated 5 %-response edges."""
    return band_srf(wl, WV2_BANDS, rolloff_nm, shift_nm, level=0.05)


def ikonos_srf(wl: np.ndarray = PAVIA_WL, rolloff_nm: float = 4.0, shift_nm: float = 0.0) -> np.ndarray:
    """(B, 4) IKONOS-type response (blue, green, red, NIR), 50 % response at the published range edges."""
    return band_srf(wl, IKONOS_BANDS, rolloff_nm, shift_nm, level=0.5)


# CAVE (400-700 nm) and Harvard (420-720 nm): 31 bands at 10 nm, HR-MSI = RGB from the Nikon D700 response.
CAVE_WL = np.linspace(400.0, 700.0, 31)
HARVARD_WL = np.linspace(420.0, 720.0, 31)
# Nikon D700 (R, G, B) response of the CAVE / Harvard fusion literature: matrix A of "response
# coefficient.mat" distributed with the DCTransformer code (Ma et al., Information Fusion 2024; the same
# matrix as create_F() in MHF-Net and its successors). Stored there as integer counts / channel sum
# (255, 203, 251); these are the counts, one per band (400-700 nm for CAVE, applied band-wise to Harvard).
NIKON_D700_COUNTS = np.array([
    [2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
    [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
    [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], np.float64).T                                           # (31, 3)
NIKON_D700_SOURCE = "https://github.com/qingma2016/DCTransformer/blob/main/response%20coefficient.mat"


def nikon_srf(shift_nm: float = 0.0) -> np.ndarray:
    """(31, 3) Nikon D700 response, columns sum to 1; `shift_nm` resamples it shifted along wavelength."""
    r = NIKON_D700_COUNTS                                   # (31, 3) on the 10 nm band grid
    if shift_nm:
        r = np.stack([np.interp(CAVE_WL - shift_nm, CAVE_WL, r[:, j], left=0, right=0) for j in range(3)], 1)
    return (r / r.sum(0, keepdims=True)).astype(np.float32)


def dataset_srf(name: str, shift_nm: float = 0.0) -> np.ndarray:
    if name == "pavia":
        return ikonos_srf(shift_nm=shift_nm)
    if name in ("cave", "harvard"):
        return nikon_srf(shift_nm)
    return wv2_srf(shift_nm=shift_nm)


def dataset_wl(name: str) -> np.ndarray:
    return dict(pavia=PAVIA_WL, cave=CAVE_WL, harvard=HARVARD_WL).get(name, CHIKUSEI_WL)


def gaussian_kernel(size: int = 7, sigma: float = 2.0) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def reflect_pad_adjoint(g: torch.Tensor, p: int) -> torch.Tensor:
    """Adjoint of F.pad(x, (p, p, p, p), mode="reflect"): (.., H+2p, W+2p) -> (.., H, W).

    Padded row j < p is a copy of row p - j, padded row H+p+m is a copy of row H-2-m,
    so their values are added back onto those rows (then the same for columns).
    """
    h = g.shape[-2] - 2 * p
    out = g[..., p:p + h, :].clone()
    out[..., 1:p + 1, :] += g[..., :p, :].flip(-2)
    out[..., h - 1 - p:h - 1, :] += g[..., h + p:, :].flip(-2)
    w = out.shape[-1] - 2 * p
    res = out[..., p:p + w].clone()
    res[..., 1:p + 1] += out[..., :p].flip(-1)
    res[..., w - 1 - p:w - 1] += out[..., w + p:].flip(-1)
    return res


class Degradation(nn.Module):
    """Differentiable D (blur + decimate), its adjoint Dt, spectral R and Rt."""

    def __init__(self, srf: np.ndarray, scale: int = 4, ksize: int = 7, sigma: float = 2.0, pad: str = "zeros"):
        super().__init__()
        assert pad in ("zeros", "reflect"), pad
        self.scale, self.ksize, self.pad = scale, ksize, pad
        self.register_buffer("k", torch.from_numpy(gaussian_kernel(ksize, sigma))[None, None])
        self.register_buffer("srf", torch.from_numpy(srf))  # (B, M)

    def _k(self, x: torch.Tensor) -> torch.Tensor:
        return self.k.expand(x.shape[1], 1, -1, -1).to(x.dtype)

    def blur(self, x: torch.Tensor, pad_mode: str = "zeros") -> torch.Tensor:
        """Full-resolution blur (used by the robustness study and GSA)."""
        c, p = x.shape[1], self.ksize // 2
        if pad_mode != "zeros":
            x = F.pad(x, (p, p, p, p), mode=pad_mode)
            p = 0
        return F.conv2d(x, self._k(x), padding=p, groups=c)

    def D(self, x: torch.Tensor, pad_mode: str | None = None) -> torch.Tensor:
        """Blur, then keep every scale-th pixel starting at 0 (a strided conv, same result)."""
        mode, p = pad_mode or self.pad, self.ksize // 2
        if mode != "zeros":
            x = F.pad(x, (p, p, p, p), mode=mode)
            p = 0
        return F.conv2d(x, self._k(x), stride=self.scale, padding=p, groups=x.shape[1])

    def Dt(self, y: torch.Tensor) -> torch.Tensor:
        """Exact adjoint of D (with this operator's border mode)."""
        s, p = self.scale, self.ksize // 2
        op = s - 1  # output_padding: gives exactly scale * h rows (zeros) / scale * h + 2p (reflect)
        if self.pad == "zeros":
            return F.conv_transpose2d(y, self._k(y), stride=s, padding=p, output_padding=op, groups=y.shape[1])
        full = F.conv_transpose2d(y, self._k(y), stride=s, output_padding=op, groups=y.shape[1])
        return reflect_pad_adjoint(full, p)

    def R(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nbhw,bm->nmhw", x, self.srf.to(x.dtype))

    def Rt(self, m: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nmhw,bm->nbhw", m, self.srf.to(m.dtype))

    @torch.no_grad()
    def simulate(self, x: torch.Tensor):
        """Whole-image simulation (reflect padding at the image border)."""
        return self.D(x, pad_mode="reflect"), self.R(x)


class LearnedDegradation(nn.Module):
    """Ablation: the operators are learned with the network, as in unfolding networks that model D and R
    with trainable layers instead of the known simulation.

    D = stride-s correlation with one learned ksize x ksize kernel shared by all bands (zero border),
    R = a learned non-negative (B, M) response with unit-sum columns, and Dt / Rt their exact adjoints, so
    only the operators themselves differ from the exact model. The kernel starts flat and R uniform, i.e.
    no knowledge of the true blur or SRF.
    """

    def __init__(self, bands: int, msi: int, scale: int = 4, ksize: int = 7):
        super().__init__()
        self.scale, self.ksize, self.pad = scale, ksize, "zeros"
        self.k_logit = nn.Parameter(torch.zeros(1, 1, ksize, ksize))
        self.r_logit = nn.Parameter(torch.zeros(bands, msi))
        self.register_buffer("srf", torch.full((bands, msi), 1.0 / bands))  # shape only (band counts)

    def _k(self, x):
        k = torch.softmax(self.k_logit.flatten(), 0).view_as(self.k_logit)
        return k.expand(x.shape[1], 1, -1, -1).to(x.dtype)

    def _r(self, dtype):
        return torch.softmax(self.r_logit, 0).to(dtype)

    def D(self, x, pad_mode=None):
        return F.conv2d(x, self._k(x), stride=self.scale, padding=self.ksize // 2, groups=x.shape[1])

    def Dt(self, y):
        return F.conv_transpose2d(y, self._k(y), stride=self.scale, padding=self.ksize // 2,
                                  output_padding=self.scale - 1, groups=y.shape[1])

    def R(self, x):
        return torch.einsum("nbhw,bm->nmhw", x, self._r(x.dtype))

    def Rt(self, m):
        return torch.einsum("nmhw,bm->nbhw", m, self._r(m.dtype))
