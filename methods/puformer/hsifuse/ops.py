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


def neon_file(name: str) -> str:
    """A file of the NEON real-data set: under $PUF_NEON, or anywhere below /kaggle/input."""
    import glob
    import os
    root = os.environ.get("PUF_NEON", "/kaggle/input")
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f"{name} under {root}")
    return hits[0]


def neon_srf(shift_nm: float = 0.0) -> np.ndarray:
    """(B, 3) response of NEON's RGB camera to the spectrometer bands. Unlike the other datasets it is not known:
    it is fitted to the measured (linearized) RGB on the training tiles (kaggle/neon/neon_fit.py)."""
    r = np.load(neon_file("neon_srf.npy")).astype(np.float64)
    if shift_nm:
        wl = np.load(neon_file("wavelengths.npy"))
        r = np.stack([np.interp(wl - shift_nm, wl, r[:, j], left=0, right=0) for j in range(r.shape[1])], 1)
    return r.astype(np.float32)


def dataset_srf(name: str, shift_nm: float = 0.0) -> np.ndarray:
    if name == "pavia":
        return ikonos_srf(shift_nm=shift_nm)
    if name in ("cave", "harvard"):
        return nikon_srf(shift_nm)
    if name == "neon":
        return neon_srf(shift_nm)
    return wv2_srf(shift_nm=shift_nm)


def dataset_wl(name: str) -> np.ndarray:
    if name == "neon":
        return np.load(neon_file("wavelengths.npy"))
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


def axis_operator(n: int, scale: int, ksize: int, sigma: float, pad: str) -> np.ndarray:
    """(n / scale, n) matrix of D along one image axis: pad by p = ksize // 2 (reflect or zeros), correlate with the
    normalized 1-D Gaussian, keep every scale-th sample from index 0. The 2-D Gaussian of gaussian_kernel is the
    outer product of this 1-D kernel, so D = D_rows x D_cols exactly (Kronecker), border included."""
    p = ksize // 2
    ax = np.arange(ksize, dtype=np.float64) - p
    g = np.exp(-ax ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    m = np.zeros((n // scale, n))
    for i in range(n // scale):
        for t in range(ksize):
            j = scale * i + t - p
            if pad == "reflect":
                j = -j if j < 0 else (2 * (n - 1) - j if j >= n else j)
            elif not 0 <= j < n:
                continue
            m[i, j] += g[t]
    return m


class KronFactors:
    """Eigendecompositions that diagonalize the HSI-MSI data subproblem (I + a D^T D + b R^T R) on an H x W image:
    D^T D = I_B (x) G_rows (x) G_cols and R^T R = G_spec (x) I (x) I commute and share the eigenbasis
    V (x) U_rows (x) U_cols, with eigenvalues lam_rows[k] * lam_cols[l] (spatial) and nu[b] (spectral)."""

    def __init__(self, deg, h: int, w: int, device):
        dr = axis_operator(h, deg.scale, deg.ksize, deg.sigma, deg.pad)
        dc = axis_operator(w, deg.scale, deg.ksize, deg.sigma, deg.pad)
        srf = deg.srf.detach().double().cpu().numpy()                       # (B, M)
        lr, ur = np.linalg.eigh(dr.T @ dr)
        lc, uc = np.linalg.eigh(dc.T @ dc)
        nu, v = np.linalg.eigh(srf @ srf.T)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
        self.dr, self.dc, self.ur, self.uc, self.v = t(dr), t(dc), t(ur), t(uc), t(v)
        self.lam = t(np.clip(lr, 0, None))[:, None] * t(np.clip(lc, 0, None))[None, :]   # (H, W)
        self.nu = t(np.clip(nu, 0, None))                                                # (B,)
        tol = 1e-8
        self.obs_spatial = (self.lam > tol * self.lam.max()).float()                     # row space of D
        self.obs_spectral = (self.nu > tol * self.nu.max()).float()                      # row space of R

    def to_eig(self, x):
        return torch.einsum("cb,nchw,hk,wl->nbkl", self.v, x, self.ur, self.uc)

    def from_eig(self, e):
        return torch.einsum("bc,nckl,hk,wl->nbhw", self.v, e, self.ur, self.uc)

    def Dt(self, y):
        return torch.einsum("ih,nbij,jw->nbhw", self.dr, y, self.dc)


def kron_factors(deg, h: int, w: int, device) -> KronFactors:
    """Cached per operator object and image size."""
    cache = deg.__dict__.setdefault("_kron", {})
    key = (h, w, str(device), deg.sigma, deg.pad)
    if key not in cache:
        cache[key] = KronFactors(deg, h, w, device)
    return cache[key]


def kron_prox(deg, x, yh, ym, a, b):
    """Closed-form exact data step: argmin_z 1/2||z - x||^2 + a/2 ||D z - Y_H||^2 + b/2 ||R z - Y_M||^2 for the exact
    operators of `deg` (any border rule along each axis), by one change of basis per tensor mode. a, b: scalars or
    (N, 1, 1, 1)."""
    kf = kron_factors(deg, x.shape[-2], x.shape[-1], x.device)
    a = a.reshape(-1, 1, 1, 1) if torch.is_tensor(a) else a
    b = b.reshape(-1, 1, 1, 1) if torch.is_tensor(b) else b
    rhs = x + a * kf.Dt(yh) + b * torch.einsum("nmhw,bm->nbhw", ym, deg.srf.to(ym.dtype))
    denom = 1 + a * kf.lam[None, None] + b * kf.nu[None, :, None, None]
    return kf.from_eig(kf.to_eig(rhs) / denom)


LN2 = 0.6931471805599453


def kron_prox_learned(deg, x, yh, ym, a, b, gain_spatial, gain_spectral, code):
    """Learned spectral-domain data step. In the eigenbasis of the exact operators the exact proximal step is
    z = x + Gamma o g with g = a D^T(Y_H - D x) + b R^T(Y_M - R x) and Gamma = 1 / (1 + a lam + b nu). Here Gamma is
    multiplied by exp(ln2 tanh(u)), u = u_spatial(log lam, code) + u_spectral(log nu, code) from two small networks
    (zero at initialization: the exact step). A zero residual still gives z = x (the fixed point of Theorem 2 is kept),
    and Gamma (a lam + b nu) < 2 keeps the step nonexpansive."""
    kf = kron_factors(deg, x.shape[-2], x.shape[-1], x.device)
    n = x.shape[0]
    a = a.reshape(-1, 1, 1, 1) if torch.is_tensor(a) else a
    b = b.reshape(-1, 1, 1, 1) if torch.is_tensor(b) else b
    srf = deg.srf.to(x.dtype)
    g = a * kf.Dt(yh - torch.einsum("ih,nbhw,jw->nbij", kf.dr, x, kf.dc)) + \
        b * torch.einsum("nmhw,bm->nbhw", ym - torch.einsum("nbhw,bm->nmhw", x, srf), srf)
    denom = 1 + a * kf.lam[None, None] + b * kf.nu[None, :, None, None]
    hh, ww = kf.lam.shape
    ls = torch.log10(kf.lam + 1e-8)[None, :, :, None].expand(n, hh, ww, 1)
    lb = torch.log10(kf.nu + 1e-8)[None, :, None].expand(n, -1, 1)
    us = gain_spatial(torch.cat([ls, code[:, None, None, :].expand(n, hh, ww, code.shape[1])], -1))[..., 0]
    ub = gain_spectral(torch.cat([lb, code[:, None, :].expand(n, lb.shape[1], code.shape[1])], -1))[..., 0]
    gamma = torch.exp(LN2 * torch.tanh(us[:, None] + ub[:, :, None, None])) / denom
    return x + kf.from_eig(kf.to_eig(g) * gamma)


@torch.no_grad()
def null_split(deg, e):
    """Energy of an error image e (N, B, H, W) in the observed subspace row(D) + row(R) and in the fusion null space
    N(D) cap N(R) = range((I - P_spec) (x) (I - P_rows (x) P_cols)); returns two (N,) tensors."""
    kf = kron_factors(deg, e.shape[-2], e.shape[-1], e.device)
    c = kf.to_eig(e.float()).pow(2)
    null = (1 - kf.obs_spectral)[None, :, None, None] * (1 - kf.obs_spatial)[None, None]
    return (c * (1 - null)).flatten(1).sum(1), (c * null).flatten(1).sum(1)


class Degradation(nn.Module):
    """Differentiable D (blur + decimate), its adjoint Dt, spectral R and Rt."""

    def __init__(self, srf: np.ndarray, scale: int = 4, ksize: int = 7, sigma: float = 2.0, pad: str = "zeros",
                 shift_nm: float = 0.0):
        super().__init__()
        assert pad in ("zeros", "reflect"), pad
        self.scale, self.ksize, self.pad = scale, ksize, pad
        self.sigma, self.shift_nm = float(sigma), float(shift_nm)   # operator parameters (read by --omc models)
        self.gaussian = True        # False once k is replaced by another kernel (then D is not separable)
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
        """Whole-image simulation (reflect padding at the image border). Real data (NEON) carry the measured
        HR-MSI as extra channels after the B hyperspectral bands; then only the LR-HSI is simulated."""
        b = self.srf.shape[0]
        if x.shape[1] > b:
            return self.D(x[:, :b], pad_mode="reflect"), x[:, b:]
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
