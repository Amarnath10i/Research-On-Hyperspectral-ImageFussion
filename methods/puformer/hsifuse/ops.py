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


# Multispectral sensors for the sensor-independence tests: published nominal band ranges (nm), used as 50 %-response
# edges. Sentinel-2 MSI (ESA, centre +- half the bandwidth, the VNIR bands B1-B9), Landsat-8 OLI (B1-B5), QuickBird,
# PlanetScope (PS2.SD) and IKONOS; WorldView-2 uses its tabulated 5 % edges (WV2_BANDS).
SENSOR_BANDS = {
    "sentinel2": [("B1", 433, 453), ("B2", 457.5, 522.5), ("B3", 542.5, 577.5), ("B4", 650, 680), ("B5", 697.5, 712.5),
                  ("B6", 732.5, 747.5), ("B7", 773, 793), ("B8", 784.5, 899.5), ("B8A", 855, 875), ("B9", 935, 955)],
    "landsat8": [("B1", 435, 451), ("B2", 452, 512), ("B3", 533, 590), ("B4", 636, 673), ("B5", 851, 879)],
    "quickbird": [("blue", 450, 520), ("green", 520, 600), ("red", 630, 690), ("nir", 760, 900)],
    "planetscope": [("blue", 464, 517), ("green", 547, 585), ("red", 650, 682), ("nir", 846, 888)],
}
SENSORS = ("worldview2", "ikonos", "quickbird", "planetscope", "landsat8", "sentinel2", "rgb")


def sensor_srf(name: str, wl: np.ndarray, rolloff_nm: float = 4.0) -> np.ndarray:
    """(B, M) response of a named multispectral sensor on the wavelengths wl; bands that lie (mostly) outside wl are
    dropped, so the same sensor has fewer bands on a narrower hyperspectral range."""
    if name == "rgb":                            # Nikon D700 curves (400-700 nm), interpolated onto wl
        counts = np.stack([np.interp(wl, CAVE_WL, NIKON_D700_COUNTS[:, j], left=0.0, right=0.0) for j in range(3)], 1)
        return (counts / counts.sum(0, keepdims=True)).astype(np.float32)
    bands, level = (WV2_BANDS, 0.05) if name == "worldview2" else (IKONOS_BANDS, 0.5) if name == "ikonos" else \
        (SENSOR_BANDS[name], 0.5)
    keep = [b for b in bands if min(b[2], wl[-1]) - max(b[1], wl[0]) >= 0.5 * (b[2] - b[1])]
    return band_srf(wl, keep, rolloff_nm, level=level)


def random_srf(wl: np.ndarray, rng: np.random.Generator, m_range=(3, 10), width_range=(15.0, 120.0)) -> np.ndarray:
    """(B, M) random multispectral response for sensor-independent training: M bands (uniform in m_range) with
    centres uniform over the hyperspectral range, log-uniform widths and 2-8 nm flanks, each wider than two
    hyperspectral bands; columns sum to 1."""
    m = int(rng.integers(m_range[0], m_range[1] + 1))
    step = float(np.median(np.diff(wl)))
    lo_w, hi_w = max(width_range[0], 2.5 * step), width_range[1]
    widths = np.exp(rng.uniform(np.log(lo_w), np.log(hi_w), m))
    centres = np.sort(rng.uniform(wl[0] + widths / 2, wl[-1] - widths / 2))
    bands = [("b%d" % i, c - w / 2, c + w / 2) for i, (c, w) in enumerate(zip(centres, widths))]
    return band_srf(wl, bands, float(rng.uniform(2.0, 8.0)), level=0.5)


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


def gauss1d(size: int, sigma: float, off: float = 0.0) -> np.ndarray:
    """Normalized 1-D Gaussian of width sigma whose center is moved by off pixels (a sub-pixel misregistration of the
    LR-HSI sampling grid against the HR grid)."""
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2 - off
    g = np.exp(-ax ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_kernel(size: int = 7, sigma: float = 2.0, sigma_c: float | None = None, off_r: float = 0.0,
                    off_c: float = 0.0) -> np.ndarray:
    """Separable Gaussian: width sigma and offset off_r along rows, sigma_c (default sigma) and off_c along columns.
    The defaults give the isotropic, centered kernel of the protocol."""
    k = np.outer(gauss1d(size, sigma, off_r), gauss1d(size, sigma if sigma_c is None else sigma_c, off_c))
    return k.astype(np.float32)


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


def axis_operator(n: int, scale: int, ksize: int, sigma: float, pad: str, off: float = 0.0) -> np.ndarray:
    """(n / scale, n) matrix of D along one image axis: pad by p = ksize // 2 (reflect or zeros), correlate with the
    normalized 1-D Gaussian, keep every scale-th sample from index 0. The 2-D Gaussian of gaussian_kernel is the
    outer product of this 1-D kernel, so D = D_rows x D_cols exactly (Kronecker), border included."""
    p = ksize // 2
    g = gauss1d(ksize, sigma, off)
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
        dr = axis_operator(h, deg.scale, deg.ksize, deg.sigma, deg.pad, getattr(deg, "off_r", 0.0))
        dc = axis_operator(w, deg.scale, deg.ksize, getattr(deg, "sigma_c", deg.sigma), deg.pad,
                           getattr(deg, "off_c", 0.0))
        srf = deg.srf.detach().double().cpu().numpy()                       # (B, M)
        lr, ur = np.linalg.eigh(dr.T @ dr)
        lc, uc = np.linalg.eigh(dc.T @ dc)
        nu, v = np.linalg.eigh(srf @ srf.T)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
        self.dr, self.dc, self.ur, self.uc, self.v = t(dr), t(dc), t(ur), t(uc), t(v)
        self.lam_r, self.lam_c = t(np.clip(lr, 0, None)), t(np.clip(lc, 0, None))          # per axis
        self.lam = self.lam_r[:, None] * self.lam_c[None, :]                             # (H, W)
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
    key = (h, w, str(device), deg.sigma, getattr(deg, "sigma_c", None), getattr(deg, "off_r", 0.0),
           getattr(deg, "off_c", 0.0), deg.pad)
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


def local_misreg(ms: torch.Tensor, rng, max_px: float = 1.5):
    """HR-MSI with 1-2 locally misregistered regions (parallax, moving objects): a smooth random displacement field of
    up to max_px pixels inside random rectangles, the rest registered. Returns the warped image and the mask of the
    pixels displaced by more than 0.25 pixel."""
    n, _, h, w = ms.shape
    disp = torch.zeros(n, 2, h, w, dtype=ms.dtype, device=ms.device)
    for _ in range(int(rng.uniform(1, 3))):
        ph, pw = int(rng.uniform(h // 4, h // 2)), int(rng.uniform(w // 4, w // 2))
        r, c = int(rng.uniform(0, h - ph)), int(rng.uniform(0, w - pw))
        yy = torch.linspace(-1, 1, ph, device=ms.device, dtype=ms.dtype)[:, None]
        xx = torch.linspace(-1, 1, pw, device=ms.device, dtype=ms.dtype)[None, :]
        bump = torch.clamp(1 - yy ** 2, 0) * torch.clamp(1 - xx ** 2, 0)                    # smooth, zero at the edges
        amp = [float(rng.uniform(-max_px, max_px)) for _ in range(2)]
        disp[:, 0, r:r + ph, c:c + pw] += amp[0] * bump
        disp[:, 1, r:r + ph, c:c + pw] += amp[1] * bump
    gy, gx = torch.meshgrid(torch.arange(h, device=ms.device, dtype=ms.dtype),
                            torch.arange(w, device=ms.device, dtype=ms.dtype), indexing="ij")
    grid = torch.stack([(gx + disp[:, 1]) / (w - 1) * 2 - 1, (gy + disp[:, 0]) / (h - 1) * 2 - 1], -1)
    warped = F.grid_sample(ms, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    return warped, (disp.norm(dim=1, keepdim=True) > 0.25).to(ms.dtype)


def band_noise(lr: torch.Tensor, rng, snr_lo: float = 25.0, snr_hi: float = 45.0):
    """Band-dependent, partly signal-dependent noise on the LR-HSI: the SNR follows a random smooth profile over the
    bands between snr_lo and snr_hi dB (low at one or both ends, as at the sensor's band edges), and half of the noise
    variance scales with the signal (Poisson-like)."""
    n, b = lr.shape[:2]
    t = torch.linspace(0, 1, b, device=lr.device, dtype=lr.dtype)
    c, wdt = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.3, 1.0))
    snr = snr_lo + (snr_hi - snr_lo) * torch.exp(-((t - c) / wdt) ** 2)                      # dB per band
    p = lr.pow(2).mean(dim=(2, 3), keepdim=True)                                              # (N, B, 1, 1)
    var = p / 10 ** (snr[None, :, None, None] / 10)
    sig = torch.sqrt(0.5 * var + 0.5 * var * lr.clamp_min(0) / lr.mean(dim=(2, 3), keepdim=True).clamp_min(1e-8))
    return lr + torch.randn_like(lr) * sig


def corrupt(lr: torch.Tensor, ms: torch.Tensor, rng, changes: bool | None = None, stripes: bool | None = None,
            return_masks: bool = False, misreg: bool | None = None, bandnoise: bool | None = None):
    """Corrupted observations (the reference stays clean): local changes in the HR-MSI - 1-3 rectangles of 1/8 to
    1/3 of the side taken from another image of the batch (or a shifted copy for a single image), as land-cover change
    or clouds between two acquisition dates - 1-3 dead or miscalibrated LR-HSI columns, locally misregistered HR-MSI
    regions (local_misreg) and band-dependent LR-HSI noise (band_noise). rng: numpy random state or generator; each
    kind: force (True/False) or draw (None: 70 % / 50 % / 40 % / 30 %). Masks: corrupted LR pixels, corrupted (changed
    or unregistered) HR-MSI pixels."""
    lr, ms = lr.clone(), ms.clone()
    n, _, h, w = ms.shape
    mm, ml = torch.zeros_like(ms[:, :1]), torch.zeros_like(lr[:, :1])           # where the inputs were corrupted
    if misreg if misreg is not None else rng.uniform() < 0.4:
        ms, m2 = local_misreg(ms, rng)
        mm = torch.maximum(mm, m2)
    if bandnoise if bandnoise is not None else rng.uniform() < 0.3:
        lr = band_noise(lr, rng)
    if changes if changes is not None else rng.uniform() < 0.7:
        src = ms.roll(1, 0) if n > 1 else ms.roll((h // 2, w // 2), (2, 3))
        for _ in range(int(rng.uniform(1, 4))):
            ph, pw = int(rng.uniform(h // 8, h // 3)), int(rng.uniform(w // 8, w // 3))
            r, c = int(rng.uniform(0, h - ph)), int(rng.uniform(0, w - pw))
            ms[:, :, r:r + ph, c:c + pw] = src[:, :, r:r + ph, c:c + pw]
            mm[:, :, r:r + ph, c:c + pw] = 1.0
    if stripes if stripes is not None else rng.uniform() < 0.5:
        for _ in range(int(rng.uniform(1, 4))):
            c = int(rng.uniform(0, lr.shape[-1]))
            lr[:, :, :, c] *= 0.0 if rng.uniform() < 0.5 else float(rng.uniform(0.5, 1.5))
            ml[:, :, :, c] = 1.0
    return (lr, ms, (ml, mm)) if return_masks else (lr, ms)


def null_gate(deg, d, gamma):
    """Null-space-gated correction: the part of d in the joint null space of D and R (components that neither
    observation sees, Theorem 1c) passes unchanged, the observed part is scaled by gamma in [0, 1] (scalar or (N,1,1,1)).
    gamma = 0: the correction cannot change D x or R x, so the observed part of the estimate is left to the data step."""
    kf = kron_factors(deg, d.shape[-2], d.shape[-1], d.device)
    e = kf.to_eig(d.float())
    null = (1 - kf.obs_spectral)[None, :, None, None] * (1 - kf.obs_spatial)[None, None]
    en = e * null
    return kf.from_eig(en + gamma * (e - en))


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
                 shift_nm: float = 0.0, sigma_c: float | None = None, off: tuple = (0.0, 0.0)):
        super().__init__()
        assert pad in ("zeros", "reflect"), pad
        self.scale, self.ksize, self.pad = scale, ksize, pad
        self.sigma, self.shift_nm = float(sigma), float(shift_nm)   # operator parameters (read by --omc models)
        # blur width along columns (anisotropic, axis-aligned) and sub-pixel offsets of the LR sampling grid
        self.sigma_c = float(sigma if sigma_c is None else sigma_c)
        self.off_r, self.off_c = float(off[0]), float(off[1])
        self.gaussian = True        # False once k is replaced by another kernel (then D is not separable)
        self.register_buffer("k", torch.from_numpy(gaussian_kernel(ksize, sigma, self.sigma_c, self.off_r,
                                                                   self.off_c))[None, None])
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
