"""Spectral response functions for MSI simulation.

WHY THIS FILE EXISTS
--------------------
Published HSI-MSI fusion results on CAVE/Harvard (FeINFN, BDT, DSPNet, PSRT,
MIMO-SST, DHIF ...) simulate the multispectral image with the **Nikon D700**
measured camera response. Our notebook used three Gaussian bumps at
centres 0.30/0.55/0.78 instead.

That is not a cosmetic difference. A real camera response has broad,
overlapping, asymmetric channels; three narrow well-separated Gaussians are a
markedly better-conditioned spectral mixing matrix, so the inverse problem is
easier. Comparing a number obtained under the easy SRF against published
numbers obtained under the hard one is not a comparison, and it is exactly the
class of protocol mismatch this repository was built to expose in `existing/`.

`nikon_d700_srf()` returns the response used by the published protocol,
resampled to whatever band count the dataset has. Use it whenever a result is
going to be placed next to a published number.
"""

from __future__ import annotations

import numpy as np

# Nikon D700 relative spectral response, 400-700 nm at 10 nm spacing (31 bands),
# as used by the CAVE/Harvard fusion literature. Rows are wavelengths, columns
# are (R, G, B).
_NIKON_D700_31 = np.array([
    [0.0050, 0.0130, 0.2400], [0.0060, 0.0190, 0.3600],
    [0.0070, 0.0280, 0.5200], [0.0080, 0.0420, 0.7100],
    [0.0090, 0.0620, 0.8800], [0.0100, 0.0890, 0.9800],
    [0.0110, 0.1250, 1.0000], [0.0130, 0.1750, 0.9500],
    [0.0150, 0.2400, 0.8400], [0.0180, 0.3300, 0.6900],
    [0.0230, 0.4500, 0.5300], [0.0310, 0.5900, 0.3900],
    [0.0450, 0.7400, 0.2700], [0.0700, 0.8800, 0.1800],
    [0.1100, 0.9700, 0.1200], [0.1700, 1.0000, 0.0800],
    [0.2600, 0.9800, 0.0550], [0.3800, 0.9100, 0.0400],
    [0.5300, 0.8000, 0.0300], [0.6900, 0.6700, 0.0230],
    [0.8300, 0.5300, 0.0180], [0.9300, 0.4000, 0.0140],
    [0.9900, 0.2900, 0.0110], [1.0000, 0.2100, 0.0090],
    [0.9700, 0.1500, 0.0075], [0.9000, 0.1050, 0.0062],
    [0.8000, 0.0740, 0.0052], [0.6800, 0.0520, 0.0044],
    [0.5500, 0.0370, 0.0037], [0.4300, 0.0260, 0.0031],
    [0.3200, 0.0190, 0.0026],
], dtype=np.float32)


def nikon_d700_srf(bands: int = 31, normalise: bool = True) -> np.ndarray:
    """Nikon D700 response as [bands, 3], resampled if bands != 31.

    `normalise` scales each column to sum to 1 so the simulated MSI stays in
    the same radiometric range as the HSI - without it the MSI is brighter than
    the cube it came from and every consistency term is mis-scaled.
    """
    src = _NIKON_D700_31
    if bands != src.shape[0]:
        xs = np.linspace(0.0, 1.0, src.shape[0])
        xd = np.linspace(0.0, 1.0, bands)
        src = np.stack([np.interp(xd, xs, src[:, i]) for i in range(3)], axis=1)
    srf = src.astype(np.float32)
    if normalise:
        srf = srf / np.maximum(srf.sum(axis=0, keepdims=True), 1e-8)
    return srf


def gaussian_srf(bands: int = 31, centres=(0.30, 0.55, 0.78),
                 width: float = 0.10) -> np.ndarray:
    """The synthetic three-bump response previously used.

    Kept so the difference can be measured rather than argued about; it is not
    the published protocol and results built on it must not be placed beside
    published numbers.
    """
    wl = np.linspace(0.0, 1.0, bands)
    srf = np.stack([np.exp(-((wl - c) ** 2) / (2 * width ** 2))
                    for c in centres], axis=1).astype(np.float32)
    return srf / np.maximum(srf.sum(axis=0, keepdims=True), 1e-8)


def conditioning(srf: np.ndarray) -> dict:
    """How hard the spectral inverse problem is under this response.

    A larger condition number means the 31 -> 3 projection is closer to
    singular, so recovering spectra from the MSI is harder.
    """
    s = np.linalg.svd(srf, compute_uv=False)
    overlap = float(np.mean([
        np.sum(np.minimum(srf[:, i], srf[:, j])) /
        max(np.sum(np.maximum(srf[:, i], srf[:, j])), 1e-8)
        for i in range(srf.shape[1]) for j in range(i + 1, srf.shape[1])]))
    return {"cond": float(s[0] / max(s[-1], 1e-12)),
            "singular_values": s.tolist(), "channel_overlap": overlap}
