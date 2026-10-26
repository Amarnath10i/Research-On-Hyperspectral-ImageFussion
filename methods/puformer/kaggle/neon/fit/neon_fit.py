"""Spectral response of NEON's RGB camera to the spectrometer bands, fitted on the training tiles (Kaggle CPU).

Reads the output of puf-neon-data (attached as a kernel source). The camera RGB is gamma-encoded, so it is
first linearized with the sRGB decoding curve; then, per channel, y_c = r_c . x / norm + b_c is fitted by
non-negative least squares (r_c >= 0, free offset b_c) on 40,000 random pixels of every training tile. The
measured RGB minus the offset is the HR-MSI; it is not registered or otherwise adjusted. As for the simulated
datasets, every response is scaled to unit sum, and its RGB channel by the same factor (a change of units: the
data step uses R^T R, whose scale would otherwise depend on the fit).
The camera is an uncalibrated frame camera (its own tone curve and white balance, at-sensor radiance), so a linear
response explains only part of the RGB: the fit quality on the test tiles is reported at 1 m, at 4 m (block means)
and with both images blurred, to show that the mismatch is spectral / radiometric rather than a registration or
PSF difference. (A dense optical-flow registration and a 16 m relative radiometric normalization were tried and
did not raise R^2 at 4 m.)
Writes, for the loader in hsifuse/data.py (load_neon):
    neon_srf.npy        (B, 3) fitted response (the network's R for this dataset)
    <tile>_msi.npy      (3, 1000, 1000) float16 measured HR-MSI at 1 m: linearized RGB minus the offset
    <tile>_msi025.npy   (3, 4000, 4000) float16 the same at 0.25 m, test tiles only (full-resolution fusion)
    neon_fit.json       split (training / test tiles), normalization, offsets and the fit quality on the test tiles
The spectrometer tiles themselves stay in puf-neon-data; training sessions attach both kernel outputs.
"""
import glob
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls

OUT = "/kaggle/working"


def srgb_decode(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def fit_response(pairs, rng):
    """NNLS fit of y_c = r_c . x + b_c on 40,000 random pixels of every (x, y) pair."""
    A, T = [], []
    for x, y in pairs:
        b = x.shape[0]
        idx = rng.choice(x.shape[1] * x.shape[2], 40000, replace=False)
        A.append(x.reshape(b, -1)[:, idx].T), T.append(y.reshape(3, -1)[:, idx].T)
    A, T = np.concatenate(A).astype(np.float64), np.concatenate(T).astype(np.float64)
    aug = np.hstack([A, np.ones((len(A), 1)), -np.ones((len(A), 1))])     # r >= 0, offset of either sign
    # the same NNLS problem in the n x n form: with G = aug^T aug = L L^T, ||aug c - t||^2 = ||L^T c - L^-1 aug^T t||^2
    # + const (a tiny ridge keeps G positive definite: the two offset columns are collinear)
    G = aug.T @ aug
    L = np.linalg.cholesky(G + 1e-9 * np.trace(G) / len(G) * np.eye(len(G)))
    R, off = np.zeros((A.shape[1], 3)), np.zeros(3)
    for c in range(3):
        coef, _ = nnls(L.T, np.linalg.solve(L, aug.T @ T[:, c]), maxiter=50 * aug.shape[1])
        R[:, c], off[c] = coef[:-2], coef[-2] - coef[-1]
    return R, off


def block4(a):
    c, h, w = a.shape
    return a.reshape(c, h // 4, 4, w // 4, 4).mean((2, 4))


def r2(p, m):
    return [float(1 - ((p[c] - m[c]) ** 2).mean() / m[c].var()) for c in range(3)]


def main():
    root = os.path.dirname(sorted(glob.glob("/kaggle/input/**/neon_meta.json", recursive=True), key=len)[0])
    meta = json.load(open(f"{root}/neon_meta.json"))
    names = [t["name"] for t in meta["tiles"]]
    assert len(names) >= 3, names
    test = names[-2:] if len(names) >= 5 else names[-1:]
    train = [n for n in names if n not in test]
    norm = 0.0
    for n in names:
        norm = max(norm, float(np.load(f"{root}/{n}_hsi.npy", mmap_mode="r").max()))
    rng = np.random.default_rng(0)
    hsi = lambda n: np.load(f"{root}/{n}_hsi.npy").astype(np.float32) / norm
    rgb = lambda n: srgb_decode(np.load(f"{root}/{n}_rgb1m.npy").astype(np.float32))
    R, off = fit_response([(hsi(n), rgb(n)) for n in train], rng)
    kappa = R.sum(0)                                         # unit-sum responses; the MSI in the same units
    R = R / kappa
    np.save(f"{OUT}/neon_srf.npy", R.astype(np.float32))
    quality = {}
    for n in names:
        y = rgb(n)
        msi = np.clip(y - off[:, None, None], 0, None) / kappa[:, None, None]
        np.save(f"{OUT}/{n}_msi.npy", msi.astype(np.float16))
        if n in test:
            x = hsi(n)
            pred = np.einsum("bc,bhw->chw", R, x)
            q = dict(psnr=float(10 * np.log10(1.0 / float(((pred - msi) ** 2).mean()))), r2=r2(pred, msi),
                     r2_4m=r2(block4(pred), block4(msi)))
            for s in (1.0, 1.5):
                blur = lambda a: np.stack([gaussian_filter(a[c], s) for c in range(3)])
                q[f"r2_blur{s}"] = r2(blur(pred), blur(msi))
            quality[n] = q
            hi = srgb_decode(np.load(f"{root}/{n}_rgb025.npy").astype(np.float32) / 255.0)
            np.save(f"{OUT}/{n}_msi025.npy",
                    (np.clip(hi - off[:, None, None], 0, None) / kappa[:, None, None]).astype(np.float16))
            del hi
            print(n, q, flush=True)
    import shutil
    shutil.copyfile(f"{root}/wavelengths.npy", f"{OUT}/wavelengths.npy")
    wl = np.load(f"{root}/wavelengths.npy")
    fit = dict(train=train, test=test, norm=norm, offset=off.tolist(), msi_scale=kappa.tolist(), srgb_decoded=True,
               quality=quality,
               response_peak_nm=[float(wl[int(np.argmax(R[:, c]))]) for c in range(3)], source=meta)
    json.dump(fit, open(f"{OUT}/neon_fit.json", "w"), indent=1)
    print(json.dumps({k: fit[k] for k in ("train", "test", "norm", "offset", "quality", "response_peak_nm")}, indent=1))


if __name__ == "__main__":
    main()
