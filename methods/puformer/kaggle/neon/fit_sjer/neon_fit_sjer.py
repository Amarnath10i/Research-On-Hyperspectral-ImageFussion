"""Camera response and test split of the second NEON site (SJER), for cross-scene tests of the networks trained on
SERC (Kaggle CPU).

Reads the SJER tiles (puf-neon-sjer) and the SERC fit (puf-neon-fit). As for SERC (neon_fit.py), the camera RGB is
linearized with the sRGB decoding curve and the response y_c = r_c . x / norm + b_c is fitted by non-negative least
squares, here on the first three SJER tiles (calibration of the operator, no network involved); the other tiles are
the test set. The reflectance is divided by SERC's normalization, so the SERC-trained networks see inputs in the
units they were trained in, and the spectra are interpolated onto SERC's wavelengths if the two flights differ.
Writes, for the loader in hsifuse/data.py (load_neon), with train = [] and the SJER test tiles:
    neon_srf.npy, <tile>_hsi.npy (on SERC's wavelengths), <tile>_msi.npy, <tile>_msi025.npy (test tiles),
    wavelengths.npy (SERC's), neon_fit.json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "/kaggle/working"


def srgb_decode(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def fit_response(pairs, rng):
    """NNLS fit of y_c = r_c . x + b_c on 40,000 random pixels of every (x, y) pair (as neon_fit.py)."""
    from scipy.optimize import nnls
    A, T = [], []
    for x, y in pairs:
        b = x.shape[0]
        idx = rng.choice(x.shape[1] * x.shape[2], 40000, replace=False)
        A.append(x.reshape(b, -1)[:, idx].T), T.append(y.reshape(3, -1)[:, idx].T)
    A, T = np.concatenate(A).astype(np.float64), np.concatenate(T).astype(np.float64)
    aug = np.hstack([A, np.ones((len(A), 1)), -np.ones((len(A), 1))])
    G = aug.T @ aug
    L = np.linalg.cholesky(G + 1e-9 * np.trace(G) / len(G) * np.eye(len(G)))
    R, off = np.zeros((A.shape[1], 3)), np.zeros(3)
    for c in range(3):
        coef, _ = nnls(L.T, np.linalg.solve(L, aug.T @ T[:, c]), maxiter=50 * aug.shape[1])
        R[:, c], off[c] = coef[:-2], coef[-2] - coef[-1]
    return R, off


def r2(p, m):
    return [float(1 - ((p[c] - m[c]) ** 2).mean() / m[c].var()) for c in range(3)]


def main():
    sjer = os.path.dirname(sorted(glob.glob("/kaggle/input/**/neon_meta.json", recursive=True), key=len)[0])
    serc_fit = sorted(glob.glob("/kaggle/input/**/neon_fit.json", recursive=True), key=len)[0]
    serc = os.path.dirname(serc_fit)
    fit0 = json.load(open(serc_fit))
    norm = float(fit0["norm"])
    wl_serc = np.load(f"{serc}/wavelengths.npy").astype(np.float64)
    wl_sjer = np.load(f"{sjer}/wavelengths.npy").astype(np.float64)
    same = len(wl_serc) == len(wl_sjer) and np.allclose(wl_serc, wl_sjer, atol=0.5)
    meta = json.load(open(f"{sjer}/neon_meta.json"))
    names = [t["name"] for t in meta["tiles"]]
    assert len(names) >= 4, names
    calib, test = names[:3], names[3:]

    def hsi(n):
        x = np.load(f"{sjer}/{n}_hsi.npy").astype(np.float32) / norm
        if not same:                                           # onto SERC's wavelengths, band by band
            flat = x.reshape(len(wl_sjer), -1)
            out = np.empty((len(wl_serc), flat.shape[1]), np.float32)
            for j in range(0, flat.shape[1], 200000):
                blk = flat[:, j:j + 200000]
                out[:, j:j + 200000] = np.stack([np.interp(wl_serc, wl_sjer, blk[:, k]) for k in range(blk.shape[1])], 1)
            x = out.reshape(len(wl_serc), *x.shape[1:])
        return x

    rgb = lambda n: srgb_decode(np.load(f"{sjer}/{n}_rgb1m.npy").astype(np.float32))
    rng = np.random.default_rng(0)
    R, off = fit_response([(hsi(n), rgb(n)) for n in calib], rng)
    kappa = R.sum(0)
    R = R / kappa
    np.save(f"{OUT}/neon_srf.npy", R.astype(np.float32))
    quality = {}
    for n in test:
        x = hsi(n)
        np.save(f"{OUT}/{n}_hsi.npy", (x * norm).astype(np.float16))   # the loader divides by norm again
        y = rgb(n)
        msi = np.clip(y - off[:, None, None], 0, None) / kappa[:, None, None]
        np.save(f"{OUT}/{n}_msi.npy", msi.astype(np.float16))
        pred = np.einsum("bc,bhw->chw", R, x)
        quality[n] = dict(psnr=float(10 * np.log10(1.0 / float(((pred - msi) ** 2).mean()))), r2=r2(pred, msi))
        hi = srgb_decode(np.load(f"{sjer}/{n}_rgb025.npy").astype(np.float32) / 255.0)
        np.save(f"{OUT}/{n}_msi025.npy", (np.clip(hi - off[:, None, None], 0, None) / kappa[:, None, None]).astype(np.float16))
        del hi
        print(n, quality[n], flush=True)
    np.save(f"{OUT}/wavelengths.npy", wl_serc.astype(np.float32))
    fit = dict(train=[], test=test, calibration=calib, norm=norm, offset=off.tolist(), msi_scale=kappa.tolist(),
               srgb_decoded=True, quality=quality, wavelengths_resampled=not same, site="SJER", source=meta,
               serc_norm_from=serc_fit,
               response_peak_nm=[float(wl_serc[int(np.argmax(R[:, c]))]) for c in range(3)])
    json.dump(fit, open(f"{OUT}/neon_fit.json", "w"), indent=1)
    print(json.dumps({k: fit[k] for k in ("calibration", "test", "norm", "quality", "wavelengths_resampled",
                                          "response_peak_nm")}, indent=1))


if __name__ == "__main__":
    main()
