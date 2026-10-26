"""Spectral response of NEON's RGB camera to the spectrometer bands, fitted on the training tiles (Kaggle CPU).

Reads the output of puf-neon-data (attached as a kernel source). The camera RGB is gamma-encoded, so it is
first linearized with the sRGB decoding curve; then, per channel, y_c = r_c . x / norm + b_c is fitted by
non-negative least squares (r_c >= 0, free offset b_c) on 40,000 random pixels of every training tile.
Writes, for the loader in hsifuse/data.py (load_neon):
    neon_srf.npy        (B, 3) fitted response (the network's R for this dataset)
    <tile>_msi.npy      (3, 1000, 1000) float16 measured HR-MSI at 1 m: linearized RGB minus the offset
    <tile>_msi025.npy   (3, 4000, 4000) float16 the same at 0.25 m, test tiles only (full-resolution fusion)
    neon_fit.json       split (training / test tiles), normalization, offsets and the fit quality on the test
                        tiles (PSNR and R^2 of R x against the measured MSI: the cross-sensor mismatch)
The spectrometer tiles themselves stay in puf-neon-data; training sessions attach both kernel outputs.
"""
import glob
import json
import os

import numpy as np
from scipy.optimize import nnls

OUT = "/kaggle/working"


def srgb_decode(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


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
    A, T = [], []
    for n in train:
        x = np.load(f"{root}/{n}_hsi.npy").astype(np.float32) / norm
        y = srgb_decode(np.load(f"{root}/{n}_rgb1m.npy").astype(np.float32))
        b = x.shape[0]
        idx = rng.choice(x.shape[1] * x.shape[2], 40000, replace=False)
        A.append(x.reshape(b, -1)[:, idx].T), T.append(y.reshape(3, -1)[:, idx].T)
    A, T = np.concatenate(A).astype(np.float64), np.concatenate(T).astype(np.float64)
    aug = np.hstack([A, np.ones((len(A), 1)), -np.ones((len(A), 1))])     # r >= 0, offset of either sign
    R, off = np.zeros((A.shape[1], 3)), np.zeros(3)
    for c in range(3):
        coef, _ = nnls(aug, T[:, c], maxiter=50 * aug.shape[1])
        R[:, c], off[c] = coef[:-2], coef[-2] - coef[-1]
    np.save(f"{OUT}/neon_srf.npy", R.astype(np.float32))
    quality = {}
    for n in names:
        y = srgb_decode(np.load(f"{root}/{n}_rgb1m.npy").astype(np.float32))
        msi = np.clip(y - off[:, None, None], 0, None)
        np.save(f"{OUT}/{n}_msi.npy", msi.astype(np.float16))
        if n in test:
            x = np.load(f"{root}/{n}_hsi.npy").astype(np.float32) / norm
            pred = np.einsum("bc,bhw->chw", R, x)
            mse = float(((pred - msi) ** 2).mean())
            r2 = [float(1 - ((pred[c] - msi[c]) ** 2).mean() / msi[c].var()) for c in range(3)]
            quality[n] = dict(psnr=float(10 * np.log10(1.0 / mse)), r2=r2)
            hi = srgb_decode(np.load(f"{root}/{n}_rgb025.npy").astype(np.float32) / 255.0)
            np.save(f"{OUT}/{n}_msi025.npy", np.clip(hi - off[:, None, None], 0, None).astype(np.float16))
    import shutil
    shutil.copyfile(f"{root}/wavelengths.npy", f"{OUT}/wavelengths.npy")
    fit = dict(train=train, test=test, norm=norm, offset=off.tolist(), srgb_decoded=True, quality=quality,
               response_peak_nm=[float(np.load(f"{root}/wavelengths.npy")[int(np.argmax(R[:, c]))]) for c in range(3)],
               source=meta)
    json.dump(fit, open(f"{OUT}/neon_fit.json", "w"), indent=1)
    print(json.dumps({k: fit[k] for k in ("train", "test", "norm", "offset", "quality", "response_peak_nm")},
                     indent=1))


if __name__ == "__main__":
    main()
