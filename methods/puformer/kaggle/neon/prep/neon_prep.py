"""Real hyperspectral + RGB data from NEON's Airborne Observation Platform (Kaggle CPU kernel, internet on).

NEON flies an imaging spectrometer (1 m surface reflectance, 426 bands, 380-2510 nm) and an RGB camera (10 cm)
together; both are delivered as 1 km x 1 km tiles on the same UTM grid (products DP3.30006.001 and DP3.30010.001;
the files are read directly from NEON's public storage). For the flight 2019_SERC_4 this script finds the tiles
that have both products and no missing pixels, and writes for up to N_TILES of them:
    <tile>_hsi.npy    (B, 1000, 1000) float16 reflectance, 400-1000 nm (the VNIR range of the other datasets)
    <tile>_rgb1m.npy  (3, 1000, 1000) float16 RGB averaged to 1 m (reduced-resolution protocol, in [0, 1])
    <tile>_rgb025.npy (3, 4000, 4000) uint8 RGB averaged to 0.25 m (full-resolution x4 fusion, no reference)
    wavelengths.npy, neon_meta.json (tile list, valid fractions, HSI-RGB correlation and phase-correlation shift)
"""
import json
import os
import re

import numpy as np
import requests

BASE = "https://storage.googleapis.com/neon-aop-products"
YEAR, SITE, DOMAIN, VISIT = 2019, "SERC", "D02", 4
CENTER = (368000, 4306000)                 # the tile of NEON's tutorials
N_TILES = 8
WL_RANGE = (400.0, 1000.0)
OUT, TMP = "/kaggle/working", "/tmp/neon"
os.makedirs(TMP, exist_ok=True)


def urls(e, n):
    root = f"{BASE}/{YEAR}/FullSite/{DOMAIN}/{YEAR}_{SITE}_{VISIT}/L3"
    return (f"{root}/Spectrometer/Reflectance/NEON_{DOMAIN}_{SITE}_DP3_{e}_{n}_reflectance.h5",
            f"{root}/Camera/Mosaic/{YEAR}_{SITE}_{VISIT}_{e}_{n}_image.tif")


def exists(url):
    try:
        return requests.get(url, headers={"Range": "bytes=0-15"}, timeout=30).status_code in (200, 206)
    except requests.RequestException:
        return False


def download(url, path):
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)


def phase_shift(a, b):
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    p = np.fft.ifft2(A * np.conj(B) / (np.abs(A * np.conj(B)) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(p), p.shape)
    h, w = p.shape
    return [int(dy if dy < h // 2 else dy - h), int(dx if dx < w // 2 else dx - w)]


def main():
    import subprocess
    subprocess.run("pip install -q tifffile imagecodecs", shell=True)
    import cv2
    import h5py
    import tifffile

    # candidate tiles: a 13 x 13 km window, nearest to the tutorial tile first
    cand = [(CENTER[0] + 1000 * i, CENTER[1] + 1000 * j) for i in range(-6, 7) for j in range(-6, 7)]
    cand.sort(key=lambda t: (t[0] - CENTER[0]) ** 2 + (t[1] - CENTER[1]) ** 2)
    both = [t for t in cand if exists(urls(*t)[0]) and exists(urls(*t)[1])]
    print(f"{len(both)} tiles with both products:", both, flush=True)

    meta, wl_keep = dict(year=YEAR, site=SITE, visit=VISIT, tiles=[]), None
    for e, n in both:
        if len(meta["tiles"]) >= N_TILES:
            break
        uh, ur = urls(e, n)
        ph, pr = f"{TMP}/h.h5", f"{TMP}/r.tif"
        download(uh, ph)
        with h5py.File(ph, "r") as f:
            g = f[SITE]["Reflectance"]
            refl = g["Reflectance_Data"]
            wl = g["Metadata"]["Spectral_Data"]["Wavelength"][()]
            nodata = float(refl.attrs.get("Data_Ignore_Value", -9999))
            scale = float(refl.attrs.get("Scale_Factor", 10000))
            sel = np.flatnonzero((wl >= WL_RANGE[0]) & (wl <= WL_RANGE[1]))
            cube = refl[:, :, sel[0]:sel[-1] + 1]                           # (1000, 1000, B) int16
            mapinfo = str(g["Metadata"]["Coordinate_System"]["Map_Info"][()])
        os.remove(ph)
        valid = (cube != nodata).all(-1)
        frac = float(valid.mean())
        print(f"tile {e}_{n}: HSI {cube.shape}, valid {frac:.3f}, map {mapinfo[:80]}", flush=True)
        if frac < 0.999:                                                     # skip tiles at the flight-box edge
            continue
        x = np.clip(np.moveaxis(cube, -1, 0).astype(np.float32) / scale, 0, None)
        wl_keep = wl[sel[0]:sel[-1] + 1]
        download(ur, pr)
        rgb = tifffile.imread(pr)
        os.remove(pr)
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4):
            rgb = np.moveaxis(rgb, 0, -1)
        rgb = rgb[..., :3]
        h, w = x.shape[1:]
        fy, fx = rgb.shape[0] // h, rgb.shape[1] // w
        rgb1 = rgb[:h * fy, :w * fx].reshape(h, fy, w, fx, 3).astype(np.float32).mean((1, 3)) / 255.0
        rgb025 = cv2.resize(rgb, (4 * w, 4 * h), interpolation=cv2.INTER_AREA)
        near = [int(np.argmin(abs(wl_keep - t))) for t in (650.0, 550.0, 470.0)]       # R, G, B
        corr = [float(np.corrcoef(rgb1[..., c].ravel(), x[near[c]].ravel())[0, 1]) for c in range(3)]
        shift = phase_shift(rgb1[..., 1], x[near[1]])
        name = f"{e}_{n}"
        np.save(f"{OUT}/{name}_hsi.npy", x.astype(np.float16))
        np.save(f"{OUT}/{name}_rgb1m.npy", np.moveaxis(rgb1, -1, 0).astype(np.float16))
        np.save(f"{OUT}/{name}_rgb025.npy", np.moveaxis(rgb025, -1, 0))
        meta["tiles"].append(dict(name=name, valid=frac, corr_rgb=corr, shift_px=shift, rgb_shape=list(rgb.shape),
                                  max_refl=float(x.max()), p99_refl=float(np.percentile(x, 99))))
        print(meta["tiles"][-1], flush=True)
    np.save(f"{OUT}/wavelengths.npy", wl_keep)
    meta["bands"] = int(len(wl_keep))
    json.dump(meta, open(f"{OUT}/neon_meta.json", "w"), indent=1)
    if meta["tiles"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = meta["tiles"][0]["name"]
        x = np.load(f"{OUT}/{t}_hsi.npy").astype(np.float32)
        r = np.load(f"{OUT}/{t}_rgb1m.npy").astype(np.float32)
        near = [int(np.argmin(abs(wl_keep - v))) for v in (650.0, 550.0, 470.0)]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        hs = np.stack([x[i] for i in near], -1)
        ax[0].imshow(np.clip(hs / np.percentile(hs, 99), 0, 1)), ax[0].set_title(f"NEON HSI 1 m ({t})")
        ax[1].imshow(np.clip(np.moveaxis(r, 0, -1), 0, 1)), ax[1].set_title("RGB camera, averaged to 1 m")
        fig.savefig(f"{OUT}/neon_preview.png", dpi=80)
    print("done:", len(meta["tiles"]), "tiles,", meta["bands"], "bands", flush=True)


if __name__ == "__main__":
    main()
