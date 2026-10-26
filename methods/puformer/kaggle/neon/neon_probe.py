"""Probe of the NEON airborne data for a real hyperspectral + RGB fusion test (Kaggle CPU kernel, internet on).

NEON's Airborne Observation Platform flies an imaging spectrometer (DP3.30006.001: 1 m surface reflectance,
426 bands, 380-2510 nm, 1 km x 1 km HDF5 tiles) together with an RGB camera (DP3.30010.001: 10 cm
orthomosaic, 1 km x 1 km GeoTIFF tiles), both on the same UTM tile grid. This script finds a site and year
with both products, downloads one matching tile pair, and reports sizes, wavelengths, georeferencing and how
well the two images line up (correlation and phase-correlation shift between the RGB at 1 m and the
hyperspectral bands nearest to red, green and blue). Writes /kaggle/working/neon_probe.json and previews.
"""
import json
import os
import re
import subprocess
import time

import numpy as np
import requests

API = "https://data.neonscience.org/api/v0"
HSI, RGB = "DP3.30006.001", "DP3.30010.001"
SITES = ["SERC", "HARV", "SCBI", "UKFS", "ORNL", "TALL", "OSBS", "BART", "MLBS", "GRSM", "DELA", "LENO"]
OUT = "/kaggle/working"
UA = {"User-Agent": "puformer-neon-probe/1.0"}


def get(url, **kw):
    for i in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
            if r.status_code in (403, 404):
                break                                    # not served: no point in retrying
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print("retry", url, e, flush=True)
            time.sleep(5)
    raise RuntimeError(url)


def months(product):
    d = get(f"{API}/products/{product}").json()["data"]
    return {s["siteCode"]: set(s["availableMonths"]) for s in d["siteCodes"]}


def files(product, site, month):
    return get(f"{API}/data/{product}/{site}/{month}").json()["data"]["files"]


def tile_xy(name):
    m = re.search(r"_(\d{6})_(\d{7})_", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def download(url, path):
    with requests.get(url, headers=UA, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    return os.path.getsize(path)


def main():
    subprocess.run("pip install -q tifffile imagecodecs", shell=True)
    report = {}
    mh, mr = months(HSI), months(RGB)
    pairs = []
    for site in SITES:
        common = sorted(mh.get(site, set()) & mr.get(site, set()))
        report[site] = common
        print(site, "common months:", common, flush=True)
        if common:
            pairs.append((site, common[-1]))
    # newest months first; some site-months are not served (403), so try until one works for both products
    tries = sorted(((m, s) for s in SITES for m in report.get(s, [])), reverse=True)
    report["attempts"] = []
    for month, site in tries[:40]:
        try:
            fh = [f for f in files(HSI, site, month) if f["name"].endswith("reflectance.h5")]
            fr = [f for f in files(RGB, site, month) if f["name"].endswith("image.tif")]
        except RuntimeError as e:
            report["attempts"].append([site, month, "error"])
            continue
        th, tr = {tile_xy(f["name"]): f for f in fh}, {tile_xy(f["name"]): f for f in fr}
        common_tiles = sorted(set(th) & set(tr) - {None})
        report["attempts"].append([site, month, len(common_tiles)])
        print("try", site, month, len(fh), len(fr), len(common_tiles), flush=True)
        if common_tiles:
            break
    else:
        json.dump(report, open(f"{OUT}/neon_probe.json", "w"), indent=1)
        raise SystemExit("no site-month with both products accessible")
    print(f"{site} {month}: {len(fh)} HSI tiles, {len(fr)} RGB tiles, {len(common_tiles)} in common", flush=True)
    report["chosen"] = dict(site=site, month=month, hsi_tiles=len(fh), rgb_tiles=len(fr), common=len(common_tiles),
                            hsi_size_MB=float(np.median([int(f["size"]) for f in fh]) / 1e6),
                            rgb_size_MB=float(np.median([int(f["size"]) for f in fr]) / 1e6))
    # one tile near the middle of the flight box
    xy = common_tiles[len(common_tiles) // 2]
    ph, pr = f"{OUT}/hsi.h5", f"{OUT}/rgb.tif"
    print("downloading", th[xy]["name"], download(th[xy]["url"], ph) / 1e6, "MB", flush=True)
    print("downloading", tr[xy]["name"], download(tr[xy]["url"], pr) / 1e6, "MB", flush=True)

    import h5py
    import tifffile
    with h5py.File(ph, "r") as f:
        g = f[site]["Reflectance"]
        refl = g["Reflectance_Data"]
        wl = g["Metadata"]["Spectral_Data"]["Wavelength"][()]
        attrs = {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in refl.attrs.items()}
        mapinfo = g["Metadata"]["Coordinate_System"]["Map_Info"][()]
        shape = refl.shape
        # bands nearest to blue / green / red, and a strided full cube for statistics
        idx = [int(np.argmin(abs(wl - w))) for w in (470.0, 550.0, 650.0)]
        bgr = np.stack([refl[:, :, i].astype(np.float32) for i in idx], 0)
    nodata = float(attrs.get("Data_Ignore_Value", -9999))
    scale = float(attrs.get("Scale_Factor", 10000))
    rgb = tifffile.imread(pr)
    print("HSI", shape, "wl", float(wl[0]), "-", float(wl[-1]), "nm; RGB", rgb.shape, rgb.dtype, flush=True)
    rgb = rgb[..., :3] if rgb.shape[-1] in (3, 4) else np.moveaxis(rgb, 0, -1)[..., :3]
    h, w = shape[0], shape[1]
    fy, fx = rgb.shape[0] // h, rgb.shape[1] // w
    rgb1 = rgb[:h * fy, :w * fx].reshape(h, fy, w, fx, 3).mean((1, 3)).astype(np.float32)   # 10 cm -> 1 m
    valid = (bgr[0] != nodata) & (bgr[1] != nodata) & (bgr[2] != nodata)
    corr, shifts = [], []
    for c, hs in zip((2, 1, 0), bgr):          # RGB channel order: R, G, B; bgr holds blue, green, red
        a, b = rgb1[..., c][valid], hs[valid] / scale
        corr.append(float(np.corrcoef(a, b)[0, 1]))
        A = np.fft.fft2(np.where(valid, rgb1[..., c] - a.mean(), 0))
        B = np.fft.fft2(np.where(valid, hs / scale - b.mean(), 0))
        p = np.fft.ifft2(A * np.conj(B) / (np.abs(A * np.conj(B)) + 1e-9)).real
        dy, dx = np.unravel_index(np.argmax(p), p.shape)
        shifts.append([int(dy if dy < h // 2 else dy - h), int(dx if dx < w // 2 else dx - w)])
    report["tile"] = dict(xy=xy, hsi_shape=list(shape), wl_first=float(wl[0]), wl_last=float(wl[-1]),
                          n_bands=int(len(wl)), map_info=str(mapinfo)[:200], rgb_shape=list(rgb.shape),
                          rgb_dtype=str(rgb.dtype), rgb_factor=[fy, fx], valid_fraction=float(valid.mean()),
                          corr_RGB=corr, phase_shift_px=shifts, nodata=nodata, scale=scale,
                          hsi_attrs={k: attrs[k] for k in list(attrs)[:12]})
    print(json.dumps(report["tile"], indent=1), flush=True)
    json.dump(report, open(f"{OUT}/neon_probe.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        hsrgb = np.stack([bgr[2], bgr[1], bgr[0]], -1) / scale
        ax[0].imshow(np.clip(hsrgb / np.percentile(hsrgb[valid], 99), 0, 1)), ax[0].set_title("HSI (1 m)")
        ax[1].imshow(np.clip(rgb1 / 255.0, 0, 1)), ax[1].set_title("RGB camera (averaged to 1 m)")
        fig.savefig(f"{OUT}/neon_preview.png", dpi=80)
    except Exception as e:
        print("preview failed", e)
    for p in (ph, pr):
        os.remove(p)


if __name__ == "__main__":
    main()
