"""Where the spectral-angle (SAM) error comes from (CPU is enough; run after test.py).

    python sam_analysis.py --dataset pavia --mat /kaggle/input --pred_dir /kaggle/input --out out/sam

Reads the saved test predictions of the final model (test_pred_tta.npy), recomputes bicubic and GSA
from the same observations, and writes sam.json:
    sam              mean SAM per method, over all pixels (as reported) and without zero-norm pixels
                     (the convention of SAM.m in the pansharpening toolbox)
    zero_pixels      ground-truth pixels whose spectrum is all zero, per test image
    by_brightness    mean SAM per decile of ground-truth spectral norm, per method, with the decile's norm (DN)
    darkest_share    share of the mean SAM of each method that comes from the darkest 5 / 10 / 20 % of pixels
    water            NDWI > 0 mask from the ground truth (green vs NIR): pixel share per image, and each
                     method's SAM over water and over all other pixels
    noise_floor      per-band noise of the ground truth (robust spectral second differences, DN) and the
                     mean angle this noise alone makes with the noise-free spectrum, overall and per decile
and sam_maps.npz (per-pixel angles of every method, the water mask, the spectral norm and the noise floor).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from hsifuse import metrics
from hsifuse.baselines import gsa
from hsifuse.data import load_dataset, make_pairs
from hsifuse.ops import CHIKUSEI_WL, PAVIA_WL, Degradation, dataset_srf

WL = {"chikusei": CHIKUSEI_WL, "pavia": PAVIA_WL}


def angles(g, x, eps=1e-8):
    """Per-pixel spectral angle in degrees, (N, H, W); the same formula as hsifuse.metrics.sam."""
    cos = (g * x).sum(1) / (g.norm(dim=1) * x.norm(dim=1) + eps)
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))


def band(wl, nm):
    return int(np.argmin(np.abs(wl - nm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--dataset", default="pavia", choices=["chikusei", "pavia"])
    ap.add_argument("--cache", default="")
    ap.add_argument("--pred_dir", required=True, help="folder that contains test_pred_tta.npy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    _, te, _, metrics.DN_SCALE = load_dataset(a.dataset, a.mat, a.cache or None)
    DN, wl = metrics.DN_SCALE, WL[a.dataset]
    gt = torch.from_numpy(np.ascontiguousarray(te)).double()
    deg = Degradation(dataset_srf(a.dataset)).double()
    lr, ms, _ = make_pairs(gt, deg)
    pred = sorted(glob.glob(os.path.join(a.pred_dir, "**", "test_pred_tta.npy"), recursive=True))
    pred = [p for p in pred if "first_run" not in p][0]
    preds = {"Bicubic": F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1),
             "GSA": gsa(lr.float(), ms.float(), Degradation(dataset_srf(a.dataset))).double().clamp(0, 1),
             "PUFormer": torch.from_numpy(np.load(pred)).double()}
    print("prediction:", pred, tuple(preds["PUFormer"].shape), flush=True)
    ang = {k: angles(gt, x) for k, x in preds.items()}                    # (N, H, W) degrees
    norm = gt.norm(dim=1) * DN                                             # (N, H, W) spectral norm in DN
    zero = norm == 0
    res = {"dataset": a.dataset, "prediction": pred,
           "zero_pixels": zero.flatten(1).sum(1).tolist(),
           "sam": {k: {"all_pixels": float(t.flatten(1).mean(1).mean()),
                       "nonzero_pixels": float(np.mean([float(t[i][~zero[i]].mean()) for i in range(len(t))])),
                       "per_image": t.flatten(1).mean(1).tolist()} for k, t in ang.items()}}

    # SAM by brightness decile (pooled over all test pixels)
    flat_norm = norm.flatten()
    edges = torch.quantile(flat_norm, torch.linspace(0, 1, 11, dtype=torch.float64))
    idx = torch.bucketize(flat_norm, edges[1:-1])
    res["by_brightness"] = {"norm_dn_median": [float(flat_norm[idx == d].median()) for d in range(10)],
                            **{k: [float(t.flatten()[idx == d].mean()) for d in range(10)] for k, t in ang.items()}}
    order = flat_norm.argsort()
    res["darkest_share"] = {k: {f"{f:g}": float(t.flatten()[order[: int(len(order) * f)]].sum() / t.flatten().sum())
                                for f in (0.05, 0.1, 0.2)} for k, t in ang.items()}

    # water: NDWI = (green - NIR) / (green + NIR) > 0 on the ground truth
    g_b, n_b = band(wl, 560), band(wl, 800)
    ndwi = (gt[:, g_b] - gt[:, n_b]) / (gt[:, g_b] + gt[:, n_b] + 1e-12)
    water = ndwi > 0
    res["water"] = {"bands_nm": [float(wl[g_b]), float(wl[n_b])], "share_per_image": water.flatten(1).double().mean(1).tolist(),
                    **{k: {"water": float(t[water].mean()) if water.any() else None, "other": float(t[~water].mean()),
                           "per_image_other": [float(t[i][~water[i]].mean()) for i in range(len(t))]}
                       for k, t in ang.items()}}

    # noise floor: robust per-band noise from spectral second differences, then the expected angle that
    # this noise alone makes with each (noise-free) spectrum, atan(|n| / |x|)
    g_dn = gt * DN
    sres = g_dn[:, 1:-1] - 0.5 * (g_dn[:, :-2] + g_dn[:, 2:])
    sigma = 1.4826 * sres.flatten(2).abs().median(-1).values.median(0).values / np.sqrt(1.5)   # bands 1..B-2
    sigma = torch.cat([sigma[:1], sigma, sigma[-1:]])                                           # edge bands: copy
    floor = torch.rad2deg(torch.atan(sigma.pow(2).sum().sqrt() / norm.clamp_min(1e-9)))
    res["noise_floor"] = {"sigma_dn": sigma.tolist(), "noise_norm_dn": float(sigma.pow(2).sum().sqrt()),
                          "mean_angle": float(floor.mean()), "per_image": floor.flatten(1).mean(1).tolist(),
                          "by_brightness": [float(floor.flatten()[idx == d].mean()) for d in range(10)]}
    with open(os.path.join(a.out, "sam.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(a.out, "sam_maps.npz"), water=water.numpy(), norm_dn=norm.float().numpy().astype(np.float16),
                        noise_floor=floor.float().numpy().astype(np.float16),
                        **{k.lower(): t.float().numpy().astype(np.float16) for k, t in ang.items()})
    print(json.dumps({k: res[k] for k in ("zero_pixels", "sam", "darkest_share")}, indent=1), flush=True)
    print("water share:", [round(v, 4) for v in res["water"]["share_per_image"]],
          {k: (res["water"][k]["water"], res["water"][k]["other"]) for k in ang}, flush=True)
    print("noise floor:", res["noise_floor"]["mean_angle"], "by decile", [round(v, 3) for v in res["noise_floor"]["by_brightness"]])
    print("by brightness:", {k: [round(v, 3) for v in vals] for k, vals in res["by_brightness"].items()}, flush=True)


if __name__ == "__main__":
    main()
