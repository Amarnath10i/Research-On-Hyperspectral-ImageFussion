"""Numbers and image data for the paper (CPU is enough; run after a training run).

    python paper_analysis.py --mat /kaggle/input --cache /tmp/crop.npy --pred_dir /kaggle/input/<v3 output> --out out/paper

Reads the saved test predictions of the final model (test_pred_tta.npy) and of the first run,
recomputes bicubic and GSA, and writes:
    per_band.json    band-wise PSNR (peak 1), RMSE (DN) and UIQI for every method
    q2n.json         where the Q2n deficit comes from (bands, blocks), oracle and calibration checks
    visual.npz       false-colour composites (R-60, G-40, B-20), mean absolute error maps, spectra
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from hsifuse.baselines import gsa
from hsifuse.data import find_mat, load_chikusei, make_pairs, split
from hsifuse.metrics import DN_SCALE, evaluate, onions_quality, q2n
from hsifuse.ops import Degradation, wv2_srf

RGB = (60, 40, 20)   # false-colour bands used for Chikusei in TIP'26


def uiqi(g, x):
    """Universal image quality index per band over the whole image (B,) ."""
    g, x = g.flatten(1), x.flatten(1)
    mg, mx = g.mean(1), x.mean(1)
    vg, vx = g.var(1), x.var(1)
    cov = ((g - mg[:, None]) * (x - mx[:, None])).mean(1)
    return 4 * cov * mg * mx / ((vg + vx) * (mg ** 2 + mx ** 2))


def block_q(g, p, block=32):
    c = g.shape[0]
    q = lambda im: torch.round((im.double() * DN_SCALE).clamp(0, 65535))
    bl = lambda im: im.unfold(1, block, block).unfold(2, block, block).permute(1, 2, 0, 3, 4).reshape(-1, c, block * block)
    G, P = bl(q(g)).transpose(1, 2), bl(q(p)).transpose(1, 2)
    return onions_quality(G, P).norm(dim=1), G.std(1), (G - P).pow(2).mean(1).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--pred_dir", required=True, help="folder that contains the saved test predictions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    path = a.mat if a.mat.endswith(".mat") else find_mat(a.mat)
    _, te, _ = split(np.asarray(load_chikusei(path, a.cache or None)))
    gt = torch.from_numpy(np.ascontiguousarray(te)).double()
    deg = Degradation(wv2_srf()).double()
    lr, ms, _ = make_pairs(gt, deg)
    find = lambda pat: sorted(glob.glob(os.path.join(a.pred_dir, "**", pat), recursive=True))
    v3 = [p for p in find("test_pred_tta.npy") if "first_run" not in p][0]
    first = [p for p in find("test_pred.npy") if "first_run" in p][0]
    preds = {
        "Bicubic": F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1),
        "GSA": gsa(lr.float(), ms.float(), Degradation(wv2_srf())).double().clamp(0, 1),
        "PUFormer (first run)": torch.from_numpy(np.load(first)).double(),
        "PUFormer v3": torch.from_numpy(np.load(v3)).double(),
    }
    print("predictions:", {k: tuple(v.shape) for k, v in preds.items()}, "from", v3, first, flush=True)

    # 1) band-wise curves and overall metrics
    per_band, overall = {}, {}
    for name, x in preds.items():
        mse = (x - gt).pow(2).mean(dim=(0, 2, 3))
        per_band[name] = dict(psnr=(10 * torch.log10(1 / mse)).tolist(), rmse_dn=(mse.sqrt() * DN_SCALE).tolist(),
                              uiqi=torch.stack([uiqi(gt[i], x[i]) for i in range(len(gt))]).mean(0).tolist())
        overall[name] = evaluate(gt, x, full=True)
        print(name, {k: round(v, 4) for k, v in overall[name].items()}, flush=True)
    json.dump(dict(per_band=per_band, overall=overall, wavelengths_nm=np.linspace(363.0, 1018.0, 128).tolist()),
              open(os.path.join(a.out, "per_band.json"), "w"))

    # 2) Q2n analysis of the final model
    p = preds["PUFormer v3"]
    res = {"Q2n": float(np.mean([q2n(gt[i], p[i]) for i in range(8)]))}
    for name, bands in (("bands_0_7_exact", range(0, 8)), ("bands_0_7_104_127_exact", list(range(0, 8)) + list(range(104, 128)))):
        pp = p.clone(); pp[:, list(bands)] = gt[:, list(bands)]
        res[name] = float(np.mean([q2n(gt[i], pp[i]) for i in range(8)]))
    qs, stds, errs = zip(*[block_q(gt[i], p[i]) for i in range(8)])
    q, std, err = torch.cat(qs), torch.cat(stds), torch.cat(errs)
    order = (1 - q).argsort(descending=True)
    res["deficit_share_worst"] = {f"{f:g}": float((1 - q[order[:int(len(q) * f)]]).sum() / (1 - q).sum()) for f in (0.01, 0.05, 0.1, 0.25, 0.5)}
    worst = order[: len(q) // 10]
    rel = (err[worst] ** 2 / std[worst].clamp_min(1e-6) ** 2).mean(0)
    res["relative_error_worst10pct_by_group_of_8_bands"] = rel.view(16, 8).mean(1).tolist()
    res["per_tile_Q2n"] = [float(q2n(gt[i], p[i])) for i in range(8)]
    # noise floor estimates (DN) and oracle linear detail injection from the HR-MSI
    g_dn = gt * DN_SCALE
    d = g_dn[..., 1:] - g_dn[..., :-1]
    res["noise_spatial_mad_dn"] = (1.4826 * d.flatten(2).abs().median(-1).values.median(0).values / np.sqrt(2)).tolist()
    sres = g_dn[:, 1:-1] - 0.5 * (g_dn[:, :-2] + g_dn[:, 2:])
    res["noise_spectral_mad_dn"] = [None] + (1.4826 * sres.flatten(2).abs().median(-1).values.median(0).values / np.sqrt(1.5)).tolist() + [None]
    up = lambda z: F.interpolate(z, scale_factor=4, mode="bicubic", align_corners=False)
    hp_gt, hp_ms = gt - up(lr), ms - up(deg.D(ms, pad_mode="reflect"))
    A = torch.cat([hp_ms, torch.ones_like(hp_ms[:, :1])], 1).permute(0, 2, 3, 1).reshape(-1, 9)
    oracle = {}
    for b in list(range(10)) + [60, 100, 120, 127]:
        y = hp_gt[:, b].reshape(-1, 1)
        w = torch.linalg.lstsq(A, y).solution
        oracle[b] = dict(model=float((p[:, b] - gt[:, b]).pow(2).mean().sqrt() * DN_SCALE),
                         oracle_linear=float((y - A @ w).pow(2).mean().sqrt() * DN_SCALE),
                         bicubic=float(hp_gt[:, b].pow(2).mean().sqrt() * DN_SCALE))
    res["oracle_injection_rmse_dn"] = oracle
    calib = []
    for alpha in (0.0, 0.02, 0.05, 0.1, 0.2, 0.35):
        x = (p + alpha * (preds["Bicubic"] - p)).clamp(0, 1)
        calib.append(dict(alpha=alpha, PSNR=float(np.mean([10 * np.log10(1 / float((x[i] - gt[i]).pow(2).mean())) for i in range(8)])),
                          Q2n=float(np.mean([q2n(gt[i], x[i]) for i in range(8)]))))
    res["q2n_vs_psnr_blend_to_bicubic"] = calib
    json.dump(res, open(os.path.join(a.out, "q2n.json"), "w"), indent=1)
    print("Q2n analysis:", {k: v for k, v in res.items() if not isinstance(v, (list, dict))}, flush=True)

    # 3) visual data: composites, error maps, spectra (all 8 tiles, float16)
    comp = lambda x: x[:, list(RGB)].permute(0, 2, 3, 1).float().numpy().astype(np.float16)
    np.savez_compressed(os.path.join(a.out, "visual.npz"), rgb_bands=np.array(RGB), gt=comp(gt),
                        **{f"rgb_{i}": comp(x) for i, x in enumerate(preds.values())},
                        **{f"mae_{i}": ((x - gt).abs().mean(1) * DN_SCALE).float().numpy().astype(np.float16) for i, x in enumerate(preds.values())},
                        names=np.array(list(preds)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
