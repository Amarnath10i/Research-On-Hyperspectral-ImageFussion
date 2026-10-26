"""Full-resolution fusion of real data (NEON): the measured 1 m spectrometer image with the 0.25 m camera RGB.

    python neon_fullres.py --dataset neon --scale 4 --model puformer --ckpt best_ema.pt --out DIR

No reference exists at 0.25 m, so the fused image X^ is scored without one:
    D_lambda  spectral distortion, 1 - mean band-wise Q (UIQI) between D(X^), the fused image blurred and decimated
              with the protocol operator, and the measured 1 m HSI (the hybrid form of Khan et al.)
    D_s       spatial distortion, mean over HSI bands b and RGB channels m of |Q(X^_b, Y_M,m) - Q(Y_H,b, D(Y_M)_m)|
              (the QNR spatial term)
    HQNR      (1 - D_lambda)(1 - D_s)
and the consistency PSNR(D X^, Y_H), PSNR(R X^, Y_M) and SAM(D X^, Y_H). Q is computed on 32x32 blocks at 0.25 m
and 8x8 blocks at 1 m. For each test tile, the central 512x512 HSI region and the matching 2048x2048 RGB are fused
in 256x256 HSI tiles (1024x1024 at 0.25 m; 128x128 if a network runs out of memory). Bicubic upsampling and GSA are
scored as references. Writes DIR/neon_fullres.json and DIR/neon_fullres_crop.npz (a 256 x 256 crop of the first test tile at 0.25 m:
the fused image, GSA, the camera RGB and the HSI, in the bands nearest 650 / 550 / 470 nm).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from hsifuse.baselines import gsa
from hsifuse.config import MODELS, add_ablation_args, blur_of, build_model, make_deg
from hsifuse.ops import dataset_wl, neon_file

REGION = 512                     # central HSI region of every test tile (pixels at 1 m)


def q_blocks(x, y, block, chunk=16):
    """Mean UIQI over non-overlapping block x block windows; x, y: (N, H, W) -> (N,) (in chunks of bands)."""
    out = []
    for i in range(0, x.shape[0], chunk):
        a, b = x[i:i + chunk, None].float(), y[i:i + chunk, None].float()
        ma, mb = F.avg_pool2d(a, block), F.avg_pool2d(b, block)
        va = (F.avg_pool2d(a * a, block) - ma ** 2).clamp(min=0)
        vb = (F.avg_pool2d(b * b, block) - mb ** 2).clamp(min=0)
        cab = F.avg_pool2d(a * b, block) - ma * mb
        q = 4 * cab * ma * mb / ((va + vb) * (ma ** 2 + mb ** 2) + 1e-12)
        out.append(q.flatten(1).mean(1))
    return torch.cat(out)


def psnr(a, b):
    return float(10 * torch.log10(1.0 / F.mse_loss(a.float(), b.float())))


def sam_deg(a, b):
    num = (a * b).sum(0)
    den = a.norm(dim=0) * b.norm(dim=0)
    ok = den > 1e-8
    return float(torch.rad2deg(torch.acos((num[ok] / den[ok]).clamp(-1, 1))).mean())


@torch.no_grad()
def no_reference(xh, yh, ym, deg):
    """xh (B, 4H, 4W) fused, yh (B, H, W) measured HSI, ym (3, 4H, 4W) measured RGB."""
    dx = deg.D(xh[None], pad_mode="reflect")[0]
    dym = deg.D(ym[None], pad_mode="reflect")[0]
    d_lambda = float(1 - q_blocks(dx, yh, 8).mean())
    qh = torch.stack([q_blocks(xh, ym[m].expand_as(xh), 32) for m in range(3)], 1)          # (B, 3)
    ql = torch.stack([q_blocks(yh, dym[m].expand_as(yh), 8) for m in range(3)], 1)
    d_s = float((qh - ql).abs().mean())
    return dict(D_lambda=d_lambda, D_s=d_s, HQNR=(1 - d_lambda) * (1 - d_s), cons_H_dB=psnr(dx, yh),
                cons_M_dB=psnr(deg.R(xh[None])[0], ym), SAM_DxH=sam_deg(dx, yh))


@torch.no_grad()
def fuse(fn, yh, ym, tile):
    """Tile-wise fusion of yh (B, H, W) with ym (3, 4H, 4W); fn(lr, ms) -> (1, B, 4h, 4w)."""
    b, h, w = yh.shape
    out = torch.zeros(b, 4 * h, 4 * w, device=yh.device)
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            lr = yh[None, :, r:r + tile, c:c + tile]
            ms = ym[None, :, 4 * r:4 * (r + tile), 4 * c:4 * (c + tile)]
            out[:, 4 * r:4 * (r + tile), 4 * c:4 * (c + tile)] = fn(lr, ms)[0].float().clamp(0, 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="")
    p.add_argument("--dataset", default="neon", choices=["neon"])
    p.add_argument("--scale", type=int, default=4, choices=[4])
    p.add_argument("--cache", default="")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="puformer", choices=MODELS)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--pad", default="reflect", choices=["reflect", "zeros"])
    add_ablation_args(p)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fit = json.load(open(neon_file("neon_fit.json")))
    deg = make_deg(a).to(dev)
    true = make_deg(argparse.Namespace(**dict(vars(a), pad="reflect"))).to(dev)       # the simulation operator
    model = build_model(a, deg).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    if hasattr(model, "amp"):
        model.amp = dev == "cuda"
    wl = dataset_wl("neon")
    rgb_bands = [int(np.argmin(abs(wl - t))) for t in (650.0, 550.0, 470.0)]
    o = (1000 - REGION) // 2
    keys = ("D_lambda", "D_s", "HQNR", "cons_H_dB", "cons_M_dB", "SAM_DxH")
    res = dict(model=a.model, ckpt=a.ckpt, tiles={}, region=REGION, quadrant=REGION // 2)
    crops = {}
    bic = lambda lr, ms: F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
    for name in fit["test"]:
        hsi = np.load(neon_file(f"{name}_hsi.npy"), mmap_mode="r")
        msi = np.load(neon_file(f"{name}_msi025.npy"), mmap_mode="r")
        quads, q = [], REGION // 2
        for r in (o, o + q):
            for c in (o, o + q):
                yh = torch.from_numpy(np.array(hsi[:, r:r + q, c:c + q], dtype=np.float32) / fit["norm"]).to(dev)
                ym = torch.from_numpy(np.array(msi[:, 4 * r:4 * (r + q), 4 * c:4 * (c + q)], dtype=np.float32)).to(dev)
                xg = fuse(lambda lr, ms: gsa(lr, ms, true), yh, ym, q)
                rows = {"bicubic": no_reference(fuse(bic, yh, ym, q), yh, ym, true),
                        "gsa": no_reference(xg, yh, ym, true)}
                for tile in (q, q // 2, q // 4):
                    try:
                        xh = fuse(lambda lr, ms: model(lr, ms), yh, ym, tile)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"out of memory at tile {tile}, retrying smaller", flush=True)
                rows["model"] = dict(no_reference(xh, yh, ym, true), tile=tile)
                quads.append(rows)
                if not crops and name == fit["test"][0]:                        # one 256 x 256 crop at 0.25 m
                    cs = slice(384, 640)
                    up = F.interpolate(yh[None, rgb_bands], scale_factor=4, mode="nearest")[0]
                    crops = {k: v[:, cs, cs].cpu().numpy().astype(np.float16) for k, v in
                             (("model", xh[rgb_bands]), ("gsa", xg[rgb_bands]), ("rgb", ym), ("hsi", up))}
                del xh, xg
        res["tiles"][name] = {m: {k: float(np.mean([qq[m][k] for qq in quads])) for k in keys}
                              for m in ("bicubic", "gsa", "model")}
        print(name, {m: {k: round(v, 4) for k, v in d.items()} for m, d in res["tiles"][name].items()}, flush=True)
    res["mean"] = {m: {k: float(np.mean([res["tiles"][t][m][k] for t in res["tiles"]])) for k in keys}
                   for m in ("bicubic", "gsa", "model")}
    print("mean:", res["mean"], flush=True)
    with open(os.path.join(a.out, "neon_fullres.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(os.path.join(a.out, "neon_fullres_crop.npz"), **crops)


if __name__ == "__main__":
    main()
