"""Where does each method's error lie? Split into the observed subspace and the fusion null space.

    python null_analysis.py --dataset pavia --mat ... --preds DIR --out DIR

With the separable exact operators, row(D) + row(R) and N(D) cap N(R) are orthogonal complements with closed-form
projectors (hsifuse.ops.null_split). The error of a fused image in the observed subspace is the part that
contradicts the observations (it would vanish if the output reproduced both inputs exactly); the error in the null
space is the part that no data step can correct, which only a prior can reduce. For every method with test
predictions (<preds>/**/bench-<dataset>-x4-<method>/test_pred_f16.npy), plus bicubic and GSA, writes the MSE in
each part and the PSNR the method would reach with an exact observed part (the null-space error alone).
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
from hsifuse.config import blur_of, make_deg
from hsifuse.data import load_dataset, make_pairs
from hsifuse.ops import null_split

METHODS = ("ssrnet", "psrt", "dct", "mimformer", "dhif", "puformer")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--mat", required=True)
    p.add_argument("--cache", default="")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--preds", default="/kaggle/input")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.pad = "reflect"
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, te, _, peak = load_dataset(a.dataset, a.mat, a.cache or None)
    gt = torch.from_numpy(np.asarray(te)).float().to(dev)
    deg = make_deg(a).to(dev)
    lr, ms, _ = make_pairs(gt, deg)
    preds = {"bicubic": F.interpolate(lr, scale_factor=a.scale, mode="bicubic", align_corners=False).clamp(0, 1),
             "gsa": gsa(lr, ms, deg).clamp(0, 1)}
    for m in METHODS:
        hits = sorted(glob.glob(os.path.join(a.preds, "**", f"bench-{a.dataset}-x{a.scale}-{m}", "test_pred_f16.npy"),
                                recursive=True))
        if hits:
            preds[m] = torch.from_numpy(np.load(hits[0]).astype(np.float32)).to(dev)
    n_px = gt[0].numel()
    res = dict(dataset=a.dataset, scale=a.scale, sigma=blur_of(a.scale)[1], methods={})
    for m, x in preds.items():
        obs, null = null_split(deg, x - gt)
        mse_o, mse_n = (obs / n_px).cpu().numpy(), (null / n_px).cpu().numpy()
        mse = mse_o + mse_n
        res["methods"][m] = dict(
            psnr=float(np.mean(10 * np.log10(1 / mse))), psnr_null_only=float(np.mean(10 * np.log10(1 / mse_n))),
            mse_observed=float(mse_o.mean()), mse_null=float(mse_n.mean()),
            null_share=float(np.mean(mse_n / mse)), per_image_null_share=[float(v) for v in mse_n / mse])
        r = res["methods"][m]
        print(f"{m:10s} PSNR {r['psnr']:.2f}  null-space share of the error {100 * r['null_share']:.1f}%  "
              f"PSNR with an exact observed part {r['psnr_null_only']:.2f}", flush=True)
    with open(os.path.join(a.out, "null_analysis.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
