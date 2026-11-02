"""What does the HR-MSI add to the LR-HSI? An information budget of the true images (no network involved).

    python msi_value.py --dataset chikusei --mat ... --out DIR [--scales 4,8,16]

In the eigenbasis of the exact operators (Theorem 1), every component of the HR-HSI is seen by the LR-HSI (low
spatial frequencies, every band), seen only by the HR-MSI (high spatial frequencies in the M spectral directions
the MSI responses span), or by neither (the fusion null space). For the test images this reports
    energy shares       of the three parts, and the MSI's share of the fine-detail energy (everything the LR-HSI
                        cannot see), which single-image super-resolution has to invent entirely
    data-only PSNR      PSNR of the image that keeps exactly the measured components (noise-free exact data, zero
                        elsewhere): what the measurements alone determine, for the LR-HSI alone, the HR-MSI alone
                        and both
Writes DIR/msi_value.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from hsifuse.config import DATASETS, make_deg
from hsifuse.data import load_dataset
from hsifuse.ops import kron_factors


def psnr(x, y):
    return float(10 * math.log10(1.0 / max(float(((x - y) ** 2).mean()), 1e-20)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--cache", default="")
    p.add_argument("--scales", default="4")
    p.add_argument("--pad", default="reflect")
    p.add_argument("--sensor", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _, te, _, _ = load_dataset(a.dataset, a.mat, a.cache or None)
    gt = torch.from_numpy(np.asarray(te)).double()
    res = dict(dataset=a.dataset, scales={})
    for s in (int(v) for v in a.scales.split(",")):
        a.scale = s
        if gt.shape[-1] % s or gt.shape[-2] % s:
            continue
        deg = make_deg(a)
        kf = kron_factors(deg, gt.shape[-2], gt.shape[-1], "cpu")
        seen_lr = kf.obs_spatial.double()[None, None]                      # (1, 1, H, W)
        seen_ms = kf.obs_spectral.double()[None, :, None, None]           # (1, B, 1, 1)
        rows = []
        for i in range(gt.shape[0]):
            x = gt[i:i + 1].float()
            e = kf.to_eig(x).double()
            e2 = e ** 2
            tot = float(e2.sum())
            lr_part = float((e2 * seen_lr).sum())
            ms_only = float((e2 * (1 - seen_lr) * seen_ms).sum())
            null = float((e2 * (1 - seen_lr) * (1 - seen_ms)).sum())
            keep = lambda m: kf.from_eig((e * m).float()).double()
            union = seen_lr + seen_ms - seen_lr * seen_ms
            rows.append(dict(lr_share=lr_part / tot, msi_only_share=ms_only / tot, null_share=null / tot,
                             msi_share_of_detail=ms_only / max(ms_only + null, 1e-30),
                             psnr_lr_only=psnr(keep(seen_lr.expand_as(e)), x.double()),
                             psnr_msi_only=psnr(keep(seen_ms.expand_as(e)), x.double()),
                             psnr_both=psnr(keep(union.expand_as(e)), x.double())))
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res["scales"][f"x{s}"] = dict(mean=mean, per_image=rows, bands=int(gt.shape[1]),
                                      msi_bands=int(deg.srf.shape[1]))
        print(a.dataset, f"x{s}", {k: round(v, 4) for k, v in mean.items()}, flush=True)
        with open(os.path.join(a.out, "msi_value.json"), "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
