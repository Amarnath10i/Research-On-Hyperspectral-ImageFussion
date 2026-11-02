"""Numerical check of the observed / null-space split (Theorem kron.(c), hsifuse.ops.null_split) on the test images.

    python null_check.py --dataset chikusei --scale 4 --mat ... --cache ... --out DIR [--ckpt (ignored)]

For the exact operators of the protocol (reflect border):
  basis       relative error of the round trip from_eig(to_eig(x)) (orthonormality of V (x) U_r (x) U_c)
  null leak   ||D x_null|| / ||D x|| and ||R x_null|| / ||R x|| for the null-space component x_null of x (zero if the
              computed null space is the null space of the actual operators)
  projector   idempotence ||P(P x) - P x|| / ||x|| and symmetry |<P x, y> - <x, P y>| / (||x|| ||y||)
  pythagoras  relative gap between ||x||^2 and ||x_obs||^2 + ||x_null||^2
  fraction    null-space fraction of the unknowns (Theorem kron.(c): (1 - M/B)(1 - 1/s^2))
x: the test images, random images, and the error of a bicubic estimate (a typical error image).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from hsifuse.config import DATASETS, make_deg
from hsifuse.data import load_dataset, make_pairs
from hsifuse.ops import kron_factors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--cache", default="")
    p.add_argument("--out", required=True)
    a, _ = p.parse_known_args()
    a.pad = "reflect"
    os.makedirs(a.out, exist_ok=True)
    _, te, _, _ = load_dataset(a.dataset, a.mat, a.cache or None)
    gt = torch.from_numpy(np.asarray(te)).float()
    deg = make_deg(a)
    kf = kron_factors(deg, gt.shape[-2], gt.shape[-1], "cpu")
    null = (1 - kf.obs_spectral)[None, :, None, None] * (1 - kf.obs_spatial)[None, None]

    def split(x):
        e = kf.to_eig(x)
        return kf.from_eig(e * (1 - null)), kf.from_eig(e * null)

    lr, _, _ = make_pairs(gt, deg)
    bic = F.interpolate(lr, size=gt.shape[-2:], mode="bicubic", align_corners=False)
    g = torch.Generator().manual_seed(0)
    sets = {"test images": gt, "random images": torch.rand(gt.shape, generator=g), "bicubic error": bic - gt}
    res = dict(dataset=a.dataset, scale=a.scale, shape=list(gt.shape[1:]),
               null_fraction=float(null.mean()), predicted_fraction=float(
                   (1 - deg.srf.shape[1] / deg.srf.shape[0]) * (1 - 1 / a.scale ** 2)), sets={})
    y = torch.rand(gt.shape, generator=g)
    for name, x in sets.items():
        xo, xn = split(x)
        rt = kf.from_eig(kf.to_eig(x))
        pxo, _ = split(xo)
        yo, _ = split(y)
        dx, rx = deg.D(x, pad_mode="reflect"), deg.R(x)
        res["sets"][name] = dict(
            basis_roundtrip=float((rt - x).norm() / x.norm()),
            null_leak_D=float(deg.D(xn, pad_mode="reflect").norm() / dx.norm()),
            null_leak_R=float(deg.R(xn).norm() / rx.norm()),
            idempotence=float((pxo - xo).norm() / x.norm()),
            symmetry=float(abs((xo * y).sum() - (x * yo).sum()) / (x.norm() * y.norm())),
            pythagoras=float(abs(x.pow(2).sum() - xo.pow(2).sum() - xn.pow(2).sum()) / x.pow(2).sum()),
            null_share=float(xn.pow(2).sum() / x.pow(2).sum()))
        print(name, res["sets"][name], flush=True)
    with open(os.path.join(a.out, "null_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
