"""Operators that are estimated rather than known, and the fixed-point residual of each operator model.

    python blind_eval.py --dataset pavia --mat ... --ckpt full.pt [--ckpt_zeros zeros.pt] [--ckpt_learned learned.pt] --out DIR

1. Blur estimated from the test pair. D (spatial) and R (spectral) commute, so for the true blur
   D_sigma(Y_M) = R(Y_H) exactly; sigma_hat = argmin_sigma ||D_sigma(Y_M) - R(Y_H)||^2 is found by a grid search
   on each test image, using only the observations and the training SRF. PUFormer is then run with
   D(sigma_hat) (estimated swap) and compared with the training operator (fixed) and the true one (oracle),
   for test blurs sigma in {0.75, 1, 1.25, 1.5} x the nominal width, with and without 35 dB input noise.
2. Fixed-point residual. With the network's operator D~, the data step moves the ground truth by
   s^2 D~^T (D~ X - Y_H); it is zero for the exact operator and, for the zero border or a learned operator,
   measures the bias the prior has to undo. Reported relative to s^2 ||D~^T D~ X|| on the test images and on
   64x64 validation-sized crops, where the border is a larger share of the pixels.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from hsifuse import metrics
from hsifuse.config import DATASETS, blur_of, set_metric_scale
from hsifuse.data import load_dataset, make_pairs
from hsifuse.metrics import evaluate
from hsifuse.models import build
from hsifuse.ops import Degradation, LearnedDegradation, dataset_srf


def deg_at(ds, s, sigma, pad="reflect"):
    return Degradation(dataset_srf(ds), scale=s, ksize=blur_of(s)[0], sigma=sigma, pad=pad)


@torch.no_grad()
def estimate_sigma(yh, ym, ds, s, grid):
    """Per-image sigma minimising ||D_sigma(Y_M) - R(Y_H)||^2 (reflect border, as the simulation)."""
    deg0 = deg_at(ds, s, 1.0).to(yh.device)
    target = deg0.R(yh)                                         # M-band image at LR
    errs = []
    for sg in grid:
        d = deg_at(ds, s, float(sg)).to(yh.device)
        errs.append(((d.D(ym, pad_mode="reflect") - target) ** 2).flatten(1).mean(1))
    errs = torch.stack(errs, 1)                                 # (N, G)
    return grid[errs.argmin(1).cpu().numpy()]


@torch.no_grad()
def run(model, lr, ms, degs=None):
    out = []
    for i in range(lr.shape[0]):
        if degs is not None:
            model.deg = degs[i]
        out.append(model(lr[i:i + 1], ms[i:i + 1]))
    return torch.cat(out).clamp(0, 1)


def add_noise(x, snr_db, g):
    p = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
    return x + torch.randn(x.shape, generator=g).to(x) * torch.sqrt(p / 10 ** (snr_db / 10))


def load_model(path, ds, s, dev, op="exact", pad="reflect"):
    deg = deg_at(ds, s, blur_of(s)[1], pad).to(dev)
    b, m = deg.srf.shape
    opmod = LearnedDegradation(b, m, scale=s, ksize=deg.ksize).to(dev) if op == "learned" else deg
    net = build("puformer", opmod).to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    net.amp = dev == "cuda"
    return net


@torch.no_grad()
def fixed_point_residual(net_deg, true_deg, x):
    """||s^2 D~^T(D~ X - D X)|| / ||s^2 D~^T D~ X|| per image (D = simulation operator)."""
    s2 = true_deg.scale ** 2
    yh = true_deg.D(x, pad_mode="reflect")
    num = (s2 * net_deg.Dt(net_deg.D(x) - yh)).flatten(1).norm(dim=1)
    den = (s2 * net_deg.Dt(net_deg.D(x))).flatten(1).norm(dim=1)
    return (num / den).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--cache", default="")
    p.add_argument("--ckpt", required=True, help="full PUFormer (exact reflect operator)")
    p.add_argument("--ckpt_zeros", default="")
    p.add_argument("--ckpt_learned", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.pad = "reflect"
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, te, va, peak = load_dataset(a.dataset, a.mat, a.cache or None)
    set_metric_scale(a, peak)
    gt = torch.from_numpy(np.asarray(te)).to(dev)
    s0 = blur_of(a.scale)[1]
    grid = np.round(np.arange(0.25 * s0, 2.0 * s0 + 1e-9, 0.025 * s0), 4)
    res = dict(dataset=a.dataset, scale=a.scale, nominal_sigma=s0, grid_step=float(grid[1] - grid[0]))

    nets = {"exact": load_model(a.ckpt, a.dataset, a.scale, dev)}
    if a.ckpt_zeros:
        nets["zeros"] = load_model(a.ckpt_zeros, a.dataset, a.scale, dev, pad="zeros")
    if a.ckpt_learned:
        nets["learned"] = load_model(a.ckpt_learned, a.dataset, a.scale, dev, op="learned")
    base = {k: n.deg for k, n in nets.items()}

    g = torch.Generator().manual_seed(0)
    rows = {}
    for f in (0.75, 1.0, 1.25, 1.5):
        for snr in (None, 35):
            true = deg_at(a.dataset, a.scale, f * s0).to(dev)
            lr, ms, _ = make_pairs(gt, true)
            if snr:
                lr, ms = add_noise(lr, snr, g), add_noise(ms, snr, g)
            sig = estimate_sigma(lr, ms, a.dataset, a.scale, grid)
            key = f"sigma={f * s0:g}" + (f",snr={snr}" if snr else "")
            row = dict(sigma_true=f * s0, sigma_hat=[float(v) for v in sig],
                       sigma_abs_err=float(np.abs(sig - f * s0).mean()))
            for name, net in nets.items():
                net.deg = base[name]
                row[f"{name}_fixed"] = evaluate(gt, run(net, lr, ms))["PSNR"]
                if name != "learned":
                    pad = "zeros" if name == "zeros" else "reflect"
                    est = [deg_at(a.dataset, a.scale, float(v), pad).to(dev) for v in sig]
                    row[f"{name}_estimated"] = evaluate(gt, run(net, lr, ms, est))["PSNR"]
                    orc = deg_at(a.dataset, a.scale, f * s0, pad).to(dev)
                    row[f"{name}_oracle"] = evaluate(gt, run(net, lr, ms, [orc] * len(lr)))["PSNR"]
                    net.deg = base[name]
            rows[key] = row
            print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "sigma_hat"},
                  flush=True)
    res["blind"] = rows

    true = deg_at(a.dataset, a.scale, s0).to(dev)
    crops = gt[:, :, :64, :64].contiguous()
    fp = {}
    for name, d in [("exact", true), ("zeros", deg_at(a.dataset, a.scale, s0, "zeros").to(dev))] + \
                   ([("learned", nets["learned"].deg)] if "learned" in nets else []):
        fp[name] = dict(test=float(np.mean(fixed_point_residual(d, true, gt))),
                        crop64=float(np.mean(fixed_point_residual(d, true, crops))))
    res["fixed_point_residual"] = fp
    print("fixed-point residual:", fp, flush=True)
    with open(os.path.join(a.out, "blind.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
