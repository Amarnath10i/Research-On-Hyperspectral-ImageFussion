"""One network for a family of sensors: test conditions with changed blur, SRF, noise, and all three at once.

    python cond_eval.py --dataset pavia --mat ... --ckpt best_ema.pt --model puformer [--pad zeros] --out DIR

Meant for the operator-randomized models (train.py --rand_sigma / --rand_shift / --rand_snr), but runs on any
checkpoint. Each model is scored with
    fixed      the operator of the training protocol in its data steps (or no operator at all, for other networks)
    oracle     the true test operator in its data steps
    estimated  an operator estimated from the test pair alone: D (spatial) and R (spectral) commute, so for the true
               blur width sigma and SRF shift d, D_sigma(Y_M) = R_d(Y_H) exactly, and
                   (sigma_hat, d_hat) = argmin ||D_sigma(Y_M) - R_d(Y_H)||^2
               over a grid (0.025 sigma0 and 1 nm steps), per test image
Only operator-explicit models (PUFormer with a data step and a fixed exact or zero-border operator) use the oracle
and the estimate. GSA with the true operator is listed as a reference. Noise is drawn from a fixed seed per
condition, so every model sees the same inputs. NEON: the measured RGB cannot be re-simulated with another
response, so only the blur and the noise change. Writes DIR/cond_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from hsifuse.baselines import gsa
from hsifuse.config import DATASETS, MODELS, add_ablation_args, blur_of, build_model, set_metric_scale
from hsifuse.data import load_dataset, make_pairs
from hsifuse.metrics import evaluate
from hsifuse.ops import Degradation, dataset_srf

KEEP = ("PSNR", "SAM", "ERGAS", "SSIM", "RMSE_DN")


def conditions(neon: bool):
    """(name, blur factor x sigma0, SRF shift nm, SNR dB or None)."""
    c = [("nominal", 1.0, 0, None)]
    c += [(f"sigma x{f:g}", f, 0, None) for f in (0.5, 0.75, 1.25, 1.5, 2.0)]
    if not neon:
        c += [(f"shift {d:+d} nm", 1.0, d, None) for d in (-12, -8, -4, 4, 8, 12)]
    c += [(f"snr {n} dB", 1.0, 0, n) for n in (40, 35, 30)]
    c += [("sigma x1.5, shift +8 nm, snr 35 dB", 1.5, 8, 35), ("sigma x0.75, shift -8 nm, snr 40 dB", 0.75, -8, 40),
          ("sigma x2, shift +12 nm, snr 30 dB", 2.0, 12, 30)]
    if neon:
        c = [(n.replace(", shift +8 nm", "").replace(", shift -8 nm", "").replace(", shift +12 nm", ""), f, 0, s)
             for n, f, _, s in c]
    return c


def deg_at(a, sigma, shift=0.0, pad=None):
    return Degradation(dataset_srf(a.dataset, shift_nm=shift), scale=a.scale, ksize=blur_of(a.scale)[0],
                       sigma=sigma, pad=pad or a.pad, shift_nm=shift)


def add_noise(x, snr_db, g):
    p = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
    return x + torch.randn(x.shape, generator=g, device="cpu").to(x) * torch.sqrt(p / 10 ** (snr_db / 10))


@torch.no_grad()
def estimate(lr, ms, a, sig_grid, shift_grid, dev):
    """Per-image (sigma_hat, d_hat) minimising ||D_sigma(Y_M) - R_d(Y_H)||^2 (reflect border, as the simulation)."""
    ryh = torch.stack([deg_at(a, 1.0, float(d)).to(dev).R(lr) for d in shift_grid])          # (Gd, N, M, h, w)
    err = torch.stack([((deg_at(a, float(sg), pad="reflect").to(dev).D(ms, pad_mode="reflect")[None] - ryh) ** 2)
                       .flatten(2).mean(2) for sg in sig_grid])                              # (Gs, Gd, N)
    flat = err.flatten(0, 1).argmin(0).cpu().numpy()
    return sig_grid[flat // len(shift_grid)], shift_grid[flat % len(shift_grid)]


@torch.no_grad()
def run(model, lr, ms, degs=None):
    out = []
    for i in range(lr.shape[0]):
        if degs is not None:
            model.deg = degs[i]
        out.append(model(lr[i:i + 1], ms[i:i + 1]))
    return torch.cat(out).clamp(0, 1)


def score(gt, pred):
    m = evaluate(gt, pred)
    return {k: round(float(m[k]), 4) for k in KEEP if k in m}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--scale", type=int, default=4)
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
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    _, te, _, peak = load_dataset(a.dataset, a.mat, a.cache or None)
    set_metric_scale(a, peak)
    gt = torch.from_numpy(np.asarray(te)).to(dev)
    neon = a.dataset == "neon"
    s0 = blur_of(a.scale)[1]
    base = deg_at(a, s0).to(dev)
    gtb = gt[:, :base.srf.shape[0]]
    model = build_model(a, base).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    if hasattr(model, "amp"):
        model.amp = dev == "cuda"
    swappable = a.model == "puformer" and a.op == "exact" and bool(a.physics)

    sig_grid = np.round(np.arange(0.25 * s0, 2.5 * s0 + 1e-9, 0.025 * s0), 4)
    shift_grid = np.array([0.0]) if neon else np.arange(-16.0, 16.0 + 1e-9, 1.0)
    res = dict(dataset=a.dataset, scale=a.scale, model=a.model, pad=a.pad, op=a.op, physics=a.physics,
               ckpt=a.ckpt, nominal_sigma=s0, swappable=swappable, conditions={})
    for idx, (name, f, shift, snr) in enumerate(conditions(neon)):
        true = deg_at(a, f * s0, shift).to(dev)
        lr, ms, _ = make_pairs(gt, true)
        if snr:
            g = torch.Generator().manual_seed(1000 + idx)
            lr, ms = add_noise(lr, snr, g), add_noise(ms, snr, g)
        sig, sh = estimate(lr, ms, a, sig_grid, shift_grid, dev)
        row = dict(sigma=f * s0, shift_nm=shift, snr_db=snr, sigma_hat=[float(v) for v in sig],
                   shift_hat=[float(v) for v in sh], sigma_abs_err=float(np.abs(sig - f * s0).mean()),
                   shift_abs_err=float(np.abs(sh - shift).mean()))
        row["gsa_oracle"] = score(gtb, gsa(lr, ms, true).clamp(0, 1))
        if swappable:
            model.deg = base
        row["fixed"] = score(gtb, run(model, lr, ms))
        if swappable:
            row["oracle"] = score(gtb, run(model, lr, ms, [deg_at(a, f * s0, shift).to(dev)] * len(lr)))
            row["estimated"] = score(gtb, run(model, lr, ms, [deg_at(a, float(sg), float(d)).to(dev)
                                                              for sg, d in zip(sig, sh)]))
            model.deg = base
        res["conditions"][name] = row
        print(name, f"sigma_err {row['sigma_abs_err']:.3f} shift_err {row['shift_abs_err']:.2f}",
              {k: row[k]["PSNR"] for k in ("gsa_oracle", "fixed", "oracle", "estimated") if k in row}, flush=True)
    with open(os.path.join(a.out, "cond_eval.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
