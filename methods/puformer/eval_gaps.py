"""Evaluations that most Chikusei fusion papers skip (run after train.py).

1. Observation consistency: how well the fused image reproduces its own inputs,
   PSNR(D X^, Y_H) and PSNR(R X^, Y_M), with D exactly as in the simulation (reflect border).
   A high value means the network did not hallucinate content that contradicts the measurements.
2. Degradation mismatch: test inputs made with a different blur sigma or a shifted
   SRF than training. Each model is scored twice:
     fixed  - the training operator is kept
     swap   - the true test operator is plugged into PUFormer's physics steps
              (possible only for operator-explicit models; no retraining)
3. Input noise: Gaussian noise at 40 / 35 / 30 dB SNR on both LR-HSI and HR-MSI.

    python eval_gaps.py --mat ... --ckpt out/puformer/best_ema.pt --model puformer --pad reflect --out out/puformer
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from hsifuse.baselines import gsa
from hsifuse import metrics
from hsifuse.data import load_dataset, make_pairs
from hsifuse.evaluation import consistency
from hsifuse.metrics import evaluate
from hsifuse.models import build
from hsifuse.ops import Degradation, dataset_srf


@torch.no_grad()
def run(model, lr, ms):
    return torch.cat([model(lr[i:i + 1], ms[i:i + 1]) for i in range(lr.shape[0])]).clamp(0, 1)


def add_noise(x, snr_db, g):
    p = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
    return x + torch.randn(x.shape, generator=g, device="cpu").to(x) * torch.sqrt(p / 10 ** (snr_db / 10))


def score(gt, pred, lr, ms, deg):
    m = dict(evaluate(gt, pred), **consistency(pred, lr, ms, deg))
    return {k: round(float(v), 4) for k, v in m.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--dataset", default="chikusei", choices=["chikusei", "pavia"])
    p.add_argument("--cache", default="")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="puformer")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--pad", default="reflect", choices=["reflect", "zeros"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    _, te, _, metrics.DN_SCALE = load_dataset(a.dataset, a.mat, a.cache or None)
    gt = torch.from_numpy(te).to(dev)

    base = Degradation(dataset_srf(a.dataset), pad=a.pad).to(dev)
    model = build(a.model, base, **(dict(width=a.width, stages=a.stages) if a.model == "puformer" else {})).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    if a.model == "puformer":
        model.amp = dev == "cuda"
    swappable = hasattr(model, "deg") and a.model == "puformer"

    settings = [("nominal", 2.0, 0.0)] + [(f"sigma={s}", s, 0.0) for s in (1.5, 2.5, 3.0)] + \
               [(f"srf_shift={d:+d}nm", 2.0, d) for d in (-8, 8)]
    res = {}
    for name, sigma, shift in settings:
        true = Degradation(dataset_srf(a.dataset, shift_nm=shift), sigma=sigma, pad=a.pad).to(dev)
        lr, ms, _ = make_pairs(gt, true)
        row = {"gsa": score(gt, gsa(lr, ms, true).clamp(0, 1), lr, ms, true)}
        model.deg = base
        row["fixed"] = score(gt, run(model, lr, ms), lr, ms, true)
        if swappable and name != "nominal":
            model.deg = true
            row["swap"] = score(gt, run(model, lr, ms), lr, ms, true)
            model.deg = base
        res[name] = row
        print(name, {k: (v["PSNR"], v["SAM"], v["cons_H_dB"]) for k, v in row.items()}, flush=True)

    g = torch.Generator().manual_seed(0)
    lr0, ms0, _ = make_pairs(gt, base)
    for snr in (40, 35, 30):
        lr, ms = add_noise(lr0, snr, g), add_noise(ms0, snr, g)
        res[f"noise_{snr}dB"] = {"gsa": score(gt, gsa(lr, ms, base).clamp(0, 1), lr, ms, base),
                                 "fixed": score(gt, run(model, lr, ms), lr, ms, base)}
        print(f"noise {snr}dB", {k: v["PSNR"] for k, v in res[f"noise_{snr}dB"].items()}, flush=True)

    with open(os.path.join(a.out, "gaps.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
