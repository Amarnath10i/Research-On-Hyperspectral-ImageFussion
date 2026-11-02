"""Observability-aware uncertainty: does the predicted null-space error match the real one?

    python unc_eval.py --dataset chikusei --mat ... --ckpt best_ema.pt --model puformer --unc 1 [model flags] --out DIR

Per test image: the error e = X_hat - X, its part e_N in the fusion null space (Theorem 1c) and the predicted
sigma = exp(s / 2) of the uncertainty head. Reports
    spearman   rank correlation between the band-averaged sigma and |e_N| over pixels
    ause       area under the sparsification error curve: RMSE of the pixels kept when the most uncertain are removed
               first, minus the same with the oracle order (by the true error), averaged over 0-90 % removed,
               relative to the RMSE of all pixels (0 = perfect ranking)
    coverage90 share of (pixel, band) entries with |e_N| <= 1.645 sigma (0.90 for a calibrated Gaussian)
    null_share share of the squared error in the null space
Writes DIR/unc_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from hsifuse.config import DATASETS, MODELS, add_ablation_args, build_model, make_deg, set_metric_scale
from hsifuse.data import load_dataset, make_pairs
from hsifuse.ops import null_gate


def sparsification(u, err, steps=np.arange(0.0, 0.91, 0.05)):
    """RMSE of the kept pixels after removing the fraction f with the largest u, for each f."""
    order = np.argsort(-u)
    e2 = err ** 2
    out = []
    for f in steps:
        keep = order[int(f * len(order)):]
        out.append(float(np.sqrt(e2[keep].mean())))
    return np.array(out)


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
    _, te, _, peak = load_dataset(a.dataset, a.mat, a.cache or None)
    set_metric_scale(a, peak)
    gt = torch.from_numpy(np.asarray(te)).to(dev)
    base = make_deg(a).to(dev)
    model = build_model(a, base).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    if hasattr(model, "amp"):
        model.amp = dev == "cuda"
    lr, ms, _ = make_pairs(gt, base)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(gt.shape[0]):
        with torch.no_grad():
            xh = model(lr[i:i + 1], ms[i:i + 1]).clamp(0, 1)
            s = model.last_logvar
            e = xh - gt[i:i + 1]
            en = null_gate(base, e, 0.0)
        sig = torch.exp(s / 2)
        u = sig.mean(1)[0].flatten().cpu().numpy()
        an = en.abs().mean(1)[0].flatten().cpu().numpy()
        etot = e.pow(2).mean(1)[0].sqrt().flatten().cpu().numpy()
        idx = rng.choice(len(u), size=min(50000, len(u)), replace=False)
        rho = float(spearmanr(u[idx], an[idx]).correlation)
        curve, oracle = sparsification(u, etot), sparsification(etot, etot)
        rows.append(dict(spearman=rho, ause=float((curve - oracle).mean() / curve[0]),
                         coverage90=float((en.abs() <= 1.645 * sig).float().mean()),
                         null_share=float(en.pow(2).sum() / e.pow(2).sum()),
                         curve=curve.round(6).tolist(), oracle=oracle.round(6).tolist()))
        print(i, {k: round(v, 4) for k, v in rows[-1].items() if not isinstance(v, list)}, flush=True)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("spearman", "ause", "coverage90", "null_share")}
    with open(os.path.join(a.out, "unc_eval.json"), "w") as fh:
        json.dump(dict(dataset=a.dataset, ckpt=a.ckpt, mean=mean, per_image=rows), fh, indent=1)
    print("mean", mean, flush=True)


if __name__ == "__main__":
    main()
