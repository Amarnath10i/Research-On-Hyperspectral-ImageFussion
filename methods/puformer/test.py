"""Score a saved checkpoint on the 8 Chikusei x4 test tiles (and the 64 validation patches).

    python test.py --mat ... --ckpt out/puformer/best_ema.pt --pad reflect --out out/puformer_test
    python test.py --mat ... --ckpt first_run/best_ema.pt --pad zeros --out out/first_run   # the 57.99 dB model

Writes results.json (plain and identity+transpose self-ensemble, per image, Q2n / SCC, consistency)
and the float32 predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from hsifuse import metrics
from hsifuse.data import load_dataset, make_pairs
from hsifuse.evaluation import final_test, predict
from hsifuse.metrics import evaluate
from hsifuse.models import build
from hsifuse.ops import Degradation, dataset_srf


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
    p.add_argument("--q2n", type=int, default=1)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    _, te, va, metrics.DN_SCALE = load_dataset(a.dataset, a.mat, a.cache or None)
    deg = Degradation(dataset_srf(a.dataset), pad=a.pad).to(dev)
    test = make_pairs(torch.from_numpy(te).to(dev), deg)
    val = make_pairs(torch.from_numpy(va).to(dev), deg)
    model = build(a.model, deg, **(dict(width=a.width, stages=a.stages) if a.model == "puformer" else {})).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    if a.model == "puformer":
        model.amp = dev.type == "cuda"

    v = evaluate(val[2], predict(model, val[0], val[1], 16))
    res = final_test(model, test, deg, a.out, full=bool(a.q2n))
    bic = evaluate(test[2], F.interpolate(test[0], scale_factor=4, mode="bicubic", align_corners=False))
    out = dict(ckpt=a.ckpt, pad=a.pad, val=v, **res, bicubic_test=bic, seconds=round(time.time() - t0, 1))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("VAL:", {k: round(x, 4) for k, x in v.items()}, flush=True)
    print("TEST:", {k: round(x, 4) for k, x in res["test"].items()}, flush=True)
    print("TEST (self-ensemble):", {k: round(x, 4) for k, x in res["test_tta"].items()}, flush=True)
    print("consistency:", res["consistency"], flush=True)


if __name__ == "__main__":
    main()
