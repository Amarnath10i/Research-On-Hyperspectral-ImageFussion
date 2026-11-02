"""One network for many observation models: test a checkpoint on other multispectral sensors or scale factors.

    python family_eval.py --dataset chikusei --mat ... --ckpt best_ema.pt --model puformer --sa 1 --mode sensors --out DIR
    python family_eval.py --dataset chikusei --mat ... --ckpt best_ema.pt --model puformer --mode scales --scales 4,8,16 --out DIR

sensors  test pairs with the nominal blur and the response of every sensor in hsifuse.ops.SENSORS (bands outside the
         hyperspectral range dropped). A sensor-independent model (--sa 1) is given that sensor's operator; any other
         model can only be scored on the sensor it was trained for (--sensor, default the dataset's own).
scales   test pairs at each scale factor with its blur (blur_of); PUFormer is given that operator, other models are
         scored at their training scale only.
GSA with the true operator is listed for reference. Writes DIR/family_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from hsifuse.baselines import gsa
from hsifuse.config import DATASETS, MODELS, add_ablation_args, build_model, make_deg, set_metric_scale
from hsifuse.data import load_dataset, make_pairs
from hsifuse.metrics import evaluate
from hsifuse.ops import MEASURED_SENSORS, SENSORS, dataset_wl, sensor_srf

KEEP = ("PSNR", "SAM", "ERGAS", "SSIM", "RMSE_DN")
OWN = {"chikusei": "worldview2", "pavia": "ikonos", "cave": "rgb", "harvard": "rgb"}


def score(gt, pred):
    m = evaluate(gt, pred)
    return {k: round(float(m[k]), 4) for k in KEEP if k in m}


@torch.no_grad()
def run(model, lr, ms, deg=None):
    if deg is not None:
        model.deg = deg
    return torch.cat([model(lr[i:i + 1], ms[i:i + 1]) for i in range(lr.shape[0])]).clamp(0, 1)


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
    p.add_argument("--mode", default="sensors", choices=["sensors", "scales"])
    p.add_argument("--scales", default="4,8,16")
    p.add_argument("--scale_free", type=int, default=0, help="1: the model was trained over scales (--rand_scale)")
    add_ablation_args(p)
    p.add_argument("--measured", type=int, default=0, help="1: the six sensors with measured responses "
                   "(hsifuse.srf_measured) instead of the band-edge models")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    _, te, _, peak = load_dataset(a.dataset, a.mat, a.cache or None)
    set_metric_scale(a, peak)
    gt = torch.from_numpy(np.asarray(te)).to(dev)
    base = make_deg(a).to(dev)
    model = build_model(a, base).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    if hasattr(model, "amp"):
        model.amp = dev == "cuda"
    explicit = a.model == "puformer" or bool(a.sa)       # models that take an operator
    res = dict(dataset=a.dataset, model=a.model, sa=a.sa, sensor=a.sensor or OWN.get(a.dataset), mode=a.mode,
               ckpt=a.ckpt, results={})
    if a.mode == "sensors":
        own = a.sensor or OWN.get(a.dataset)
        for name in (MEASURED_SENSORS if a.measured else SENSORS):
            srf = sensor_srf(name, dataset_wl(a.dataset))
            deg = make_deg(a, srf=srf).to(dev)
            lr, ms, _ = make_pairs(gt, deg)
            row = dict(bands=int(srf.shape[1]), gsa=score(gt, gsa(lr, ms, deg).clamp(0, 1)))
            if a.sa or name == own:
                row["model"] = score(gt, run(model, lr, ms, deg if explicit else None))
            res["results"][name] = row
            print(name, row["bands"], "bands", {k: v["PSNR"] for k, v in row.items() if isinstance(v, dict)}, flush=True)
            with open(os.path.join(a.out, "family_eval_measured.json" if a.measured else "family_eval.json"), "w") as fh:
                json.dump(res, fh, indent=1)
    else:
        for s in (int(v) for v in a.scales.split(",")):
            if gt.shape[-1] % s or gt.shape[-2] % s:
                continue
            deg = make_deg(a, scale=s).to(dev)
            lr, ms, _ = make_pairs(gt, deg)
            row = dict(gsa=score(gt, gsa(lr, ms, deg).clamp(0, 1)))
            if (a.scale_free and a.model == "puformer") or s == a.scale:
                row["model"] = score(gt, run(model, lr, ms, deg if a.model == "puformer" else None))
            res["results"][f"x{s}"] = row
            print(f"x{s}", {k: v["PSNR"] for k, v in row.items()}, flush=True)
            with open(os.path.join(a.out, "family_eval_measured.json" if a.measured else "family_eval.json"), "w") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
