"""Parameters, FLOPs and GPU inference time of every method, for each dataset's band counts and scale.

    python complexity.py --out complexity.json

FLOPs are counted with torch.utils.flop_counter (multiply-adds x 2) on one 256x256 HR-MSI and its LR-HSI;
time is the mean of 10 forward passes after 3 warm-up passes (batch 1, CUDA synchronised). PUFormer runs
its priors in fp16 as in all experiments; the published networks run in fp32 as released.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from hsifuse.config import blur_of
from hsifuse.models import build
from hsifuse.ops import Degradation, dataset_srf

SETTINGS = [("chikusei", 4), ("pavia", 4), ("cave", 4), ("pavia", 8), ("pavia", 16)]
MODELS = ["puformer", "ssrnet", "psrt", "dct", "mimformer"]


@torch.no_grad()
def measure(model, lr, ms, reps=10):
    for _ in range(3):
        model(lr, ms)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        model(lr, ms)
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out", default="complexity.json")
    a = p.parse_args()
    dev = "cuda"
    res = []
    for ds, s in SETTINGS:
        k, sig = blur_of(s)
        deg = Degradation(dataset_srf(ds), scale=s, ksize=k, sigma=sig, pad="reflect").to(dev)
        b, m = deg.srf.shape
        lr, ms = torch.rand(1, b, a.size // s, a.size // s, device=dev), torch.rand(1, m, a.size, a.size, device=dev)
        for name in MODELS:
            try:
                net = build(name, deg).to(dev).eval()
                if name == "puformer":
                    net.amp = True
                params = sum(q.numel() for q in net.parameters()) / 1e6
                with torch.no_grad(), FlopCounterMode(display=False) as fc:
                    net(lr, ms)
                row = dict(dataset=ds, scale=s, model=name, bands=b, params_M=round(params, 3),
                           gflops=round(fc.get_total_flops() / 1e9, 1), seconds=round(measure(net, lr, ms), 4),
                           gpu_GB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
            except Exception as e:  # noqa: BLE001 - report and continue
                row = dict(dataset=ds, scale=s, model=name, error=repr(e)[:300])
            print(row, flush=True)
            res.append(row)
            torch.cuda.empty_cache(), torch.cuda.reset_peak_memory_stats()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
