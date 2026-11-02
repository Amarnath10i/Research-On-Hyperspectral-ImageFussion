"""End-to-end cost of ExactFuse on one GPU: operator estimation, eigendecompositions, the network, the data-step
alternatives (gradient, conjugate gradient), and the scaling with image size and number of bands.

    python pipeline_cost.py --out DIR [--ckpt (ignored)]

Times are means over repeated calls after warm-up (CUDA synchronised, batch 1, random inputs: the cost does not
depend on the content); memory is the peak allocated by the call. The networks are untrained (same cost).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from cond_eval import estimate, estimate_ext, estimate_np
from hsifuse.config import blur_of
from hsifuse.models import build
from hsifuse.ops import Degradation, KronFactors, dataset_srf, random_srf

DEV = "cuda"


def timed(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def inputs(deg, size):
    b, m = deg.srf.shape
    s = deg.scale
    return torch.rand(1, b, size // s, size // s, device=DEV), torch.rand(1, m, size, size, device=DEV)


def network(name, deg, **kw):
    if name == "mimformer":
        return build("mimformer", deg).to(DEV).eval()
    net = build("puformer", deg, mres=True, sa=True, **kw).to(DEV).eval()
    net.amp = True
    return net


@torch.no_grad()
def cost(net, lr, ms, reps=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        sec = timed(lambda: net(lr, ms), reps=reps)
        return dict(seconds=round(sec, 4), gpu_GB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(error="out of memory (16 GB)")


def eig_seconds(deg, size, reps=3):
    t = time.time()
    for _ in range(reps):
        KronFactors(deg, size, size, DEV)
    torch.cuda.synchronize()
    return round((time.time() - t) / reps, 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a, _ = p.parse_known_args()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "pipeline_cost.json")
    res = dict(gpu=torch.cuda.get_device_name(0), pipeline={}, data_step={}, size={}, bands={})

    def save():
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)

    k, s0 = blur_of(4)
    deg = Degradation(dataset_srf("chikusei"), scale=4, ksize=k, sigma=s0, pad="reflect").to(DEV)
    lr, ms = inputs(deg, 256)
    # 1) the full pipeline on a 256 x 256 Chikusei pair: estimation, eigendecompositions, network
    ea = SimpleNamespace(dataset="chikusei", scale=4, pad="reflect")
    sig_grid = np.round(np.arange(0.25 * s0, 2.5 * s0 + 1e-9, 0.025 * s0), 4)
    shift_grid = np.arange(-16.0, 16.0 + 1e-9, 1.0)
    pl = res["pipeline"]
    pl["estimate_isotropic_s"] = round(timed(lambda: estimate(lr, ms, ea, sig_grid, shift_grid, DEV), reps=2, warm=1), 3)
    sig, sh = estimate(lr, ms, ea, sig_grid, shift_grid, DEV)
    pl["estimate_wide_s"] = round(timed(lambda: estimate_ext(lr, ms, ea, sig_grid, shift_grid, sig, sh, DEV),
                                        reps=1, warm=0), 3)
    pl["estimate_nonparametric_s"] = round(timed(lambda: estimate_np(lr, ms, ea, sh, DEV), reps=3, warm=1), 4)
    pl["eigendecompositions_s"] = eig_seconds(deg, 256)
    pl["network"] = cost(network("exactfuse", deg, dstep="kronlearn", omc=True), lr, ms)
    print("pipeline", pl, flush=True)
    save()
    # 2) data-step alternatives in the same network (accuracy of the gradient step: ablation table)
    for name, kw in (("closed form, learned gains", dict(dstep="kronlearn")), ("closed form", dict(dstep="kron")),
                     ("gradient step", dict(dstep="grad")), ("conjugate gradient, 5 iterations", dict(dstep="cg", cg_iters=5)),
                     ("conjugate gradient, 10 iterations", dict(dstep="cg", cg_iters=10))):
        try:
            res["data_step"][name] = cost(network("exactfuse", deg, **kw), lr, ms)
        except Exception as err:                                   # noqa: BLE001 - report and continue
            res["data_step"][name] = dict(error=repr(err)[:200])
        print(name, res["data_step"][name], flush=True)
        save()
    # 3) image size (Chikusei bands): ExactFuse with its eigendecompositions, and MIMFormer
    for size in (128, 256, 512, 1024):
        lr, ms = inputs(deg, size)
        row = dict(eigendecompositions_s=eig_seconds(deg, size, reps=1))
        for name in ("exactfuse", "mimformer"):
            try:
                net = network(name, deg, dstep="kronlearn") if name == "exactfuse" else network(name, deg)
                row[name] = cost(net, lr, ms, reps=3)
                del net
            except Exception as err:                               # noqa: BLE001
                row[name] = dict(error=repr(err)[:200])
        res["size"][str(size)] = row
        print(size, row, flush=True)
        save()
    # 4) number of bands (eight random multispectral bands, 256 x 256)
    rng = np.random.default_rng(0)
    for b in (31, 64, 128, 224):
        srf = random_srf(np.linspace(400.0, 1000.0, b), rng, m_range=(8, 8))
        degb = Degradation(srf, scale=4, ksize=k, sigma=s0, pad="reflect").to(DEV)
        lr, ms = inputs(degb, 256)
        row = dict(eigendecompositions_s=eig_seconds(degb, 256, reps=1))
        try:
            row["exactfuse"] = cost(network("exactfuse", degb, dstep="kronlearn"), lr, ms, reps=3)
        except Exception as err:                                   # noqa: BLE001
            row["exactfuse"] = dict(error=repr(err)[:200])
        res["bands"][str(b)] = row
        print(b, row, flush=True)
        save()


if __name__ == "__main__":
    main()
