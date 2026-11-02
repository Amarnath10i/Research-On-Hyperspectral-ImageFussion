"""Checks of the eigen-operator prior and the change maps of the confidence-weighted data step.

    python arch3_check.py

1. PUFormer with --prior eon runs forward and backward; its prior is much smaller than the U-Net;
2. the same weights run on other image sizes and another scale factor (the filters depend on eigenvalues only);
3. the change map 1 - w of the confidence step is scored against the corruption masks (cond_eval.change_auroc).
"""
import sys

import numpy as np
import torch

import cond_eval
from hsifuse.models import PUFormer
from hsifuse.ops import PAVIA_WL, Degradation, corrupt, sensor_srf

torch.manual_seed(0)
ok = True


def check(name, cond, val):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {val}", flush=True)


srf = sensor_srf("ikonos", PAVIA_WL)
B, M = srf.shape
d4 = Degradation(srf, scale=4, ksize=7, sigma=2.0, pad="reflect")
kw = dict(bands=B, msi=M, omc=True, dstep="kronlearn", nsp=True, mres=True, sa=True)
eon = PUFormer(d4, prior="eon", **kw)
unet = PUFormer(d4, **kw)
pe = sum(p.numel() for p in eon.priors.parameters()) / 1e6
pu = sum(p.numel() for p in unet.priors.parameters()) / 1e6
x = torch.rand(2, B, 64, 64)
out = eon(d4.D(x), d4.R(x), return_all=True)
sum(o.abs().mean() for o in out).backward()
check("eon stage runs", out[-1].shape == x.shape and torch.isfinite(out[-1]).all(),
      f"priors {pe:.2f} M params (U-Net priors {pu:.2f} M)")
with torch.no_grad():
    x2 = torch.rand(1, B, 96, 128)
    y2 = eon(d4.D(x2), d4.R(x2))
    d8 = Degradation(srf, scale=8, ksize=15, sigma=4.0, pad="reflect")
    eon.deg = d8
    y3 = eon(d8.D(x), d8.R(x))
check("same weights on 96x128 and at x8", y2.shape == x2.shape and y3.shape == x.shape
      and torch.isfinite(y2).all() and torch.isfinite(y3).all(), f"{tuple(y2.shape)}, {tuple(y3.shape)}")

m = PUFormer(d4, cwd=True, width=16, depth=(1, 1, 1), heads=(1, 1, 1), **kw)
m.eval()
lr, ms = d4.D(x), d4.R(x)
lr, ms, masks = corrupt(lr, ms, np.random.default_rng(3), changes=True, stripes=True, return_masks=True)
auc = cond_eval.change_auroc(m, lr, ms, [d4, d4], masks)
check("change map scored against the corruption masks", set(auc) == {"hsi", "msi"}, auc)
sys.exit(0 if ok else 1)
