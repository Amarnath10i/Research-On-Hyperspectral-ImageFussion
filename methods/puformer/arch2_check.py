"""Checks of the null-space transfer prior and the confidence-weighted data step.

    python arch2_check.py

1. the confidence step keeps a correct estimate fixed: for x = X*, it returns (D X*, R X*) whatever the confidence;
2. corrupt() changes only the observations it should (MSI patches, LR-HSI columns) and is reproducible;
3. PUFormer with --ntp and --cwd (and the rest of the redesigned stage) runs forward and backward; the new branches
   start at zero / at confidence ~0.98.
"""
import sys

import numpy as np
import torch

from hsifuse.models import PUFormer
from hsifuse.ops import PAVIA_WL, Degradation, corrupt, sensor_srf

torch.manual_seed(0)
ok = True


def check(name, cond, val):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {val}", flush=True)


srf = sensor_srf("ikonos", PAVIA_WL)
d = Degradation(srf, scale=4, ksize=7, sigma=2.0, pad="reflect")
x = torch.rand(2, srf.shape[0], 64, 64)
yh, ym = d.D(x), d.R(x)
m = PUFormer(d, bands=srf.shape[0], msi=srf.shape[1], width=16, depth=(1, 1, 1), heads=(1, 1, 1), omc=True,
             dstep="kronlearn", nsp=True, mres=True, sa=True, ntp=True, cwd=True)
with torch.no_grad():
    for p in m.conf_h[0][-1].parameters():
        p.normal_()                                  # any confidence
    yh2, ym2 = m.confident(0, x, yh, ym)
e = max(((yh2 - yh).abs().max() / yh.abs().max()).item(), ((ym2 - ym).abs().max() / ym.abs().max()).item())
check("confidence step keeps a correct estimate fixed", e < 1e-5, f"{e:.1e}")

lr2, ms2 = corrupt(yh, ym, np.random.default_rng(5), changes=True, stripes=True)
lr3, ms3 = corrupt(yh, ym, np.random.default_rng(5), changes=True, stripes=True)
frac = (ms2 != ym).float().mean().item()
cols = int(((lr2 - yh).abs().sum((0, 1, 2)) > 0).sum())
check("corrupt: MSI patches and LR-HSI columns, reproducible",
      0.0 < frac < 0.5 and 1 <= cols <= 3 and torch.equal(ms2, ms3) and torch.equal(lr2, lr3),
      f"{100 * frac:.1f}% of MSI values changed, {cols} LR columns")

m = PUFormer(d, bands=srf.shape[0], msi=srf.shape[1], width=16, depth=(1, 1, 1), heads=(1, 1, 1), omc=True,
             dstep="kronlearn", nsp=True, mres=True, sa=True, ntp=True, cwd=True)
out = m(lr2, ms2, return_all=True)
sum(o.abs().mean() for o in out).backward()
check("ntp + cwd stage runs", out[-1].shape == x.shape and torch.isfinite(out[-1]).all(),
      f"{sum(p.numel() for p in m.parameters()) / 1e6:.3f} M params, transfer branch starts at 0: "
      f"{m.priors[0].xfer[-1].weight.abs().max().item() == 0}, injection starts at 0: "
      f"{m.priors[0].inj.weight.abs().max().item() == 0}")
sys.exit(0 if ok else 1)
