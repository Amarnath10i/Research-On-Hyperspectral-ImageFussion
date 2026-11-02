"""Checks of the null-space-gated prior and the redesigned stage.

    python arch_check.py

1. gamma = 0: the gated correction lies in the joint null space, D g = 0 and R g = 0;
2. gamma = 1: the correction passes unchanged;
3. the redesigned PUFormer (closed-form learned data step, gate, band-space interface) runs forward and backward,
   with and without observation-model conditioning, and its gates start at gamma = 0.5.
"""
import sys

import torch

from hsifuse.models import PUFormer
from hsifuse.ops import Degradation, null_gate, sensor_srf, PAVIA_WL

torch.manual_seed(0)
ok = True


def check(name, cond, val):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {val}", flush=True)


srf = sensor_srf("ikonos", PAVIA_WL)
d = Degradation(srf, scale=4, ksize=7, sigma=2.0, pad="reflect", sigma_c=2.5, off=(0.25, -0.5))
x = torch.rand(2, srf.shape[0], 64, 64)
g0 = null_gate(d, x, 0.0)
rd = (d.D(g0).norm() / d.D(x).norm()).item()
rr = (d.R(g0).norm() / d.R(x).norm()).item()
check("gamma = 0: D g = 0", rd < 1e-4, f"{rd:.1e}")
check("gamma = 0: R g = 0", rr < 1e-4, f"{rr:.1e}")
n = torch.randn_like(x)                 # zero-mean white noise: its null-space share is the dimension share
share, expect = (null_gate(d, n, 0.0).norm() / n.norm()).item() ** 2, (1 - srf.shape[1] / srf.shape[0]) * (1 - 1 / 16)
check("null-space share of white noise = dimension share", abs(share - expect) < 0.02,
      f"{share:.3f} vs (1 - M/B)(1 - 1/s^2) = {expect:.3f}")
g1 = null_gate(d, x, 1.0)
check("gamma = 1: unchanged", ((g1 - x).abs().max() / x.abs().max()).item() < 1e-4,
      f"{((g1 - x).abs().max() / x.abs().max()).item():.1e}")
for omc in (False, True):
    m = PUFormer(d, bands=srf.shape[0], msi=srf.shape[1], width=16, depth=(1, 1, 1), heads=(1, 1, 1),
                 omc=omc, dstep="kronlearn", nsp=True, mres=True, sa=True)
    yh, ym = d.D(x), d.R(x)
    out = m(yh, ym, return_all=True)
    loss = sum(o.abs().mean() for o in out)
    loss.backward()
    grads = sum(p.grad is not None for p in m.parameters())
    check(f"redesigned stage runs (omc={omc})", out[-1].shape == x.shape and torch.isfinite(out[-1]).all(),
          f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params, {grads} grads, gate "
          f"{torch.sigmoid(m.nsp_gate).tolist()}")
sys.exit(0 if ok else 1)
