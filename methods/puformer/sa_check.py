"""Checks of the sensor-independent operators.

    python sa_check.py

1. every sensor's response on Chikusei and Pavia Centre: band count, unit-sum columns, no NaN;
2. random responses: band counts within range, unit-sum columns;
3. the back-projection R^+ reproduces the HR-MSI: R(R^+ Y_M) = Y_M.
"""
import sys

import numpy as np
import torch

from hsifuse.models import msi_backproject
from hsifuse.ops import CHIKUSEI_WL, PAVIA_WL, SENSORS, Degradation, random_srf, sensor_srf

ok = True


def check(name, cond, val):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {val}", flush=True)


for wl, ds in ((CHIKUSEI_WL, "chikusei"), (PAVIA_WL, "pavia")):
    for s in SENSORS:
        r = sensor_srf(s, wl)
        check(f"{ds} {s}", np.isfinite(r).all() and np.allclose(r.sum(0), 1, atol=1e-5) and r.shape[1] >= 3,
              f"{r.shape[1]} bands, peaks at {[int(wl[i]) for i in r.argmax(0)]}")
rng = np.random.default_rng(0)
ms = [random_srf(CHIKUSEI_WL, rng) for _ in range(200)]
check("random responses", all(3 <= m.shape[1] <= 10 and np.allclose(m.sum(0), 1, atol=1e-5) for m in ms),
      f"band counts {sorted(set(m.shape[1] for m in ms))}")
for s in ("worldview2", "rgb", "sentinel2"):
    d = Degradation(sensor_srf(s, CHIKUSEI_WL), scale=4, ksize=7, sigma=2.0, pad="reflect")
    ym = torch.rand(2, d.srf.shape[1], 32, 32)
    err = ((d.R(msi_backproject(d, ym)) - ym).abs().max() / ym.abs().max()).item()
    check(f"R(R^+ Y_M) = Y_M ({s})", err < 1e-4, f"{err:.1e}")
sys.exit(0 if ok else 1)
