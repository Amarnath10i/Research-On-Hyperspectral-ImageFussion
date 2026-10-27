"""Checks of the wider operator family (axis-aligned anisotropic blur, sub-pixel offsets of the LR-HSI grid).

    python reg_check.py

1. adjoint test <D x, y> = <x, Dt y> with offsets and anisotropy (reflect border);
2. the Kronecker factors reproduce D exactly (D x = Dr x Dc^T per band);
3. the closed-form data step solves (I + a Dt D + b Rt R) z = rhs (residual of the normal equations);
4. the isotropic, centered operator is unchanged (same kernel as before);
5. the estimator of cond_eval recovers a known (sigma_r, sigma_c, off_r, off_c, shift) from a simulated pair.
"""
import argparse
import sys

import numpy as np
import torch

from hsifuse.ops import Degradation, dataset_srf, gaussian_kernel, kron_factors, kron_prox
import cond_eval

torch.manual_seed(0)
ok = True


def check(name, cond, val):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name}: {val}", flush=True)


srf = dataset_srf("chikusei")
B, M = srf.shape
d = Degradation(srf, scale=4, ksize=7, sigma=2.5, pad="reflect", sigma_c=1.25, off=(0.5, -0.75)).double()
x = torch.rand(2, B, 64, 64, dtype=torch.float64)
y = torch.rand(2, B, 16, 16, dtype=torch.float64)
lhs, rhs = (d.D(x) * y).sum(), (x * d.Dt(y)).sum()
check("adjoint", abs(lhs - rhs) / abs(lhs) < 1e-12, f"{abs(lhs - rhs) / abs(lhs):.2e}")

kf = kron_factors(d, 64, 64, "cpu")
dx = torch.einsum("ih,nbhw,jw->nbij", kf.dr.double(), x, kf.dc.double())
err = (dx - d.D(x)).abs().max().item()
check("Kronecker factors reproduce D", err < 1e-5, f"{err:.2e}")

d32 = Degradation(srf, scale=4, ksize=7, sigma=2.5, pad="reflect", sigma_c=1.25, off=(0.5, -0.75))
x32, yh, ym = torch.rand(1, B, 64, 64), torch.rand(1, B, 16, 16), torch.rand(1, M, 64, 64)
a, b = 0.7, 0.3
z = kron_prox(d32, x32, yh, ym, a, b)
res = z + a * d32.Dt(d32.D(z)) + b * d32.Rt(d32.R(z)) - (x32 + a * d32.Dt(yh) + b * d32.Rt(ym))
rel = (res.norm() / (x32 + a * d32.Dt(yh)).norm()).item()
check("closed-form step solves the normal equations", rel < 1e-4, f"{rel:.2e}")

k_old = np.exp(-((np.arange(7) - 3.0)[:, None] ** 2 + (np.arange(7) - 3.0)[None, :] ** 2) / (2 * 2.0 ** 2))
k_old /= k_old.sum()
check("isotropic kernel unchanged", np.abs(gaussian_kernel(7, 2.0) - k_old).max() < 1e-7,
      f"{np.abs(gaussian_kernel(7, 2.0) - k_old).max():.1e}")

ap = argparse.Namespace(dataset="chikusei", scale=4, pad="reflect")
gt = torch.rand(2, B, 128, 128)
gt = torch.nn.functional.avg_pool2d(gt, 5, 1, 2)          # smooth texture, like real images
true = cond_eval.deg_at(ap, 1.5, 4.0, sigma_c=3.0, off=(0.5, -0.25))
lr, ms = true.simulate(gt)
s0 = 2.0
sig_grid = np.round(np.arange(0.25 * s0, 2.5 * s0 + 1e-9, 0.025 * s0), 4)
shift_grid = np.arange(-16.0, 16.0 + 1e-9, 1.0)
sig, sh = cond_eval.estimate(lr, ms, ap, sig_grid, shift_grid, "cpu")
ext = cond_eval.estimate_ext(lr, ms, ap, sig_grid, shift_grid, sig, sh, "cpu")
for t in ext:
    good = abs(t["s"] - 1.5) <= 0.05 and abs(t["c"] - 3.0) <= 0.05 and abs(t["r"] - 0.5) <= 0.125 \
        and abs(t["q"] + 0.25) <= 0.125 and abs(t["d"] - 4.0) <= 1
    check("wide estimate recovers the operator", good, t)
sys.exit(0 if ok else 1)
