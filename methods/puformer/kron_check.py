"""Numerical checks of the Kronecker (separable) form of the exact operators and of the closed-form data step.

    python kron_check.py [--device cuda]

For the Chikusei (WorldView-2) and Pavia Centre (IKONOS) responses, both border rules and x4 / x8:
  1. D computed as D_rows X D_cols^T equals the strided convolution of Degradation.D
  2. <D x, y> = <x, Dt y> for the matrix adjoint, and Degradation.Dt equals it
  3. the closed-form step solves (I + a D^T D + b R^T R) z = rhs (relative residual) and matches 200 CG iterations
  4. the null-space component of a random image is annihilated by D and R, and the observed and null energies add up
Prints one line per check and exits with an error if any check fails.
"""
from __future__ import annotations

import argparse
import sys

import torch

from hsifuse.config import blur_of
from hsifuse.ops import Degradation, dataset_srf, kron_factors, kron_prox, kron_prox_learned, null_split


def cg_solve(deg, x, yh, ym, a, b, iters=200):
    op = lambda v: v + a * deg.Dt(deg.D(v)) + b * deg.Rt(deg.R(v))
    dot = lambda u, v: (u * v).sum((1, 2, 3), keepdim=True)
    z = x.clone()
    r = x + a * deg.Dt(yh) + b * deg.Rt(ym) - op(z)
    p, rs = r, dot(r, r)
    for _ in range(iters):
        q = op(p)
        step = rs / dot(p, q).clamp_min(1e-30)
        z, r = z + step * p, r - step * q
        rs_new = dot(r, r)
        p, rs = r + rs_new / rs.clamp_min(1e-30) * p, rs_new
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)
    ok = True

    def check(name, val, tol):
        nonlocal ok
        good = val < tol
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {name}: {val:.3e} (tol {tol:g})", flush=True)

    for ds in ("chikusei", "pavia"):
        for scale in (4, 8):
            for pad in ("reflect", "zeros"):
                k, s0 = blur_of(scale)
                deg = Degradation(dataset_srf(ds), scale=scale, ksize=k, sigma=1.3 * s0, pad=pad).to(dev)
                b_, m_ = deg.srf.shape
                h = w = 128 if scale == 8 else 64
                x = torch.rand(2, b_, h, w, device=dev)
                kf = kron_factors(deg, h, w, dev)
                tag = f"{ds} x{scale} {pad}"
                d_conv = deg.D(x)
                d_mat = torch.einsum("ih,nbhw,jw->nbij", kf.dr, x, kf.dc)
                check(f"{tag} D matrix = conv", ((d_mat - d_conv).norm() / d_conv.norm()).item(), 1e-5)
                y = torch.rand_like(d_conv)
                lhs, rhs = (d_mat * y).sum().item(), (x * kf.Dt(y)).sum().item()
                check(f"{tag} adjoint (matrix)", abs(lhs - rhs) / abs(lhs), 1e-5)
                check(f"{tag} Dt conv = matrix", ((deg.Dt(y) - kf.Dt(y)).norm() / kf.Dt(y).norm()).item(), 1e-5)
                yh, ym = deg.D(x), deg.R(x)
                v = torch.rand_like(x)
                aa, bb = 16.0 * 1.3, 0.7
                z = kron_prox(deg, v, yh, ym, aa, bb)
                op = z + aa * deg.Dt(deg.D(z)) + bb * deg.Rt(deg.R(z))
                target = v + aa * deg.Dt(yh) + bb * deg.Rt(ym)
                check(f"{tag} closed form residual", ((op - target).norm() / target.norm()).item(), 1e-4)
                zc = cg_solve(deg, v, yh, ym, aa, bb)
                check(f"{tag} closed form = CG", ((z - zc).norm() / zc.norm()).item(), 1e-3)
                # learned spectral gains: zero-initialized networks give the exact step; any gains keep the fixed point
                mlp = lambda: torch.nn.Sequential(torch.nn.Linear(5, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1)).to(dev)
                gs, gb = mlp(), mlp()
                for net in (gs, gb):
                    torch.nn.init.zeros_(net[-1].weight), torch.nn.init.zeros_(net[-1].bias)
                code = torch.zeros(2, 4, device=dev)
                zl = kron_prox_learned(deg, v, yh, ym, aa, bb, gs, gb, code)
                check(f"{tag} learned step at init = closed form", ((zl - z).norm() / z.norm()).item(), 1e-4)
                for net in (gs, gb):
                    torch.nn.init.normal_(net[-1].weight, std=1.0)
                fp = kron_prox_learned(deg, x, yh, ym, aa, bb, gs, gb, torch.randn(2, 4, device=dev))
                check(f"{tag} learned step keeps the fixed point", ((fp - x).norm() / x.norm()).item(), 1e-5)
                e = torch.randn_like(x)
                null = (1 - kf.obs_spectral)[None, :, None, None] * (1 - kf.obs_spatial)[None, None]
                en = kf.from_eig(kf.to_eig(e) * null)
                check(f"{tag} D(null) = 0", (deg.D(en).norm() / deg.D(e).norm()).item(), 1e-4)
                check(f"{tag} R(null) = 0", (deg.R(en).norm() / deg.R(e).norm()).item(), 1e-4)
                o, n = null_split(deg, e)
                check(f"{tag} energy split", ((o + n - e.flatten(1).pow(2).sum(1)).abs().max()
                                              / e.flatten(1).pow(2).sum(1).max()).item(), 1e-4)
                frac = n.mean().item() / (o + n).mean().item()
                print(f"     {tag}: null-space share of a random image {frac:.3f} "
                      f"(theory (1 - M/B)(1 - 1/s^2) = {(1 - m_ / b_) * (1 - 1 / scale ** 2):.3f})", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
