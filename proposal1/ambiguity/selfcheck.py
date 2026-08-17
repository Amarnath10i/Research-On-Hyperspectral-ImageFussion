"""P1 scaffold self-checks: combined A=[D;R] projector correctness and the
observable/ambiguous decomposition, hallucination metric and uncertainty map.

All checks run on a small synthetic scene with spatially concentrated
null-space content; every claim is against the operator algebra, not a network.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import (correlation, error_split, hallucination,
                      pixelwise_amb_error, uncertainty_map)
from .operator import CombinedOperator


def _smooth_rand(bands: int, hw: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(1, bands, hw, hw, generator=g)
    ax = torch.arange(9, dtype=torch.float32) - 4
    k = torch.exp(-0.5 * (ax / 1.2) ** 2)
    k = torch.outer(k, k)
    k = k / k.sum()
    wgt = k[None, None].repeat(bands, 1, 1, 1)
    return F.conv2d(z, wgt, groups=bands, padding=4)


def _make_scene(bands=6, msi=3, hw=16, scale=2, seed=0):
    P = CombinedOperator(scale, bands, msi, cg_steps=250, ridge=1e-8)
    x0 = _smooth_rand(bands, hw, seed)
    yH, yM = P.forward(x0)
    x_obs = P.pinv(yH, yM, (hw, hw))
    # patch-localised null content
    ph, pw = hw // 4, hw // 4
    mask = torch.zeros(1, 1, hw, hw)
    mask[:, :, hw // 2 - ph // 2: hw // 2 + ph // 2,
         hw // 2 - pw // 2: hw // 2 + pw // 2] = 1.0
    v_true = _smooth_rand(bands, hw, seed + 1) * mask
    x = x_obs + P.project_null(v_true)
    return P, x, x_obs, yH, yM, mask


@torch.no_grad()
def check_adjoint(P: CombinedOperator, hw=16):
    """<A x, y> == <x, A^T y> for the combined operator."""
    torch.manual_seed(0)
    x = torch.randn(1, P.bands, hw, hw)
    yH = torch.randn(1, P.bands, hw // P.scale, hw // P.scale)
    yM = torch.randn(1, P.msi_bands, hw, hw)
    lhs = (P.D(x) * yH).sum() + (P.R(x) * yM).sum()
    rhs = (x * P.adjoint(yH, yM, (hw, hw))).sum()
    err = abs(lhs - rhs).item() / max(abs(lhs.item()), 1e-12)
    ok = err < 1e-4
    print(f"[1] combined adjoint  <Ax,y> vs <x,A^T y>: rel err {err:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_consistency(P: CombinedOperator, x, yH, yM, tol=1e-2):
    """THE claim: A(consistent(y, v)) == y for arbitrary v."""
    torch.manual_seed(1)
    v = torch.randn_like(x) * 3.0
    out = P.consistent(yH, yM, v)
    yH2, yM2 = P.forward(out)
    eH = (yH2 - yH).abs().max().item() / max(yH.abs().max().item(), 1e-12)
    eM = (yM2 - yM).abs().max().item() / max(yM.abs().max().item(), 1e-12)
    ok = max(eH, eM) < tol
    print(f"[2] joint consistency max|A(consistent)-Y|/|Y|: {max(eH, eM):.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_decomposition(P: CombinedOperator, x, x_obs, tol=1e-2):
    """x_obs + x_amb == x and A(x_amb) == 0."""
    x_amb = x - x_obs
    yH, yM = P.forward(x_amb)
    e_obs = (x_obs - P.project_range(x)).abs().max().item() / max(x.abs().max().item(), 1e-12)
    e_amb = max(yH.abs().max().item(), yM.abs().max().item()) / max(x.abs().max().item(), 1e-12)
    ok = max(e_obs, e_amb) < tol
    print(f"[3] decomposition  |X_obs - P_R X|/|X| = {e_obs:.2e}, "
          f"|A X_amb|/|X| = {e_amb:.2e} ({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_hallucination(P: CombinedOperator, x, x_obs, yH, yM):
    """H is monotone in null-space accuracy: baseline ~1, half-guess ~0.5,
    full ~0; and E_obs is IDENTICAL across all three (structural identity)."""
    v = x - x_obs                       # the true null content
    x_baseline = P.consistent(yH, yM, torch.zeros_like(v))
    x_half = P.consistent(yH, yM, 0.5 * v)
    x_full = P.consistent(yH, yM, v)

    H_base, H_half, H_full = (hallucination(x, xh, P) for xh in
                              (x_baseline, x_half, x_full))
    E_obs = [error_split(x, xh, P)[0] for xh in (x_baseline, x_half, x_full)]
    e0, e1, e2 = (eo.item() for eo in E_obs)
    spread = max(e0, e1, e2) - min(e0, e1, e2)
    ok = (H_base > 0.95 and abs(H_half - 0.5) < 0.1 and H_full < 0.05
          and spread < 1e-4)
    print(f"[4] hallucination   H(base/half/full) = {H_base:.3f}/{H_half:.3f}/{H_full:.3f}, "
          f"E_obs spread = {spread:.1e} ({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_uncertainty(P: CombinedOperator, x, x_obs, yH, yM, tol=0.9):
    """The ambiguity map predicts where a null-space-ignorant reconstruction
    fails: corr(U(x,y), per-pixel ambiguous error of A^T y) must be high."""
    x_baseline = P.consistent(yH, yM, torch.zeros_like(x_obs))
    u_px, _ = uncertainty_map(x, P)
    err_px = pixelwise_amb_error(x, x_baseline, P)
    r = correlation(u_px, err_px)
    ok = r > tol
    print(f"[5] ambiguity map    corr(U(x,y), E_amb(x,y)) = {r:.4f} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


def run_all(verbose: bool = True) -> bool:
    P, x, x_obs, yH, yM, _ = _make_scene()
    ok = True
    ok &= check_adjoint(P)
    ok &= check_consistency(P, x, yH, yM)
    ok &= check_decomposition(P, x, x_obs)
    ok &= check_hallucination(P, x, x_obs, yH, yM)
    ok &= check_uncertainty(P, x, x_obs, yH, yM)
    print(f"\n[ambiguity] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()