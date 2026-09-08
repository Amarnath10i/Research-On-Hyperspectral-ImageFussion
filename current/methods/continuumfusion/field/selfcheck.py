"""P3 scaffold self-checks: the continuous field F(x,y,lambda), sensor
operators O_s, and the hold-out-sensor zero-shot test.

All checks are non-neural: the field is a parametric function, sensors are
linear operators, and "fitting" is one least-squares solve.  The headline
quantity is delta_sensor = E_unseen - E_seen.
"""

from __future__ import annotations

import torch

from .field import SceneField
from .metrics import delta_sensor, relative_error
from .sensors import Sensor, fit_field

LAM = torch.linspace(0.0, 1.0, 256)
HW = 32


def _make_sensors():
    A = Sensor("A", scale=2, msi_bands=8, srf_width=0.06)
    B = Sensor("B", scale=4, msi_bands=3, srf_width=0.10)
    C = Sensor("C", scale=2, msi_bands=6, srf_width=0.025,
               srf_lo=0.05, srf_hi=0.95)   # hold-out, different response
    return A, B, C


def _fit_to(family_bands, family_modes, A, B, yA, yB):
    """Fit a field of the given family size to given observations yA, yB."""
    f = SceneField(family_bands, family_modes)
    z, _ = fit_field([yA, yB], [A.linear_map(f, LAM, HW),
                                B.linear_map(f, LAM, HW)])
    f.Z.data.copy_(z.reshape(family_bands, family_modes, family_modes))
    return f


@torch.no_grad()
def check_linear_maps(field, A, B, C):
    """observe(F(Z)) == A_s @ z for every sensor (exact linearity)."""
    z = field.Z.detach().reshape(-1)
    errs = []
    for s in (A, B, C):
        A_s = s.linear_map(field, LAM, HW)
        y = s.observe(field, LAM, HW).reshape(-1)
        errs.append(relative_error(A_s @ z, y))
    ok = max(errs) < 1e-4
    print(f"[1] sensor operators are linear  max rel err = {max(errs):.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_zero_shot(field, A, B, C):
    """Fit on A,B -> reproduce A,B (seen), render for C zero-shot (unseen)."""
    yA, yB, yC = (s.observe(field, LAM, HW) for s in (A, B, C))
    z, _ = fit_field([yA, yB], [A.linear_map(field, LAM, HW),
                                B.linear_map(field, LAM, HW)])
    f_fit = SceneField(field.bands, field.modes)
    f_fit.Z.data.copy_(z.reshape(field.bands, field.modes, field.modes))
    e_seen = max(relative_error(A.observe(f_fit, LAM, HW), yA),
                 relative_error(B.observe(f_fit, LAM, HW), yB))
    e_unseen = relative_error(C.observe(f_fit, LAM, HW), yC)
    ok = e_seen < 1e-3 and e_unseen < 1e-3
    print(f"[2] joint fit + zero-shot   E_seen = {e_seen:.2e}, "
          f"E_unseen = {e_unseen:.2e}, "
          f"delta_sensor = {delta_sensor(e_unseen, e_seen):+.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok, e_seen, e_unseen


@torch.no_grad()
def check_baseline(field, A, C):
    """A sensor-independent field must beat a nearest-band copy of A."""
    yA = A.observe(field, LAM, HW)
    yC = C.observe(field, LAM, HW)
    ca = A.centers()
    cc = C.centers()
    base = torch.stack([yA[(cc[i] - ca).abs().argmin()]
                        for i in range(C.msi_bands)])
    e_base = relative_error(base.reshape(-1), yC.reshape(-1))
    ok = e_base > 1e-2
    print(f"[3] nearest-band baseline   E(C | A only) = {e_base:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok, e_base


@torch.no_grad()
def check_capacity(A, B, C):
    """Under-specified field family -> the zero-shot gap grows.

    Both families are fit to the SAME observations (taken from the full field);
    the under-specified family can only best-approximate them, so its
    hold-out-sensor error must be worse than the full family's.
    """
    f_true = SceneField(6, 4)
    yA, yB, yC = (s.observe(f_true, LAM, HW) for s in (A, B, C))
    f_full = _fit_to(6, 4, A, B, yA, yB)
    f_under = _fit_to(4, 2, A, B, yA, yB)
    e_seen_f = max(relative_error(A.observe(f_full, LAM, HW), yA),
                   relative_error(B.observe(f_full, LAM, HW), yB))
    e_seen_u = max(relative_error(A.observe(f_under, LAM, HW), yA),
                   relative_error(B.observe(f_under, LAM, HW), yB))
    e_unseen_f = relative_error(C.observe(f_full, LAM, HW), yC)
    e_unseen_u = relative_error(C.observe(f_under, LAM, HW), yC)
    d_f = delta_sensor(e_unseen_f, e_seen_f)
    d_u = delta_sensor(e_unseen_u, e_seen_u)
    ok = d_u > d_f + 1e-3 and e_unseen_u > e_unseen_f + 1e-3
    print(f"[4] capacity  full: E_unseen={e_unseen_f:.2e} d={d_f:+.2e}; "
          f"under: E_unseen={e_unseen_u:.2e} d={d_u:+.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


def run_all(verbose: bool = True) -> bool:
    field = SceneField(6, 4)
    A, B, C = _make_sensors()
    ok = True
    ok &= check_linear_maps(field, A, B, C)
    ok2, _, _ = check_zero_shot(field, A, B, C)
    ok &= ok2
    ok3, _ = check_baseline(field, A, C)
    ok &= ok3
    ok &= check_capacity(A, B, C)
    print(f"\n[field] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()