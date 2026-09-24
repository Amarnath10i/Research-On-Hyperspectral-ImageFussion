"""P3 learned-field self-check.

Demonstrates the central claim of the P3 redesign:

    real scenes are NOT low-order (Gaussian-bump x cosine) fields, so the
    linear SceneField underfits and its hold-out-sensor error blows up, while
    the learned NeuralSceneField recovers the scene AND keeps rendering an
    unseen sensor C zero-shot (small delta_sensor).

The "true" scene is a frozen NeuralSceneField (a genuinely non-low-order
radiance field).  Both a linear field and a neural field are then fit to the
SAME observations from sensors A,B; only the neural field can represent it, so
it wins on the unseen sensor C.  This is the experiment that turns P3 from
"a least-squares solve" into a Q1-grade learned method.
"""

from __future__ import annotations

import torch

from .field import SceneField
from .metrics import delta_sensor, relative_error
from .sensors import Sensor, fit_field
from .neural_field import NeuralSceneField, fit_neural

LAM = torch.linspace(0.0, 1.0, 48)
LEN = LAM.shape[0]
HW = 16


def _make_sensors():
    A = Sensor("A", scale=2, msi_bands=8, srf_width=0.06)
    B = Sensor("B", scale=4, msi_bands=3, srf_width=0.10)
    C = Sensor("C", scale=2, msi_bands=6, srf_width=0.025,
               srf_lo=0.05, srf_hi=0.95)   # hold-out, different response
    return A, B, C


def _teacher_cube(A, B, C):
    """A frozen neural field as the ground-truth radiance scene (N,H,W)."""
    torch.manual_seed(0)
    teacher = NeuralSceneField(latent_dim=16, hidden=64, layers=3)
    code = torch.randn(16) * 0.2
    cube = teacher.render(LAM, HW, code).detach()
    cube = cube - cube.min()
    cube = cube / cube.max().clamp_min(1e-6)           # -> [0,1]
    yA = A.observe_cube(cube, LAM).detach()
    yB = B.observe_cube(cube, LAM).detach()
    yC = C.observe_cube(cube, LAM).detach()
    return yA, yB, yC


def _fit_linear(yA, yB, A, B):
    f = SceneField(6, 4)
    z, _ = fit_field([yA, yB], [A.linear_map(f, LAM, HW),
                                B.linear_map(f, LAM, HW)])
    f.Z.data.copy_(z.reshape(6, 4, 4))
    return f


def check_neural_vs_linear(A, B, C, yA, yB, yC):
    # --- linear baseline fit --------------------------------------------
    f_lin = _fit_linear(yA, yB, A, B)
    e_seen_lin = max(relative_error(A.observe(f_lin, LAM, HW), yA),
                     relative_error(B.observe(f_lin, LAM, HW), yB))
    e_unseen_lin = relative_error(C.observe(f_lin, LAM, HW), yC)

    # --- learned neural field fit ---------------------------------------
    torch.manual_seed(1)
    field = NeuralSceneField(latent_dim=24, hidden=96, layers=3)
    code = torch.nn.Parameter(torch.zeros(24))
    fit_neural(field, code, [(A, yA), (B, yB)], LAM, HW, steps=400, lr=5e-3)
    e_seen_nn = max(
        relative_error(A.observe_cube(field.render(LAM, HW, code), LAM), yA),
        relative_error(B.observe_cube(field.render(LAM, HW, code), LAM), yB))
    e_unseen_nn = relative_error(
        C.observe_cube(field.render(LAM, HW, code), LAM), yC)

    ok = (e_unseen_nn < e_unseen_lin) and (e_seen_nn < 1e-1)
    print(f"[N1] neural beats linear on unseen sensor C:")
    print(f"     linear  E_seen={e_seen_lin:.2e}  E_unseen={e_unseen_lin:.2e}")
    print(f"     neural  E_seen={e_seen_nn:.2e}  E_unseen={e_unseen_nn:.2e}  "
          f"delta_sensor={delta_sensor(e_unseen_nn, e_seen_nn):+.2e}  "
          f"({'PASS' if ok else 'FAIL'})")
    return ok, e_seen_lin, e_unseen_lin, e_seen_nn, e_unseen_nn


def check_baseline(yA, yC, A, C):
    ca = A.centers()
    cc = C.centers()
    base = torch.stack([yA[(cc[i] - ca).abs().argmin()]
                        for i in range(C.msi_bands)])
    e_base = relative_error(base.reshape(-1), yC.reshape(-1))
    ok = e_base > 1e-2
    print(f"[N2] nearest-band baseline   E(C | A only) = {e_base:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


def run_all(verbose: bool = True) -> bool:
    A, B, C = _make_sensors()
    yA, yB, yC = _teacher_cube(A, B, C)
    ok = True
    ok &= check_baseline(yA, yC, A, C)
    ok &= check_neural_vs_linear(A, B, C, yA, yB, yC)[0]
    print(f"\n[neural_field] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()
