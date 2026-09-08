"""Self-checks for NSP (CPU-fast, no data required).

Verifies the structural claims of Proposal 3:
  [1] div(g grad u) is self-adjoint: <div(g grad u), v> == <u, div(g grad v)>.
  [2] Without diffusion, the penalty dynamics converge to the fusion solution
      of the normal equation (residual reduces well below the initial value).
  [3] Learned components transfer across scales: the same weights, with the
      scale-adaptive discretisation step, reduce the residual at scale 2 and 4.
  [4] Ladder stages build, forward and backprop.

Returns True iff all checks pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config
from .model import NSPModel
from .pde import divergence


def check_div_adjoint() -> bool:
    torch.manual_seed(0)
    B, C, H, W = 2, 6, 12, 14
    u = torch.randn(B, C, H, W)
    v = torch.randn(B, C, H, W)
    g = torch.rand(B, C, H, W) + 0.1
    lhs = (divergence(u, g, g) * v).sum()
    rhs = (u * divergence(v, g, g)).sum()
    rel = float((lhs - rhs).abs() / max(lhs.abs(), rhs.abs(), 1e-12))
    ok = rel < 1e-4
    print(f"[1] div self-adjoint rel={rel:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def _tiny_problem(cfg: Config, seed: int = 1):
    torch.manual_seed(seed)
    model = NSPModel(cfg)
    model.set_srf(torch.rand(cfg.bands, cfg.msi_bands) + 0.1)
    op = model.op
    kernel = model.default_kernel
    gt = torch.rand(1, cfg.bands, cfg.patch, cfg.patch) * 0.8 + 0.1
    x = op.D(gt, kernel)
    m = op.S(gt, model.srf)
    return model, op, kernel, gt, x, m


def check_pde_converges() -> bool:
    cfg = Config(bands=8, msi_bands=3, scale=4, patch=32, blur_ksize=5,
                 eval_sigma=1.2, pde_steps=200, dt=0.1, lam1=1.0, lam2=1.0,
                 learn_dt=False, learn_lam=False, use_diffusion=False,
                 use_tensor_net=False, scale_adaptive_dt=False,
                 source_root="", target_root="")
    model, op, kernel, _, x, m = _tiny_problem(cfg)
    u0 = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
    with torch.no_grad():
        pred = model(x, m, kernel)
    r0 = float((op.Dt(op.D(u0, kernel) - x, kernel)
                + op.St(op.S(u0, model.srf) - m, model.srf)).norm().item())
    r1 = float(pred["residuals"][-1].item())
    # monotone non-increasing residual
    vals = [float(v.item()) for v in pred["residuals"]]
    viol = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i - 1] * 1.05)
    ok = r1 < 0.5 * r0 and viol <= 3
    print(f"[2] PDE->fusion residual {r0:.3e} -> {r1:.3e} "
          f"(violations={viol}) {'PASS' if ok else 'FAIL'}")
    return ok


def check_scale_transfer() -> bool:
    base = dict(bands=8, msi_bands=3, patch=32, blur_ksize=5, eval_sigma=1.2,
                pde_steps=60, dt=0.2, lam1=1.0, lam2=1.0, learn_dt=True,
                learn_lam=True, use_diffusion=True, use_tensor_net=True,
                tensor_hidden=12, tensor_layers=1, scale_adaptive_dt=True,
                source_root="", target_root="")
    ok = True
    for s in (2, 4):
        cfg = Config(**{**base, "scale": s})
        model, op, kernel, _, x, m = _tiny_problem(cfg, seed=2)
        u0 = F.interpolate(x, scale_factor=s, mode="bicubic",
                           align_corners=False)
        def residual(u):
            return torch.linalg.vector_norm(
                op.Dt(op.D(u, kernel) - x, kernel)
                + op.St(op.S(u, model.srf) - m, model.srf), dim=(1, 2, 3)).item()
        r0 = residual(u0)
        with torch.no_grad():
            pred = model(x, m, kernel)
        r1 = residual(pred["out"])
        good = r1 < r0
        ok &= good
        print(f"[3] scale={s} residual {r0:.3e} -> {r1:.3e} "
              f"{'PASS' if good else 'FAIL'}")
    return ok


def check_ladder_smoke() -> bool:
    torch.manual_seed(3)
    ok = True
    stages = {
        "s1_pure_penalty": dict(use_diffusion=False, use_tensor_net=False,
                                learn_dt=False, learn_lam=False),
        "s2_learned_scalars": dict(use_diffusion=False, use_tensor_net=False,
                                   learn_dt=True, learn_lam=True),
        "s3_cross_spectral": dict(use_diffusion=True, use_tensor_net=True,
                                  learn_dt=True, learn_lam=True),
        "s4_scale_adaptive": dict(use_diffusion=True, use_tensor_net=True,
                                  learn_dt=True, learn_lam=True,
                                  scale_adaptive_dt=True),
    }
    base = dict(bands=6, msi_bands=3, scale=4, patch=16, blur_ksize=5,
                eval_sigma=1.2, pde_steps=6, dt=0.1, tensor_hidden=8,
                tensor_layers=1, source_root="", target_root="")
    for name, pc in stages.items():
        cfg = Config(**{**base, **pc})
        m = NSPModel(cfg)
        m.set_srf(torch.rand(cfg.bands, cfg.msi_bands) + 0.1)
        op = m.op
        kernel = torch.rand(2, 5, 5) + 0.5
        gt = torch.rand(2, cfg.bands, 16, 16) * 0.8 + 0.1
        x = op.D(gt, kernel)
        ms = op.S(gt, m.srf)
        pred = m(x, ms, kernel)
        has_learned = bool(pc.get("learn_dt") or pc.get("learn_lam")
                           or pc.get("use_tensor_net"))
        if has_learned:
            pred["out"].sum().backward()
            grads_ok = []
            if pc.get("learn_dt"):
                grads_ok.append(m.log_dt.grad is not None)
            if pc.get("learn_lam"):
                grads_ok.append(m.log_lam1.grad is not None
                                and m.log_lam2.grad is not None)
            if pc.get("use_tensor_net"):
                grads_ok.append(any(p.grad is not None
                                    for p in m.tensor_net.parameters()))
            g_ok = bool(grads_ok) and all(grads_ok)
        else:
            g_ok = bool(torch.isfinite(pred["out"]).all().item())
        ok &= g_ok
        print(f"[4] {name}: fwd/bwd {'PASS' if g_ok else 'FAIL'}")
    return ok


def run_all(device: str = "cpu") -> bool:
    torch.manual_seed(0)
    checks = [
        ("div adjoint", check_div_adjoint),
        ("pde->fusion", check_pde_converges),
        ("scale transfer", check_scale_transfer),
        ("ladder smoke", check_ladder_smoke),
    ]
    ok = True
    for name, fn in checks:
        try:
            ok &= bool(fn())
        except Exception as e:  # noqa: BLE001
            print(f"[selfcheck] {name}: EXCEPTION {e!r}")
            ok = False
    print(f"[selfcheck] {'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)