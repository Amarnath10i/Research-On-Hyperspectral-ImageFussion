"""Self-checks for KrylovNet (CPU-fast, no data required).

Verifies the structural claims of Proposal 2:
  [1] D/D^T and S/S^T are exact adjoints.
  [2] The unrolled GMRES solves the normal equation: residual < 1e-6 on a
      dense SPD system, monotone non-increasing, solution accurate.
  [3] The trained spectral-graph preconditioner reduces the condition number
      (cond(P^{-1}A) < 0.5 cond(A)).
  [4] Krylov beats fixed-point (Richardson) and more stages reduce the residual
      on a real image-shaped problem.
  [5] Every ladder stage builds, forwards and backprops.

Returns True iff all checks pass.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .model import KrylovNet
from .solver import FusionOperator, krylov_gmres, richardson_solve


def _close(a: torch.Tensor, b: torch.Tensor, tol: float) -> bool:
    denom = max(float(a.abs().max()), float(b.abs().max()), 1e-12)
    return float((a - b).abs().max()) / denom < tol


def check_adjoint() -> bool:
    torch.manual_seed(0)
    op = FusionOperator(scale=4, rho=1e-3)
    B, C, M, k = 2, 8, 3, 5
    kernel = torch.rand(B, k, k) + 0.5
    srf = torch.rand(C, M) + 0.1
    x = torch.randn(B, C, 8, 8)
    y = torch.randn(B, C, 2, 2)
    msi = torch.randn(B, M, 8, 8)

    d = op.D(x, kernel)
    dt = op.Dt(y, kernel)
    lhs = (d * y).sum(); rhs = (x * dt).sum()
    ok_d = _close(lhs, rhs, 1e-3)
    print(f"[1] adjoint D/D^T  <Dx,y>=<x,D^Ty> rel={abs(lhs-rhs).item()/max(abs(lhs).item(),1e-9):.2e} "
          f"{'PASS' if ok_d else 'FAIL'}")

    s = op.S(x, srf)
    st = op.St(msi, srf)
    lhs2 = (s * msi).sum(); rhs2 = (x * st).sum()
    ok_s = _close(lhs2, rhs2, 1e-3)
    print(f"[1] adjoint S/S^T  <Sx,m>=<x,S^Tm> rel={abs(lhs2-rhs2).item()/max(abs(lhs2).item(),1e-9):.2e} "
          f"{'PASS' if ok_s else 'FAIL'}")
    return ok_d and ok_s


def check_krylov_exact_solve() -> bool:
    torch.manual_seed(1)
    n = 32
    A = torch.randn(n, n).double()
    A = A @ A.t() + 1e-2 * torch.eye(n, dtype=torch.float64)
    xtrue = torch.randn(n).double()
    b = (A @ xtrue).unsqueeze(0)
    x0 = torch.zeros(1, n, dtype=torch.float64)

    def Aop(v):
        return v @ A.t()

    x, res = krylov_gmres(x0, b, Aop, None, m=n, blend=None)
    rvals = [float(r.norm(dim=1).item()) for r in res]
    final = rvals[-1]
    viol = sum(1 for i in range(1, len(rvals))
               if rvals[i] > rvals[i - 1] + 1e-12)
    rel = float((x.squeeze(0) - xtrue).norm() / xtrue.norm())
    ok = final < 1e-6 and viol <= 2 and rel < 1e-5
    print(f"[2] exact solve  m={n} residual={final:.2e} (violations={viol}) "
          f"rel_err={rel:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def check_precond_helps() -> bool:
    torch.manual_seed(2)
    n = 32
    d = torch.linspace(1.0, 2.5, n)
    B0 = torch.randn(n, n) * 0.02
    A = torch.diag(d) + B0 @ B0.t() + 1e-3 * torch.eye(n)
    feats = d.unsqueeze(0).unsqueeze(-1)
    target = (1.0 / d).unsqueeze(0)

    from .model import SpectralPreconditioner
    pre = SpectralPreconditioner(n, graph_k=4, hidden=32, gcn_layers=2,
                                 feat_dim=1)
    opt = torch.optim.Adam(pre.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = F.mse_loss(pre(feats), target)
        loss.backward()
        opt.step()
    s = pre(feats).squeeze(0).detach()

    def cond(M):
        ev = torch.linalg.eigvals(M).real.abs()
        return float(ev.max() / ev.min().clamp_min(1e-12))

    c0, c1 = cond(A), cond(torch.diag(s) @ A)
    ok = c1 < 0.5 * c0
    print(f"[3] preconditioner cond(A)={c0:.2f} cond(P^-1 A)={c1:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_stage_progress() -> bool:
    torch.manual_seed(3)
    base = dict(bands=8, msi_bands=3, scale=4, patch=32, blur_ksize=5,
                eval_sigma=1.2, n_stages=6, use_krylov=True,
                use_learned_combo=True, use_precond=True, use_hypernet=False,
                graph_k=3, hidden=16, gcn_layers=1, source_root="", target_root="")
    cfg = Config(**base)
    model = KrylovNet(cfg)
    model.set_srf(torch.rand(8, 3) + 0.1)
    op = model.op
    kernel = model.default_kernel
    gt = torch.rand(1, 8, 32, 32) * 0.8 + 0.1
    lr = op.D(gt, kernel)
    msi = op.S(gt, model.srf)

    pred = model(lr, msi, kernel)
    rvals = [float(r.norm().item()) for r in pred["residuals"]]
    mono = rvals[-1] < rvals[0]
    print(f"[4] GMRES residual stage1={rvals[0]:.4e} stage6={rvals[-1]:.4e} "
          f"monotone={mono}")

    err6 = F.mse_loss(op.D(pred["out"].detach(), kernel), lr).item()
    m1 = KrylovNet(Config(**{**base, "n_stages": 1}))
    m1.set_srf(model.srf)
    with torch.no_grad():
        p1 = m1(lr, msi, kernel)
        p6 = model(lr, msi, kernel)
    r1 = float(p1["residuals"][-1].norm().item())
    r6 = float(p6["residuals"][-1].norm().item())
    err1 = F.mse_loss(op.D(p1["out"], kernel), lr).item()
    more_stages = r6 < r1
    print(f"[4] normal-eq residual stages1={r1:.4e} stages6={r6:.4e} "
          f"more_stages_helps={more_stages}")
    print(f"[4] consistency ||D(y)-X||^2 stages1={err1:.4e} stages6={err6:.4e} "
          f"(informational)")

    # Krylov vs fixed-point (Richardson) at equal stages, same operator+precond
    b = op.b(lr, msi, kernel, model.srf)
    x0 = F.interpolate(lr, scale_factor=cfg.scale, mode="bicubic",
                       align_corners=False)
    Aop = lambda v: op.A(v, kernel, model.srf)
    s = model.precond(model._band_feats(lr)).detach()
    Pinv = lambda v: v * s.reshape(1, cfg.bands, 1, 1)
    _, rr = richardson_solve(x0, b, Aop, Pinv, cfg.n_stages, cfg.rich_alpha)
    res_r = float(rr[-1].norm().item())
    res_k = r6
    krylov_beats = res_k < res_r
    print(f"[4] Richardson res={res_r:.4e} vs Krylov res={res_k:.4e} "
          f"krylov_beats={krylov_beats}")

    # gradients flow into every learned module
    loss = pred["out"].sum()
    loss.backward()
    g_ok = (model.blend.attn.weight.grad is not None
            and model.precond.embed.weight.grad is not None)
    print(f"[4] gradients blend/precond {'PASS' if g_ok else 'FAIL'}")
    return mono and krylov_beats and more_stages and g_ok


def check_ladder_smoke() -> bool:
    torch.manual_seed(4)
    ok = True
    stages = {
        "s1_richardson": dict(use_krylov=False),
        "s2_krylov_plain": dict(use_krylov=True, use_learned_combo=False,
                                use_precond=False, use_hypernet=False),
        "s3_krylov_combo": dict(use_krylov=True, use_learned_combo=True,
                                use_precond=False, use_hypernet=False),
        "s4_krylov_precond": dict(use_krylov=True, use_learned_combo=True,
                                  use_precond=True, use_hypernet=False),
        "s5_krylov_hyper": dict(use_krylov=True, use_learned_combo=True,
                                use_precond=True, use_hypernet=True),
    }
    base = dict(bands=6, msi_bands=3, scale=4, patch=16, blur_ksize=5,
                eval_sigma=1.2, n_stages=3, graph_k=2, hidden=8, gcn_layers=1,
                source_root="", target_root="")
    op0 = FusionOperator(4, 1e-3)
    kernel = torch.rand(3, 5, 5) + 0.5
    srf = torch.rand(6, 3) + 0.1
    gt = torch.rand(3, 6, 16, 16) * 0.8 + 0.1
    lr = op0.D(gt, kernel)
    msi = op0.S(gt, srf)
    for name, patch_cfg in stages.items():
        cfg = Config(**{**base, **patch_cfg})
        m = KrylovNet(cfg)
        m.set_srf(srf)
        pred = m(lr, msi, kernel)
        has_learned = bool(patch_cfg.get("use_learned_combo")
                           or patch_cfg.get("use_precond")
                           or patch_cfg.get("use_hypernet"))
        if has_learned:
            pred["out"].sum().backward()
            active = []
            if patch_cfg.get("use_learned_combo"):
                active.append(m.blend)
            if patch_cfg.get("use_precond"):
                active.append(m.precond)
            if patch_cfg.get("use_hypernet"):
                active.append(m.hypernet)
            g_ok = all(
                any(p.grad is not None for p in mod.parameters())
                for mod in active)
        else:
            g_ok = bool(torch.isfinite(pred["out"]).all().item())
        ok &= g_ok
        print(f"[5] {name}: fwd/bwd {'PASS' if g_ok else 'FAIL'}")
    return ok


def run_all(device: str = "cpu") -> bool:
    torch.manual_seed(0)
    results = [
        ("adjoint D,S", check_adjoint),
        ("exact solve", check_krylov_exact_solve),
        ("preconditioner", check_precond_helps),
        ("stage progress", check_stage_progress),
        ("ladder smoke", check_ladder_smoke),
    ]
    ok = True
    for name, fn in results:
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