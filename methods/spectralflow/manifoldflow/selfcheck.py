"""Self-checks for ManifoldFlow (CPU-fast, no data required).

Verifies the structural claims of Proposal 5:
  [1] Tangent constraint: D(P_perp v) ~ 0 and D(consistent set) == X exactly.
  [2] Flow matching: the velocity field learns the rectified target u = gt - y0.
  [3] Few-step sampling: 4 Euler steps are close to 32 (straight trajectories,
      "~10x fewer steps"), and both land closer to gt than the start y0.
  [4] Consistency during sampling: every iterate satisfies D(y_k) == X.
  [5] Ladder stages build, forward and backprop.

Returns True iff all checks pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config
from .model import ManifoldFlow


def _scene(cfg: Config, seed: int, batch: int = 8):
    torch.manual_seed(seed)
    model = ManifoldFlow(cfg)
    model.set_srf(torch.rand(cfg.bands, cfg.msi_bands) + 0.1)
    D = model.projector.D
    gt = torch.rand(batch, cfg.bands, cfg.patch, cfg.patch) * 0.8 + 0.1
    x = D(gt)                                   # use the projector's own D/kernel
    m = torch.einsum("bchw,cm->bmhw", gt, model.srf)
    return model, D, gt, x, m


def check_tangent() -> bool:
    cfg = Config(bands=6, msi_bands=3, scale=4, patch=24, cg_steps=30,
                 cg_ridge=1e-6, source_root="", target_root="")
    model, D, _, x, _ = _scene(cfg, 0, batch=2)
    v = torch.randn(2, cfg.bands, cfg.patch, cfg.patch) * 3.0
    pv = model.projector.project_null(v)
    err = (D(pv) - 0).abs().max().item() / max(x.abs().max().item(), 1e-12)
    cons = model.projector.consistent(x, v)
    cerr = (D(cons) - x).abs().max().item() / max(x.abs().max().item(), 1e-12)
    ok = err < 1e-3 and cerr < 1e-3
    print(f"[1] tangent D(P_perp v) max={err:.2e} ; D(consistent)=X max={cerr:.2e} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_flow_matching_and_straight() -> bool:
    cfg = Config(bands=6, msi_bands=3, scale=4, patch=24, hidden=32,
                 n_flow_blocks=3, tangent=True, straightness_reg=False,
                 cg_steps=12, cg_ridge=1e-6, source_root="", target_root="")
    model, D, gt, x, m = _scene(cfg, 1, batch=8)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for it in range(1200):
        t = torch.rand(8)
        pred = model.training_step(x, m, gt, t)
        loss = F.mse_loss(pred["velocity"], pred["target"])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    y0 = model.pinv(x, (cfg.patch, cfg.patch))
    dist0 = torch.linalg.vector_norm(y0 - gt, dim=(1, 2, 3))
    y4 = model.sample(x, m, steps=4)
    y32 = model.sample(x, m, steps=32)
    dist4 = torch.linalg.vector_norm(y4 - gt, dim=(1, 2, 3))
    dist32 = torch.linalg.vector_norm(y32 - gt, dim=(1, 2, 3))
    straight = torch.linalg.vector_norm(y4 - y32, dim=(1, 2, 3))

    r4 = float((dist4 / dist0.clamp_min(1e-9)).mean())
    r32 = float((dist32 / dist0.clamp_min(1e-9)).mean())
    sr = float((straight / dist0.clamp_min(1e-9)).mean())
    ok = losses[0] > 5 * losses[-1] and r4 < 0.6 and r32 < 0.6 and sr < 0.4
    print(f"[2] flow loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"(x{losses[0] / losses[-1]:.1f})")
    print(f"[3] few-step 4/32 steps err/g0 = {r4:.3f}/{r32:.3f} "
          f"straightness |y4-y32|/|g0| = {sr:.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_consistency_sampling() -> bool:
    cfg = Config(bands=6, msi_bands=3, scale=4, patch=24, hidden=16,
                 n_flow_blocks=2, tangent=True, cg_steps=12, cg_ridge=1e-6,
                 sample_steps=8, source_root="", target_root="")
    model, D, _, x, m = _scene(cfg, 2, batch=2)
    y = model.pinv(x, (cfg.patch, cfg.patch))
    kernel = None
    ok = True
    for k in range(cfg.sample_steps):
        t = (k + 0.5) / cfg.sample_steps
        v = model.velocity(y, m, t)
        v = model.projector.project_null(v, kernel)
        y = y + v / cfg.sample_steps
        err = (D(y) - x).abs().max().item() / max(x.abs().max().item(), 1e-12)
        ok &= err < 1e-2
        print(f"[4] k={k}  max|D(y_k)-X|/max|X| = {err:.2e} "
              f"{'PASS' if err < 1e-2 else 'FAIL'}")
    return ok


def check_ladder_smoke() -> bool:
    torch.manual_seed(3)
    ok = True
    base = dict(bands=6, msi_bands=3, scale=4, patch=16, hidden=12,
                n_flow_blocks=1, cg_steps=8, source_root="", target_root="")
    for name, pc in {
        "s1_unconstrained": dict(tangent=False),
        "s2_tangent": dict(tangent=True),
        "s3_straightness": dict(tangent=True, straightness_reg=True),
    }.items():
        cfg = Config(**{**base, **pc})
        m = ManifoldFlow(cfg)
        m.set_srf(torch.rand(cfg.bands, cfg.msi_bands) + 0.1)
        D = m.projector.D
        gt = torch.rand(2, cfg.bands, 16, 16) * 0.8 + 0.1
        x = D(gt)
        ms = torch.einsum("bchw,cm->bmhw", gt, m.srf)
        t = torch.rand(2)
        pred = m.training_step(x, ms, gt, t)
        pred["velocity"].sum().backward()
        g = all(p.grad is not None for p in m.velocity.parameters()
                if p.requires_grad)
        ok &= g
        print(f"[5] {name}: fwd/bwd {'PASS' if g else 'FAIL'}")
    # sampling stage
    cfg = Config(**{**base, "sample_steps": 4})
    m = ManifoldFlow(cfg)
    m.set_srf(torch.rand(cfg.bands, cfg.msi_bands) + 0.1)
    D = m.projector.D
    gt = torch.rand(1, cfg.bands, 16, 16) * 0.8 + 0.1
    x = D(gt)
    ms = torch.einsum("bchw,cm->bmhw", gt, m.srf)
    y = m.sample(x, ms, steps=4)
    g = bool(torch.isfinite(y).all().item())
    ok &= g
    print(f"[5] s4_sample_4steps: {'PASS' if g else 'FAIL'}")
    return ok


def run_all(device: str = "cpu") -> bool:
    torch.manual_seed(0)
    checks = [
        ("tangent", check_tangent),
        ("flow matching", check_flow_matching_and_straight),
        ("consistency sampling", check_consistency_sampling),
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