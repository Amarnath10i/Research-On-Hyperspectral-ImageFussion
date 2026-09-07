"""Numerical self-checks for ConsistentFlow.

Each check turns a claim from the design document into something a reviewer
can run in a few seconds on CPU.  The first cell of the Kaggle notebook runs
these before any training time is spent.

Claims verified here:
  1. D(Y_hat) = X after ONE consistency-map forward pass - the algebraic
     guarantee at ~regressor cost (the whole point of the proposal).
  2. The consistency distillation actually learns: the consistency condition
     |f(y_{t+1}) - f(y_t)| shrinks along teacher trajectories.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from .config import Config
from .engine import (ConsistencyModel, build_model, consistent_output,
                     consistency_distill_loss, to_x0_estimate)
from spectralflow.nullspace import RangeNullProjector
from spectralflow.sampler import LinearNoiseSchedule


def _synthetic_patches(n: int, bands: int, size: int) -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.rand(n, bands, 1, 1)
    w = torch.rand(n, 1, size, size) * 0.2
    h = torch.rand(n, 1, size, size) * 0.2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size),
                            indexing="ij")
    ramp = (xx + yy)[None, None].expand(n, 1, size, size)
    out = x + 0.3 * w * ramp + 0.3 * h * (1 - ramp)
    return out.clamp(0, 1)


def _cfg() -> Config:
    return Config(bands=5, msi_bands=3, patch=32, ch=16, ch_mult=(1, 2),
                  num_res=1, cond_dim=32, code_dim=8, num_timesteps=30,
                  sample_steps=1, cg_steps=30, ridge=0.0, iters=60, batch=16)


def check_one_step_consistency(device: str = "cpu", tol: float = 1e-3) -> float:
    """THE claim: one forward pass (no diffusion loop) already satisfies
    D(Y_hat) = X, because the projection is applied after the map.

    The fixed evaluation kernel is supplied explicitly so that generation (lr)
    and verification use the same operator - the blind-kernel consistency is
    verified separately (see ``smoke_test``, which checks against the model's
    own estimated kernel).
    """
    torch.manual_seed(0)
    cfg = _cfg()
    m = ConsistencyModel(cfg).to(device).eval()          # untrained network
    P = RangeNullProjector(cfg.scale, cfg.blur_ksize, cg_steps=30, ridge=0.0)
    gt = _synthetic_patches(1, cfg.bands, 32).to(device)
    msi = torch.rand(1, cfg.msi_bands, 32, 32, device=device)
    lr = P.D(gt)
    m.set_kernel(P.D.default_kernel)                     # same fixed operator
    out = m(lr, msi)["out"]
    err = (P.D(out) - lr).abs().max().item() / max(lr.abs().max().item(), 1e-12)
    ok = err < tol and tuple(out.shape) == (1, cfg.bands, 32, 32)
    print(f"[check] one-step D(Y_hat)=X (fixed kernel)  max rel err = {err:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return err


def _consistency_gap(m: ConsistencyModel, target: torch.nn.Module,
                     teacher: torch.nn.Module, schedule: LinearNoiseSchedule,
                     gt: torch.Tensor, msi: torch.Tensor, lr: torch.Tensor,
                     code: torch.Tensor, kernel: torch.Tensor,
                     device: str, n_ts: int = 4) -> float:
    """Mean |f_phi(y_{t+1}) - f_phi^-(y_t)| along teacher trajectories."""
    with torch.no_grad():
        gaps = []
        for seed in range(n_ts):
            torch.manual_seed(seed)
            t = torch.randint(1, schedule.T, (gt.shape[0],), device=device)
            tp = torch.clamp(t + 1, max=schedule.T)
            a_next = schedule.a[tp].to(gt.dtype).reshape(-1, 1, 1, 1)
            a_t = schedule.a[t].to(gt.dtype).reshape(-1, 1, 1, 1)
            eps = torch.randn_like(gt)
            y_next = torch.sqrt(a_next) * gt + (1 - a_next).clamp_min(1e-8).sqrt() * eps
            eps_t = teacher(y_next, msi, code, tp)
            x0n = (y_next - (1 - a_next).clamp_min(1e-8).sqrt() * eps_t) / a_next.clamp_min(1e-8)
            y_t = torch.sqrt(a_t) * x0n + (1 - a_t).clamp_min(1e-8).sqrt() * eps_t
            f_next = consistent_output(
                to_x0_estimate(m.score_net, y_next, msi, code, tp, schedule),
                lr, kernel, m.projector)
            f_t = consistent_output(
                to_x0_estimate(m.score_net, y_t, msi, code, t, schedule),
                lr, kernel, m.projector)
            gaps.append(float((f_next - f_t).abs().mean().item()))
        return float(np.mean(gaps))


def check_distill_learns(device: str = "cpu", teacher_steps: int = 40,
                         cd_steps: int = 150) -> bool:
    """CD must shrink the consistency gap along held teacher trajectories.

    Mirrors reality: the teacher is first made into a decent denoiser by a short
    score-matching run (on GPU this is the full P5 score net), then the student
    is distilled from it.  A *random* teacher defines a meaningless trajectory
    and CD cannot converge toward it, so the score-matching warm-up is not
    optional here.
    """
    from spectralflow.losses import score_matching_loss

    torch.manual_seed(0)
    cfg = _cfg()
    m = ConsistencyModel(cfg).to(device)
    teacher = copy.deepcopy(m.score_net).to(device)
    target = copy.deepcopy(m.score_net).to(device)
    for p in target.parameters():
        p.requires_grad = False
    schedule = LinearNoiseSchedule(cfg.num_timesteps, cfg.beta_start,
                                   cfg.beta_end).to(device)

    gt = _synthetic_patches(cfg.batch, cfg.bands, 32).to(device)
    msi = torch.rand(cfg.batch, cfg.msi_bands, 32, 32, device=device)
    lr = m.projector.D(gt)
    code0 = m.deg_head(lr, msi)[0].detach()
    kernel = None

    t_opt = torch.optim.AdamW(teacher.parameters(), lr=1e-2)
    for _ in range(teacher_steps):
        t_opt.zero_grad(set_to_none=True)
        t = torch.randint(1, schedule.T + 1, (cfg.batch,), device=device)
        eps = torch.randn_like(gt)
        tloss = score_matching_loss(teacher, schedule, gt, msi, code0, t, eps)
        tloss.backward()
        t_opt.step()
    teacher.eval()

    opt = torch.optim.AdamW(list(m.score_net.parameters()) +
                            list(m.deg_head.parameters()), lr=3e-3)
    gap_before = _consistency_gap(m, target, teacher, schedule, gt, msi, lr,
                                  code0, kernel, device)
    loss0 = None
    for step in range(cd_steps):
        opt.zero_grad(set_to_none=True)
        code = m.deg_head(lr, msi)[0]            # fresh graph each step
        loss = consistency_distill_loss(
            m.score_net, target, teacher, schedule, m.projector,
            gt, msi, lr, code, kernel, use_projection=True)
        if loss0 is None:
            loss0 = float(loss.item())
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, tp in zip(m.score_net.parameters(), target.parameters()):
                tp.data.mul_(cfg.ema_decay).add_(p.data, alpha=1 - cfg.ema_decay)
    loss1 = float(loss.item())
    gap_after = _consistency_gap(m, target, teacher, schedule, gt, msi, lr,
                                 code0, kernel, device)
    ok = gap_after < gap_before and loss1 < loss0
    print(f"[check] distillation: CD loss {loss0:.5f} -> {loss1:.5f}, "
          f"consistency gap {gap_before:.4f} -> {gap_after:.4f} "
          f"({'PASS' if ok else 'FAIL'})")
    return bool(ok)


def smoke_test(device: str = "cpu") -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    m = build_model(cfg).to(device)
    lr = torch.rand(1, cfg.bands, 8, 8, device=device)
    msi = torch.rand(1, cfg.msi_bands, 32, 32, device=device)
    out = m(lr, msi)
    assert tuple(out["out"].shape) == (1, cfg.bands, 32, 32), out["out"].shape
    assert tuple(out["kernel"].shape) == (1, cfg.blur_ksize, cfg.blur_ksize)
    assert tuple(out["deg"].shape) == (1, 5)
    P = RangeNullProjector(cfg.scale, cfg.blur_ksize, cg_steps=30, ridge=0.0)
    err = (P.D(out["out"], out["kernel"]) - lr).abs().max().item()
    print(f"[check] SamplingModel one-step D(Y_hat)=X  max|err| = {err:.2e} "
          f"({'PASS' if err < 1e-3 else 'FAIL'})")
    assert err < 1e-3


def run_all(device: str = "cpu") -> bool:
    ok = True
    ok &= check_one_step_consistency(device) < 1e-3
    ok &= check_distill_learns(device)
    smoke_test(device)
    print(f"\n[selfcheck] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()
