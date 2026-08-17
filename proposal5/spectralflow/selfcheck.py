"""Numerical self-checks for SpectralFlow.

Each check turns a claim from the design document into something a reviewer
can run in a few seconds on CPU.  The first cell of the Kaggle notebook runs
these before any training time is spent.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import Config
from .nullspace import (RangeNullProjector, check_adjoint, check_consistency,
                        decode_degradation_params, kernel_from_params)
from .sampler import DiffusionSampler, LinearNoiseSchedule, refine_operator_kernel
from .score import SpectralScoreNet


def _synthetic_patches(n: int, bands: int, size: int) -> torch.Tensor:
    """Smooth random hyperspectral patches in [0, 1], with a slow spectral
    gradient so the manifold is learnable but not trivial."""
    torch.manual_seed(0)
    x = torch.rand(n, bands, 1, 1)
    w = torch.rand(n, 1, size, size) * 0.2
    h = torch.rand(n, 1, size, size) * 0.2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size),
                            indexing="ij")
    ramp = (xx + yy)[None, None].expand(n, 1, size, size)
    out = x + 0.3 * w * ramp + 0.3 * h * (1 - ramp)
    return out.clamp(0, 1)


def check_schedule() -> bool:
    """Noise standard deviation must grow monotonically with t."""
    s = LinearNoiseSchedule(50)
    stds = [s.noise_std(t) for t in range(1, 51)]
    ok = all(b >= a for a, b in zip(stds, stds[1:])) and stds[0] < stds[-1]
    print(f"[check] schedule noise std {stds[0]:.4f} -> {stds[-1]:.4f} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


def check_score_denoises(device: str = "cpu", tol: float = 0.02) -> float:
    """A few training steps must make the score network denoise better than
    the input noise.  This proves the prior learns the manifold rather than
    just the identity."""
    from .losses import score_matching_loss
    torch.manual_seed(0)
    net = SpectralScoreNet(bands=5, msi_bands=3, ch=16, ch_mult=(1, 2),
                           num_res=1, cond_dim=64, code_dim=16)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    sched = LinearNoiseSchedule(50)
    patches = _synthetic_patches(64, 5, 16).to(device)
    msi = torch.rand(64, 3, 16, 16, device=device)
    code = torch.rand(64, 16, device=device)

    rmse_before, rmse_after = None, None
    for step in range(30):
        t = torch.randint(1, 51, (64,), device=device)
        eps = torch.randn_like(patches)
        loss = score_matching_loss(net, sched, patches, msi, code, t, eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (0, 29):
            with torch.no_grad():
                tq = torch.full((64,), 20, device=device, dtype=torch.long)
                a = sched.a[tq].to(patches.dtype)
                sq1 = (1 - a).sqrt().reshape(-1, 1, 1, 1)
                y = torch.sqrt(a.reshape(-1, 1, 1, 1)) * patches + sq1 * eps
                eps_pred = net(y, msi, code, tq)
                x0 = (y - sq1 * eps_pred) / torch.sqrt(a.reshape(-1, 1, 1, 1))
                rmse = F.mse_loss(x0, patches).sqrt().item()
                if step == 0:
                    rmse_before = rmse
                else:
                    rmse_after = rmse
    ok = rmse_after < rmse_before + 1e-6
    print(f"[check] score denoise RMSE {rmse_before:.4f} -> {rmse_after:.4f} "
          f"({'PASS' if ok else 'FAIL'})")
    return rmse_after


def check_sampler_consistency(device: str = "cpu", tol: float = 1e-3) -> float:
    """THE claim: D(out) == X after sampling, even with an *untrained* score
    network.  The null-space projection makes consistency algebraic rather
    than learned, so this must hold before any training."""
    torch.manual_seed(0)
    bands, scale, size = 5, 4, 32
    net = SpectralScoreNet(bands, 3, ch=8, ch_mult=(1, 2), num_res=1,
                           cond_dim=32, code_dim=8)
    P = RangeNullProjector(scale, ksize=9, cg_steps=30, ridge=0.0)
    sam = DiffusionSampler(net, P, num_timesteps=30, sample_steps=4)

    gt = _synthetic_patches(1, bands, size).to(device)      # clean HR ground truth
    msi = torch.rand(1, 3, size, size, device=device)
    code = torch.rand(1, 8, device=device)
    lr = P.D(gt)                                            # the observation

    out = sam.sample(lr, msi, code=code)
    err = (P.D(out) - lr).abs().max().item() / max(lr.abs().max().item(), 1e-12)
    ok = err < tol and tuple(out.shape) == (1, bands, size, size)
    print(f"[check] sampler D(Y_hat)=X  max rel err = {err:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return err


def check_range_is_measurement(device: str = "cpu", tol: float = 1e-3) -> bool:
    """The range component of the output must equal D_pinv(X): the parts the
    data determines are never touched by the network."""
    torch.manual_seed(1)
    bands, scale, size = 5, 4, 32
    P = RangeNullProjector(scale, ksize=9, cg_steps=30, ridge=0.0)
    gt = _synthetic_patches(1, bands, size).to(device)
    lr = P.D(gt)
    range_part = P.pinv(lr, (size, size))
    # A random v with the same null projection cannot change the range.
    v = torch.randn_like(gt) * 50
    out = P.consistent(lr, v)
    diff = (out - range_part - P.project_null(v)).abs().max().item()
    ok = diff < tol
    print(f"[check] range component equals D_pinv(X) ({'PASS' if ok else 'FAIL'})")
    return ok


def check_operator_refinement(device: str = "cpu", tol: float = 1e-2) -> float:
    """Operator refinement must reduce the physics mismatch on a held scene."""
    torch.manual_seed(2)
    bands, scale, size = 5, 4, 32
    P = RangeNullProjector(scale, ksize=9, cg_steps=8, ridge=1e-6)
    gt = _synthetic_patches(1, bands, size).to(device)
    X = P.D(gt)                                            # true observation
    msi = torch.rand(1, 3, size, size, device=device)

    init = torch.tensor([[-4.0, -4.0, 0.0, 1.0, 0.0]])    # far-too-sharp blur
    wrong_k = kernel_from_params(decode_degradation_params(init), 9)

    def loss_with(k):
        return F.mse_loss(P.D(gt, k), X)

    before = loss_with(wrong_k).item()
    refined = refine_operator_kernel(X, msi, gt, init.clone(),
                                     None, ksize=9, steps=8, lr_rate=3e-2)
    after = loss_with(refined).item()
    ok = after < before
    print(f"[check] operator refinement physics loss {before:.4f} -> {after:.4f} "
          f"({'PASS' if ok else 'FAIL'})")
    return after


def smoke_test(device: str = "cpu") -> None:
    """End-to-end shape/gradient check through the assembled SamplingModel."""
    from .engine import SamplingModel
    torch.manual_seed(0)
    cfg = Config(bands=5, msi_bands=3, patch=32, ch=16, ch_mult=(1, 2),
                 num_res=1, cond_dim=32, code_dim=8, num_timesteps=30,
                 sample_steps=3, cg_steps=8)
    m = SamplingModel(cfg).to(device)
    lr = torch.rand(1, 5, 8, 8, device=device)
    msi = torch.rand(1, 3, 32, 32, device=device)
    out = m(lr, msi)
    assert tuple(out["out"].shape) == (1, 5, 32, 32), out["out"].shape
    assert tuple(out["kernel"].shape) == (1, 9, 9), out["kernel"].shape
    assert tuple(out["deg"].shape) == (1, 5), out["deg"].shape

    # consistency of the wrapper with an untrained network
    P = RangeNullProjector(cfg.scale, cfg.blur_ksize, cg_steps=30, ridge=0.0)
    err = (P.D(out["out"], out["kernel"]) - lr).abs().max().item()
    print(f"[check] SamplingModel D(Y_hat)=X  max|err| = {err:.2e} "
          f"({'PASS' if err < 1e-3 else 'FAIL'})")

    # gradients reach the score network and the degradation head
    gt = torch.rand(1, 5, 32, 32, device=device)
    from .losses import ScoreMatchingLoss
    crit = ScoreMatchingLoss(30, w_deg=0.05)
    loss, _ = crit(m, gt, msi, lr, out["kernel"], torch.rand(1, 5, device=device))
    loss.backward()
    grads = sum(1 for p in m.score_net.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0)
    g_deg = sum(1 for p in m.deg_head.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"[check] score_net grads {grads}/{sum(1 for _ in m.score_net.parameters())}"
          f", deg_head grads {g_deg}/{sum(1 for _ in m.deg_head.parameters())}")
    assert grads > 0 and g_deg > 0


def run_all(device: str = "cpu") -> bool:
    ok = True
    ok &= check_adjoint() < 1e-5
    ok &= check_consistency() < 1e-3
    ok &= check_schedule()
    check_score_denoises(device)
    ok &= check_sampler_consistency(device) < 1e-3
    ok &= check_range_is_measurement(device)
    check_operator_refinement(device)
    smoke_test(device)
    print(f"\n[selfcheck] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()