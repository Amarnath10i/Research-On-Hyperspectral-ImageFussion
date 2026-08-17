"""Numerical self-checks.

Each of these turns a claim from the design document into something a reviewer
can run. They execute in a few seconds on CPU and are the first cell of the
Kaggle notebook, so a broken environment fails loudly and immediately rather
than 3 hours into training.
"""

from __future__ import annotations

import torch

from .config import Config
from .degrade import blur_downsample, gaussian_kernel2d
from .engine import test_time_adapt, tiled_inference
from .losses import SPCLoss
from .metrics import evaluate_arrays
from .model import DAETFNet
from .modules import (EquivariantFeatureExtractor, HaarDWT,
                      TensorSpectralSpatialEncoder)
from .spectral_embed import (ProjectiveSpectralEmbedding,
                             check_intensity_invariance,
                             check_metric_calibration)


def check_equivariance(device: str = "cpu", tol: float = 1e-4) -> float:
    """rot90(EFE(x)) must equal EFE(rot90(x))."""
    torch.manual_seed(0)
    efe = EquivariantFeatureExtractor(5, 8, 12, depth=2).to(device).eval()
    x = torch.randn(2, 5, 32, 32, device=device)
    with torch.no_grad():
        a = torch.rot90(efe(x), 1, (-2, -1))
        b = efe(torch.rot90(x, 1, (-2, -1)))
    err = float((a - b).abs().max())
    print(f"[check] p4 equivariance max|err| = {err:.3e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


def check_wavelet(tol: float = 1e-5) -> float:
    """IDWT(DWT(x)) must reconstruct x exactly (orthonormal Haar)."""
    dwt = HaarDWT()
    x = torch.randn(2, 7, 16, 16)
    err = float((dwt.inverse(dwt(x)) - x).abs().max())
    print(f"[check] Haar DWT reconstruction max|err| = {err:.3e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


def check_core_used() -> bool:
    """The Tucker core must receive gradient - the v1 bug was that it did not."""
    t = TensorSpectralSpatialEncoder(8, 8, 8, rank=4)
    a, b = torch.randn(1, 8, 8, 8), torch.randn(1, 8, 8, 8)
    t(a, b).sum().backward()
    ok = t.core.grad is not None and float(t.core.grad.abs().sum()) > 0
    print(f"[check] Tucker core receives gradient ({'PASS' if ok else 'FAIL'})")
    return ok


def check_observation_model(tol: float = 1e-5) -> bool:
    """Down(HR) must reproduce the LR observation the dataset built, otherwise
    the spatial-consistency term is penalising the wrong thing."""
    gt = torch.rand(1, 5, 64, 64)
    k = gaussian_kernel2d(9, 1.2, 1.2, 0.0)
    lr_a = blur_downsample(gt, k, 4)
    lr_b = blur_downsample(gt, k.unsqueeze(0), 4)     # per-sample kernel path
    err = float((lr_a - lr_b).abs().max())
    ok = err < tol and lr_a.shape[-1] == 16
    print(f"[check] observation model consistent, shape {tuple(lr_a.shape)}, "
          f"max|err| {err:.2e} ({'PASS' if ok else 'FAIL'})")
    return ok


def check_tta_isolation(device: str = "cpu", tol: float = 1e-6) -> bool:
    """Test-time adaptation must not leak between scenes.

    With restore=True the weights must come back exactly, so adapting scene B
    cannot inherit scene A's adaptation. Otherwise the reported cross-domain
    numbers would depend on the order the test scenes happen to be listed in.
    """
    torch.manual_seed(0)
    cfg = Config(patch=32, width=16, equi_width=4, rank=4, bands=31, msi_bands=3)
    model = DAETFNet(cfg).to(device)
    crit = SPCLoss(cfg, torch.rand(cfg.bands, cfg.msi_bands)).to(device)
    gt = torch.rand(1, cfg.bands, 32, 32, device=device)
    lr = blur_downsample(gt, gaussian_kernel2d(9, 1.2, 1.2, 0.0), cfg.scale)
    msi = torch.rand(1, cfg.msi_bands, 32, 32, device=device)

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    first = test_time_adapt(model, lr, msi, crit, steps=3, restore=True)
    drift = max((v - before[k]).abs().max().item()
                for k, v in model.state_dict().items() if v.is_floating_point())
    second = test_time_adapt(model, lr, msi, crit, steps=3, restore=True)
    repeat = (first - second).abs().max().item()

    ok = drift < tol and repeat < tol
    print(f"[check] TTA isolation: weight drift {drift:.2e}, "
          f"repeat difference {repeat:.2e} ({'PASS' if ok else 'FAIL'})")
    return ok


def smoke_test(device: str = "cpu", check_ablations: bool = True) -> None:
    """End-to-end shape/gradient check on synthetic tensors."""
    cfg = Config(patch=32, width=32, equi_width=8, rank=8, batch=2,
                 bands=31, msi_bands=3)
    model = DAETFNet(cfg).to(device)
    srf = torch.rand(cfg.bands, cfg.msi_bands)
    crit = SPCLoss(cfg, srf).to(device)
    gt = torch.rand(2, cfg.bands, cfg.patch, cfg.patch, device=device)
    lr = blur_downsample(gt, gaussian_kernel2d(9, 1.2, 1.2, 0.0), cfg.scale)
    msi = torch.rand(2, cfg.msi_bands, cfg.patch, cfg.patch, device=device)

    out = model(lr, msi)
    assert out["out"].shape == gt.shape, (out["out"].shape, gt.shape)
    loss, logs = crit(out, gt, lr, msi, model, deg_gt=torch.rand(2, 5, device=device))
    loss.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for _ in model.parameters())
    print(f"[check] forward {tuple(out['out'].shape)}  loss {loss.item():.4f}  "
          f"params {model.n_params() / 1e6:.2f}M  tensors with grad {grads}/{total}")
    print(f"[check] loss terms: { {k: round(v, 4) for k, v in logs.items()} }")

    big_gt = torch.rand(1, cfg.bands, 96, 96, device=device)
    big_lr = blur_downsample(big_gt, gaussian_kernel2d(9, 1.2, 1.2, 0.0), cfg.scale)
    big_msi = torch.rand(1, cfg.msi_bands, 96, 96, device=device)
    pred = tiled_inference(model, big_lr, big_msi, cfg.scale, tile_hr=32, overlap=8)
    assert pred.shape == big_gt.shape, (pred.shape, big_gt.shape)
    m = evaluate_arrays(pred[0].detach().cpu().numpy().transpose(1, 2, 0),
                        big_gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
    print(f"[check] tiled inference {tuple(pred.shape)} PASS; "
          f"metrics { {k: round(v, 3) for k, v in m.items()} }")

    if check_ablations:
        for switch in ("use_equivariant", "use_tsse", "use_moe", "use_fdrm",
                       "use_backprojection", "use_degradation_code"):
            acfg = Config(patch=32, width=32, equi_width=8, rank=8, batch=2,
                          bands=31, msi_bands=3, **{switch: False})
            am = DAETFNet(acfg).to(device)
            ao = am(lr, msi)
            assert ao["out"].shape == gt.shape
            acrit = SPCLoss(acfg, srf).to(device)
            aloss, _ = acrit(ao, gt, lr, msi, am,
                             deg_gt=torch.rand(2, 5, device=device))
            aloss.backward()
            print(f"[check] ablation {switch}=False OK "
                  f"({am.n_params() / 1e6:.2f}M params)")


def run_all(device: str = "cpu") -> bool:
    """Every check. Returns True when all of them pass."""
    ok = True
    ok &= check_equivariance(device) < 1e-4
    ok &= check_wavelet() < 1e-5
    ok &= check_core_used()
    ok &= check_observation_model()
    ok &= check_tta_isolation(device)

    # v5: the projective spectral embedding.  The headline claim is that the
    # network fuses on an illumination-invariant, SAM-metric-calibrated
    # manifold, so intensity cannot leak into the learned representation and
    # optimising L2 there is optimising spectral angle.
    ok &= check_intensity_invariance()
    ok &= check_metric_calibration() < 0.15
    ok &= check_embed_in_loss(device)

    # v4: the range/null decomposition. These verify the central claim - that
    # D(Y_hat) = X is an algebraic identity rather than something the loss
    # negotiates - before any training time is spent.
    from . import nullspace as _ns
    ok &= _ns.check_adjoint() < 1e-5
    ok &= _ns.check_consistency() < 1e-3
    ok &= _ns.check_null_annihilation() < 1e-3
    ok &= _ns.check_idempotent() < 1e-3
    ok &= check_network_consistency(device)

    smoke_test(device)
    print(f"\n[selfcheck] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


def check_embed_in_loss(device: str = "cpu") -> bool:
    """The embedding terms receive gradient through the assembled network and
    loss: the headline contribution must actually participate in training."""
    from .config import Config
    from .model import DAETFNet
    from .losses import SPCLoss
    from .degrade import blur_downsample, gaussian_kernel2d

    torch.manual_seed(0)
    cfg = Config(bands=31, msi_bands=3, patch=32, scale=4, width=16,
                 equi_width=4, rank=4, code_dim=32, use_moe=False,
                 use_tsse=False, use_equivariant=False, use_fdrm=False,
                 use_projective_embed=True)
    m = DAETFNet(cfg).to(device)
    crit = SPCLoss(cfg, torch.rand(cfg.bands, cfg.msi_bands)).to(device)
    gt = torch.rand(1, cfg.bands, 32, 32, device=device)
    lr = blur_downsample(gt, gaussian_kernel2d(9, 1.2, 1.2, 0.0), cfg.scale)
    msi = torch.rand(1, cfg.msi_bands, 32, 32, device=device)
    out = m(lr, msi)
    loss, logs = crit(out, gt, lr, msi, m, deg_gt=torch.rand(1, 5, device=device))
    loss.backward()
    grads = sum(1 for p in m.embed.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for _ in m.embed.parameters())
    ok = grads == total and "embed" in logs and "cal" in logs
    print(f"[check] embedding participates in loss: embed grads {grads}/{total}, "
          f"terms {logs.get('embed', float('nan')):.4f}/{logs.get('cal', float('nan')):.4f}"
          f" ({'PASS' if ok else 'FAIL'})")
    return ok


def check_network_consistency(device: str = "cpu", tol: float = 1e-2) -> bool:
    """D(Y_hat) = X through the whole assembled network, then again after the
    reconstruction head is deliberately wrecked.

    The second half is the point. A physics *loss* degrades when the network
    misbehaves; an identity cannot. If this ever starts failing under the
    stress case, the decomposition has been bypassed somewhere.
    """
    from .config import Config
    from .model import DAETFNet
    from .nullspace import RangeNullProjector

    torch.manual_seed(0)
    cfg = Config(bands=31, msi_bands=3, scale=4, width=32, equi_width=8,
                 rank=8, code_dim=64)
    if not cfg.use_nullspace:
        print("[check] network consistency SKIPPED (use_nullspace=False)")
        return True
    m = DAETFNet(cfg).to(device).eval()
    lr = torch.rand(2, 31, 16, 16, device=device)
    msi = torch.rand(2, 3, 64, 64, device=device)
    P = RangeNullProjector(cfg.scale, ksize=cfg.blur_ksize, sigma=cfg.eval_sigma,
                           cg_steps=cfg.cg_steps, ridge=cfg.ridge).to(device)
    with torch.no_grad():
        out = m(lr, msi)
        e_norm = ((P.D(out["out"], out["kernel"]) - lr).abs().max()
                  / lr.abs().max().clamp_min(1e-12)).item()
        for p in m.recon.parameters():
            p.mul_(50.0)
        out2 = m(lr, msi)
        e_stress = ((P.D(out2["out"], out2["kernel"]) - lr).abs().max()
                    / lr.abs().max().clamp_min(1e-12)).item()
    ok = e_norm < tol and e_stress < tol
    print(f"[check] network D(Y)=X: {e_norm:.2e} normal, {e_stress:.2e} with the "
          f"recon head x50 ({'PASS' if ok else 'FAIL'})")
    return ok


if __name__ == "__main__":
    run_all()
