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
    smoke_test(device)
    print(f"\n[selfcheck] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()
