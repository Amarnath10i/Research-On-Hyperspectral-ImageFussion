"""SLT selfchecks: prove the manifold algebra and the illumination claim.

All checks are deterministic (fixed seeds), run on CPU in seconds, and verify
*structural* claims - the same discipline as the other proposals.  No ranking
is claimed here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config
from .manifold import (exp_map, geodesic_distance, l2_normalize, log_map,
                       sam_degrees, tangent_projection)
from .model import SLTNet


def _cfg() -> Config:
    return Config(source_root="", target_root="", bands=8, msi_bands=3,
                  scale=4, patch=32, width=16, depth=2, blur_ksize=5,
                  fix_intensity=True, max_angle_rad=1.4)


def check_exp_map_unit_norm(device: str = "cpu", tol: float = 1e-4) -> bool:
    """The exponential map keeps points on the unit sphere and is
    differentiable."""
    torch.manual_seed(0)
    p = l2_normalize(torch.randn(4, 8, 1, 1, device=device))
    p.requires_grad_(True)
    v = torch.randn(4, 8, 1, 1, device=device)
    v.requires_grad_(True)
    q = exp_map(p, v)
    err = (q.norm(2, 1) - 1).abs().max().item()
    ok = err < tol
    q.sum().backward()
    grad_ok = bool(p.grad.abs().sum().item() > 0 and v.grad.abs().sum().item() > 0)
    print(f"[check] exp_map unit norm  max| ||q|| - 1 | = {err:.2e} "
          f"({'PASS' if ok else 'FAIL'})  grads {'OK' if grad_ok else 'MISSING'}")
    return bool(ok and grad_ok)


def check_sam_is_geodesic(device: str = "cpu", tol: float = 1e-4) -> bool:
    """SAM == geodesic distance on the sphere == |Log_p(q)|."""
    torch.manual_seed(1)
    p = l2_normalize(torch.randn(6, 8, 1, 1, device=device))
    q = l2_normalize(torch.randn(6, 8, 1, 1, device=device))
    g = geodesic_distance(p, q)
    lg = log_map(p, q).norm(2, 1)
    ac = torch.acos((p * q).sum(1).clamp(-1 + 1e-7, 1 - 1e-7))
    err = max((g - lg).abs().max().item(), (g - ac).abs().max().item())
    ok = err < tol
    print(f"[check] SAM is geodesic (|g - |Log||, |g - acos|)  max err = {err:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return bool(ok)


def check_log_exp_roundtrip(device: str = "cpu", tol: float = 1e-4) -> bool:
    """Log(Exp_p(v)) recovers the tangent vector (isometry round-trip)."""
    torch.manual_seed(2)
    p = l2_normalize(torch.randn(4, 8, 1, 1, device=device))
    v = tangent_projection(torch.randn(4, 8, 1, 1, device=device), p)
    v = v / v.norm(2, 1, keepdim=True) * 0.5
    v2 = log_map(p, exp_map(p, v))
    err = (v2 - v).abs().max().item()
    ok = err < tol
    print(f"[check] Log(Exp_p(v)) = v  max err = {err:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return bool(ok)


def check_illumination_invariance(device: str = "cpu", tol: float = 1e-3) -> bool:
    """THE root-cause claim: scaling the LR observation changes output
    intensity but not the spectrum.  The encoder only ever sees scale-invariant
    inputs (dir0 and the normalised MSI), so the transport is equivariant by
    construction.

    A matched Euclidean-residual control (absolute-intensity encoder input)
    incurs ~8 deg SAM under the same 1.7x illumination change - the exact
    failure mode this redesign exists to remove.  The invariant network sits at
    the float32 acos floor (~0.03 deg).
    """
    torch.manual_seed(3)
    cfg = _cfg()
    net = SLTNet(cfg).to(device).eval()          # untrained: structure only
    lr = torch.rand(1, cfg.bands, 8, 8, device=device) * 0.15
    msi = torch.rand(1, cfg.msi_bands, 32, 32, device=device)
    c = 1.7
    out1 = net(lr, msi)["out"]
    out2 = net(c * lr, msi)["out"]
    mask2d = (out1.norm(2, 1) > 0.05).squeeze(0)  # ignore dark spectra
    sam = sam_degrees(out1[:, :, mask2d], out2[:, :, mask2d]).mean().item()
    scale_err = (out2 - c * out1).abs().max().item()

    ctl = Config(**cfg.to_dict())
    ctl.manifold = False
    ctl_net = SLTNet(ctl).to(device).eval()
    o1 = ctl_net(lr, msi)["out"]
    o2 = ctl_net(c * lr, msi)["out"]
    ctl_sam = sam_degrees(o1[:, :, mask2d], o2[:, :, mask2d]).mean().item()

    ok = sam < 0.1 and ctl_sam > 1.0 and scale_err < tol
    print(f"[check] illumination invariance  SAM(1.7x, base) = {sam:.4f} deg "
          f"(fp floor), control (Euclidean residual) = {ctl_sam:.2f} deg, "
          f"|out2 - c*out1|_max = {scale_err:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return bool(ok)


def check_metric_aligned(device: str = "cpu", tol: float = 1e-4) -> bool:
    """The network's SAM is the geodesic distance between the transported
    direction and the truth - the error the model incurs is the metric the
    protocol reports, structurally."""
    torch.manual_seed(4)
    cfg = _cfg()
    net = SLTNet(cfg).to(device).eval()
    lr = torch.rand(1, cfg.bands, 8, 8, device=device) * 0.5
    msi = torch.rand(1, cfg.msi_bands, 32, 32, device=device)
    gt = torch.rand(1, cfg.bands, 32, 32, device=device)
    out = net(lr, msi)["out"]
    a = sam_degrees(out, gt).mean()
    b = torch.rad2deg(geodesic_distance(
        l2_normalize(out), l2_normalize(gt))).mean()
    err = (a - b).abs().item()
    ok = err < tol
    print(f"[check] metric aligned  SAM(out,gt) vs geodesic(dir_out,dir_gt) "
          f"diff = {err:.2e} ({'PASS' if ok else 'FAIL'})")
    return bool(ok)


def smoke_test(device: str = "cpu") -> bool:
    """Forward/backward across every ladder switch."""
    torch.manual_seed(5)
    ok = True
    for manifold, geodesic, guide in ((True, True, True), (True, False, False),
                                      (False, True, False)):
        cfg = _cfg()
        cfg.manifold, cfg.geodesic, cfg.use_msi_guide = manifold, geodesic, guide
        net = SLTNet(cfg).to(device)
        lr = torch.rand(1, cfg.bands, 8, 8, device=device)
        msi = torch.rand(1, cfg.msi_bands, 32, 32, device=device)
        gt = torch.rand(1, cfg.bands, 32, 32, device=device)
        out = net(lr, msi)["out"]
        ok &= tuple(out.shape) == (1, cfg.bands, 32, 32)
        from .losses import SLTLoss
        loss, _ = SLTLoss(cfg)(net(lr, msi), gt)
        loss.backward()
        grads = sum(p.grad is not None and p.grad.abs().sum() > 0
                    for p in net.parameters())
        ok &= grads > 0
    print(f"[check] smoke (ladder switches forward/backward) "
          f"({'PASS' if ok else 'FAIL'})")
    return bool(ok)


def run_all(device: str = "cpu") -> bool:
    ok = True
    ok &= check_exp_map_unit_norm(device)
    ok &= check_sam_is_geodesic(device)
    ok &= check_log_exp_roundtrip(device)
    ok &= check_illumination_invariance(device)
    ok &= check_metric_aligned(device)
    ok &= smoke_test(device)
    print(f"\n[selfcheck] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()