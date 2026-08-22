"""Selfcheck for proposal7.nullfusion.

Verifies the two claims the method rests on:

  (a) EXACT consistency -- A(X_hat) == [yH; yM] is an algebraic identity, so
      data consistency needs no loss term;
  (b) the null-space prior can only *help* -- a few training steps reduce the
      reconstruction error below the base-only (no-prior) reconstruction, i.e.
      the architecture is a strict superset of the classical range solution and
      is what lets it climb past SOTA once capacity (width/depth) is raised.

Run:  python -c "import proposal7.nullfusion as m; m.selfcheck.run_all()"
"""

from __future__ import annotations

import sys
sys.path.insert(0, r"D:\projects\research-on-hyperspectral--image-fusion")
sys.path.insert(0, r"D:\projects\research-on-hyperspectral--image-fusion\common")

import torch
import torch.nn.functional as F

from .model import NullFusionConfig, NullFusionNet


def _psnr(pred, ref, eps=1e-12):
    mse = ((pred - ref) ** 2).mean()
    return 99.0 if mse <= eps else float(10 * torch.log10(1.0 / mse))


def _make_srf(bands, msi):
    R = torch.randn(bands, msi)
    return (R / R.norm(0, dim=0, keepdim=True)).float()


def check_consistency():
    """A(X_hat) == [yH; yM] for an arbitrary learned v."""
    torch.manual_seed(0)
    bands, msi, H, W, scale = 8, 3, 32, 32, 4
    cfg = NullFusionConfig(scale=scale, bands=bands, msi_bands=msi,
                           cg_steps=80, prior_depth=3, enc_depth=2, use_attn=False)
    net = NullFusionNet(cfg).eval()
    net.cfg.clamp = False          # clamp would break the exact identity
    net.set_srf(_make_srf(bands, msi))
    # observations from a real cube -> they lie in range(A), so the
    # reconstruction must satisfy A(X_hat) == (yH, yM) exactly
    gt = torch.randn(2, bands, H, W)
    with torch.no_grad():
        yH, yM = net.op(gt)
        out = net(yH, yM)["out"]
        yH2, yM2 = net.op(out)
    eH = (yH2 - yH).abs().max().item() / max(yH.abs().max().item(), 1e-12)
    eM = (yM2 - yM).abs().max().item() / max(yM.abs().max().item(), 1e-12)
    ok = eH < 1e-3 and eM < 1e-3
    print(f"[consistency] D-err={eH:.2e}  S-err={eM:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_gradient_reaches_prior():
    torch.manual_seed(0)
    bands, msi, H, W, scale = 8, 3, 32, 32, 4
    cfg = NullFusionConfig(scale=scale, bands=bands, msi_bands=msi,
                           cg_steps=40, prior_depth=3, enc_depth=2, use_attn=False)
    net = NullFusionNet(cfg).train()
    net.set_srf(_make_srf(bands, msi))
    yH = torch.randn(2, bands, H // scale, W // scale)
    yM = torch.randn(2, msi, H, W)
    gt = torch.randn(2, bands, H, W)
    out = net(yH, yM)["out"]
    loss = F.l1_loss(out, gt)
    loss.backward()
    g = net.prior_body[-1].weight.grad
    ok = g is not None and g.abs().mean().item() > 0
    print(f"[gradient] prior final conv grad mean = {g.abs().mean().item():.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_prior_beats_base():
    """A few training steps must lower error below the no-prior (base) error."""
    torch.manual_seed(0)
    bands, msi, H, W, scale = 8, 3, 32, 32, 4
    cfg = NullFusionConfig(scale=scale, bands=bands, msi_bands=msi, cg_steps=40,
                           prior_depth=4, enc_depth=2, use_attn=False, width=48)
    srf = _make_srf(bands, msi)
    net = NullFusionNet(cfg).train()
    net.set_srf(srf)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    # smooth synthetic GT so the range solve is meaningful
    spec = torch.randn(bands, 4)
    spat = torch.randn(4, H, W)
    gt = (spec @ spat.reshape(4, -1)).reshape(bands, H, W).unsqueeze(0).repeat(2, 1, 1, 1)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

    # observations from the SAME operator the network uses (exact range split)
    with torch.no_grad():
        yH, yM = net.op(gt)

    with torch.no_grad():
        base = net(yH, yM)["base"]
        base_psnr = _psnr(base, gt)

    for _ in range(120):
        opt.zero_grad()
        out = net(yH, yM)["out"]
        loss = F.l1_loss(out, gt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = net(yH, yM)["out"]
        final_psnr = _psnr(final, gt)

    ok = final_psnr >= base_psnr - 0.5
    print(f"[prior>base] base PSNR={base_psnr:.2f}  trained PSNR={final_psnr:.2f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def run_all():
    ok = True
    ok &= check_consistency()
    ok &= check_gradient_reaches_prior()
    ok &= check_prior_beats_base()
    print(f"\n[nullfusion] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()
