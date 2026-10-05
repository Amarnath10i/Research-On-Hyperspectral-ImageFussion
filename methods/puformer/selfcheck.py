"""CPU self-checks for the operators and metrics (about 1 minute).

    python selfcheck.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from hsifuse.metrics import _onion_table, evaluate, onions_quality, q2n, scc
from hsifuse.models import build
from hsifuse.ops import Degradation, LearnedDegradation, wv2_srf


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        raise SystemExit(1)


def main():
    torch.manual_seed(0)
    x = torch.rand(2, 128, 64, 48, dtype=torch.float64)
    y = torch.rand(2, 128, 16, 12, dtype=torch.float64)
    for pad in ("zeros", "reflect"):
        deg = Degradation(wv2_srf(), pad=pad).double()
        lhs, rhs = (deg.D(x) * y).sum(), (x * deg.Dt(y)).sum()
        check(f"D/Dt adjoint ({pad})", abs(lhs - rhs) / abs(lhs) < 1e-13, f"rel err {abs(lhs - rhs) / abs(lhs):.1e}")
        xr = x.clone().requires_grad_(True)
        (g,) = torch.autograd.grad((deg.D(xr) * y).sum(), xr)
        check(f"Dt == autograd VJP of D ({pad})", torch.allclose(g, deg.Dt(y), atol=1e-12))
        m = torch.rand(2, 8, 64, 48, dtype=torch.float64)
        lhs, rhs = (deg.R(x) * m).sum(), (x * deg.Rt(m)).sum()
        check(f"R/Rt adjoint ({pad})", abs(lhs - rhs) / abs(lhs) < 1e-13)

    deg = Degradation(wv2_srf()).double()
    old_d = deg.blur(x)[..., ::4, ::4]
    up = y.new_zeros(2, 128, 64, 48); up[..., ::4, ::4] = y
    old_dt = deg.blur(up)
    check("strided D == blur + decimate (zeros)", torch.allclose(deg.D(x), old_d, atol=1e-14))
    check("strided Dt == zero-insert + blur (zeros)", torch.allclose(deg.Dt(y), old_dt, atol=1e-14))
    deg_r = Degradation(wv2_srf(), pad="reflect").double()
    check("reflect D == simulation operator", torch.allclose(deg_r.D(x), deg.simulate(x)[0], atol=1e-14))
    check("reflect D == blur(reflect) + decimate", torch.allclose(deg_r.D(x), deg.blur(x, "reflect")[..., ::4, ::4]))

    # Q2n: table-based fast path vs literal onion_mult2D port
    for n in (2, 4, 8, 16, 128):
        idx, sign = _onion_table(n)
        a, b = torch.rand(3, 64, n, dtype=torch.float64), torch.rand(3, 64, n, dtype=torch.float64)
        fast, slow = onions_quality(a, b, fast=True), onions_quality(a, b, fast=False)
        check(f"onions_quality fast == literal (n={n})", torch.allclose(fast, slow, rtol=1e-10, atol=1e-12),
              f"max diff {float((fast - slow).abs().max()):.1e}")
    gt = torch.rand(128, 64, 64, dtype=torch.float64) * 0.5 + 0.1
    check("Q2n(x, x) == 1", abs(q2n(gt, gt) - 1) < 1e-9, f"{q2n(gt, gt):.12f}")
    check("SCC(x, x) == 1", abs(scc(gt, gt) - 1) < 1e-12)
    noisy = (gt + 0.01 * torch.randn_like(gt)).clamp(0, 1)
    check("Q2n fast == literal on an image", abs(q2n(gt, noisy) - q2n(gt, noisy, fast=False)) < 1e-10,
          f"{q2n(gt, noisy):.6f}")
    check("Q2n non-multiple of the block size runs", 0 < q2n(gt[:, :50, :70], noisy[:, :50, :70]) < 1)
    m = evaluate(gt[None], noisy[None], full=True)
    print("   metrics on a noisy image:", {k: round(v, 4) for k, v in m.items()})

    # x8 decimation and the learned-operator ablation: exact adjoint pairs as well
    y8 = torch.rand(2, 128, 8, 6, dtype=torch.float64)
    for pad in ("zeros", "reflect"):
        deg = Degradation(wv2_srf(), scale=8, pad=pad).double()
        lhs, rhs = (deg.D(x) * y8).sum(), (x * deg.Dt(y8)).sum()
        check(f"x8 D/Dt adjoint ({pad})", abs(lhs - rhs) / abs(lhs) < 1e-13, f"rel err {abs(lhs - rhs) / abs(lhs):.1e}")
    check("x8 reflect D == blur(reflect) + decimate",
          torch.allclose(Degradation(wv2_srf(), scale=8, pad="reflect").double().D(x),
                         deg.blur(x, "reflect")[..., ::8, ::8]))
    for s, yy in ((4, y), (8, y8)):
        lop = LearnedDegradation(128, 8, scale=s).double()
        with torch.no_grad():
            lop.k_logit.normal_(), lop.r_logit.normal_()
        lhs, rhs = (lop.D(x) * yy).sum(), (x * lop.Dt(yy)).sum()
        m = torch.rand(2, 8, 64, 48, dtype=torch.float64)
        lhs2, rhs2 = (lop.R(x) * m).sum(), (x * lop.Rt(m)).sum()
        check(f"learned D/Dt, R/Rt adjoint (x{s})", abs(lhs - rhs) / abs(lhs) < 1e-13 and abs(lhs2 - rhs2) / abs(lhs2) < 1e-13)

    # PUFormer forward / backward: both border modes, x8, and every ablation switch
    configs = [dict(pad="zeros"), dict(pad="reflect"), dict(scale=8), dict(physics=False), dict(res_input=False),
               dict(memory=False), dict(mixer="window"), dict(mixer="conv"), dict(op="learned")]
    for cfg in configs:
        cfg = dict(cfg)
        s, pad, op = cfg.pop("scale", 4), cfg.pop("pad", "reflect"), cfg.pop("op", "exact")
        deg = Degradation(wv2_srf(), scale=s, pad=pad)
        if op == "learned":
            deg = LearnedDegradation(128, 8, scale=s)
        net = build("puformer", deg, width=16, stages=2, **cfg)
        lr, ms = torch.rand(2, 128, 64 // s, 64 // s), torch.rand(2, 8, 64, 64)
        outs = net(lr, ms, return_all=True)
        sum(o.mean() for o in outs).backward()
        ok = all(p.grad is not None for n, p in net.named_parameters()
                 if p.requires_grad and (net.physics or not n.startswith("eta")))
        check(f"PUFormer forward/backward (x{s}, {pad}, {op}, {cfg})", outs[-1].shape == (2, 128, 64, 64) and ok)
    print("all checks passed")


if __name__ == "__main__":
    main()
