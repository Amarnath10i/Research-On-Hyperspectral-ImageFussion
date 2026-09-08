"""Self-checks for GraphDIP (CPU-fast, no data required).

Verifies the structural claims of Proposal 4:
  [1] k-means superpixels produce a valid full segmentation.
  [2] The 'linear' mixing stage is genuinely a linear map; 'nonlinear' and
      'attention' stages are not (message passing = non-linear mixing).
  [3] Per-scene physics-only DIP reduces the physics objective well below its
      initial value (self-supervised fusion works without any ground truth).
  [4] Ladder stages build, forward and backprop.

Returns True iff all checks pass.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from .config import Config
from .model import GraphDIP
from .superpixels import kmeans_superpixels


def check_superpixels() -> bool:
    torch.manual_seed(0)
    msi = torch.rand(3, 24, 24)
    n_seg = 8
    labels, centres = kmeans_superpixels(msi, n_seg, seed=0)
    n = int(labels.max().item()) + 1
    counts = torch.bincount(labels.reshape(-1), minlength=n_seg)
    ok = (labels.shape == (24, 24) and n == n_seg
          and bool((counts > 0).all().item())
          and centres.shape == (n_seg, 5))
    print(f"[1] superpixels n_seg={n_seg} labels={tuple(labels.shape)} "
          f"centres={tuple(centres.shape)} non_empty={bool((counts > 0).all().item())} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _model(cfg: Config, seed: int = 1) -> GraphDIP:
    torch.manual_seed(seed)
    m = GraphDIP(cfg)
    m.set_srf(torch.rand(cfg.bands, cfg.msi_bands) + 0.1)
    return m


def check_mixing_linearity() -> bool:
    cfg = Config(bands=6, msi_bands=3, scale=4, patch=24, n_seg=12, graph_k=4,
                 hidden=16, n_layers=2, source_root="", target_root="")
    labels = torch.randint(0, 12, (24, 24))
    x = torch.rand(12, 5)
    t = 2.3

    lin = _model(dataclasses.replace(cfg, mix_type="linear"), 2)
    nb = lin.neighbors(x, cfg.graph_k)
    o1 = lin(x, labels, nb)["out"]
    o2 = lin(t * x, labels, nb)["out"]
    denom = (t * o1).abs().max().item()
    lin_err = float((o2 - t * o1).abs().max().item() / denom)
    lin_ok = lin_err < 1e-4
    print(f"[2] linear mixing is linear  rel_err={lin_err:.2e} "
          f"{'PASS' if lin_ok else 'FAIL'}")

    nonlinear_viols = {}
    for name in ("nonlinear", "attention"):
        m = _model(dataclasses.replace(cfg, mix_type=name), 3)
        nb = m.neighbors(x, cfg.graph_k)
        a1 = m(x, labels, nb)["out"]
        a2 = m(t * x, labels, nb)["out"]
        den = (t * a1).abs().max().item() + 1e-9
        err = float((a2 - t * a1).abs().max().item() / den)
        nonlinear_viols[name] = err
        print(f"[2] {name} mixing is non-linear  violation={err:.3f} "
              f"{'PASS' if err > 0.02 else 'FAIL'}")
    return lin_ok and all(v > 0.02 for v in nonlinear_viols.values())


def check_physics_dip() -> bool:
    torch.manual_seed(4)
    C, M, H, W, n_seg = 6, 3, 24, 24, 8
    cfg = Config(bands=C, msi_bands=M, scale=4, patch=H, blur_ksize=5,
                 eval_sigma=1.2, n_seg=n_seg, graph_k=4, hidden=24, n_layers=2,
                 mix_type="attention", dip_steps=400, dip_lr=1e-2,
                 source_root="", target_root="")
    msi = torch.rand(M, H, W)
    labels, centres = kmeans_superpixels(msi, n_seg, seed=0)
    node_true = torch.rand(n_seg, C) * 0.8 + 0.1
    gt = node_true[labels].permute(2, 0, 1).unsqueeze(0)      # (1,C,H,W)

    model = _model(cfg, 5)
    kernel = model.default_kernel
    x = model.op.D(gt, kernel)
    m = model.op.S(gt, model.srf)
    feats = centres

    from .engine import dip_optimize
    model, losses = dip_optimize(model, feats, labels, x, m, kernel,
                                 cfg.dip_steps, cfg.dip_lr, 0.0)
    init, final = losses[0], losses[-1]
    ratio = final / max(init, 1e-12)
    with torch.no_grad():
        pred = model(feats, labels, model.neighbors(feats, cfg.graph_k))
    node_corr = torch.corrcoef(
        torch.stack([pred["nodes"].reshape(-1), node_true.reshape(-1)]))[0, 1].item()
    ok = ratio < 0.3
    print(f"[3] physics-only DIP loss {init:.4f} -> {final:.4f} "
          f"ratio={ratio:.3f} node_corr={node_corr:.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_ladder_smoke() -> bool:
    torch.manual_seed(6)
    ok = True
    for name in ("linear", "nonlinear", "attention"):
        cfg = Config(bands=6, msi_bands=3, scale=4, patch=16, n_seg=8,
                     graph_k=3, hidden=12, n_layers=1, mix_type=name,
                     source_root="", target_root="")
        m = _model(cfg, 7)
        labels = torch.randint(0, 8, (16, 16))
        feats = torch.rand(8, 5)
        pred = m(feats, labels)
        pred["out"].sum().backward()
        g = all(p.grad is not None for p in m.parameters() if p.requires_grad)
        ok &= g
        print(f"[4] mix_type={name}: fwd/bwd {'PASS' if g else 'FAIL'}")
    return ok


def run_all(device: str = "cpu") -> bool:
    torch.manual_seed(0)
    checks = [
        ("superpixels", check_superpixels),
        ("mixing linearity", check_mixing_linearity),
        ("physics DIP", check_physics_dip),
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