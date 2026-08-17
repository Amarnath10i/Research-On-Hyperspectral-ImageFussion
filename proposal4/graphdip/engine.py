"""Per-scene DIP engine for GraphDIP.

There is no global training set: every scene is fused by optimising a fresh GNN
against the physics-only objective (deep image prior).  ``train`` runs this per
scene and aggregates protocol metrics.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proposal1.daetf.data import estimate_srf
from proposal1.daetf.degrade import FixedDegradation
from proposal1.daetf.metrics import evaluate_arrays

from .config import Config
from .model import GraphDIP
from .superpixels import kmeans_superpixels


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_model(cfg: Config, device: torch.device,
                srf: np.ndarray) -> GraphDIP:
    model = GraphDIP(cfg).to(device)
    model.set_srf(torch.from_numpy(srf.astype(np.float32)))
    return model


def dip_optimize(model: GraphDIP, feats: torch.Tensor, labels: torch.Tensor,
                 lr_hs: torch.Tensor, msi: torch.Tensor,
                 kernel: torch.Tensor, steps: int, lr: float,
                 tv_weight: float = 0.0, device=None) -> tuple:
    """Per-scene deep-image-prior optimisation (physics-only objective)."""
    nb = model.neighbors(feats, model.cfg.graph_k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        pred = model(feats, labels, nb)
        loss = model.physics_objective(pred["out"], lr_hs, msi, kernel)
        if tv_weight > 0:
            loss = loss + tv_weight * model.tv(pred["out"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def _scene_tensors(cfg: Config, gt: np.ndarray, srf: np.ndarray,
                   device: torch.device):
    H, W = gt.shape[:2]
    gt_t = torch.from_numpy(gt[..., :cfg.bands]).permute(2, 0, 1).float()
    degrade = FixedDegradation(cfg.blur_ksize, cfg.scale, cfg.eval_sigma,
                               cfg.eval_sigma, 0.0, 0.0, 0.0)
    lr = degrade(gt_t.unsqueeze(0)).squeeze(0)
    msi = torch.einsum("chw,cm->mhw", gt_t, torch.from_numpy(srf).float())
    return (gt_t.unsqueeze(0).to(device), lr.unsqueeze(0).to(device),
            msi.unsqueeze(0).to(device))


def train(cfg: Config, device: Optional[str] = None) -> dict:
    cfg.resolve()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.seed)

    from proposal1.daetf.io_utils import find_pairs
    pairs = find_pairs(cfg.source_root, cfg.target_root)[:cfg.val_scenes]
    srf = estimate_srf(cfg.source_root, cfg.bands, cfg.msi_bands)

    agg = {"psnr": [], "ssim": [], "sam": [], "ergas": []}
    t0 = time.time()
    for src, tgt in pairs:
        gt = np.load(tgt)["arr_0"]
        gt_t, lr, msi = _scene_tensors(cfg, gt, srf, device)
        labels, centres = kmeans_superpixels(
            msi.squeeze(0), cfg.n_seg, cfg.spatial_weight, seed=cfg.seed)
        feats = centres.to(device)
        model = _make_model(cfg, device, srf)
        kernel = model.default_kernel
        model, losses = dip_optimize(model, feats, labels.to(device), lr, msi,
                                     kernel, cfg.dip_steps, cfg.dip_lr,
                                     cfg.tv_weight if cfg.use_tv else 0.0,
                                     device)
        with torch.no_grad():
            pred = model(feats, labels.to(device), model.neighbors(feats, cfg.graph_k))
        fused = pred["out"][0].cpu().numpy()
        fused = fused[:, :gt.shape[0], :gt.shape[1]]
        met = evaluate_arrays(np.transpose(fused, (1, 2, 0)),
                              gt[..., :cfg.bands], cfg.scale)
        for k in agg:
            agg[k].append(met[k])
        print(f"[dip] {os.path.basename(tgt)} loss {losses[0]:.4f}->{losses[-1]:.4f} "
              f"sam={met['sam']:.4f} ({time.time() - t0:.0f}s)")
    res = {k: float(np.mean(v)) for k, v in agg.items()}
    print(f"[done] {res}")
    return {"per_scene": agg, "mean": res}


def evaluate_dataset(model: GraphDIP, cfg: Config, device: torch.device,
                     feats=None, labels=None, lr=None, msi=None,
                     kernel=None) -> dict:
    """Score one already-optimised scene (see dip_optimize / train)."""
    model.eval()
    pred = model(feats, labels, model.neighbors(feats, cfg.graph_k))
    model.train()
    return {"phys": model.physics_objective(pred["out"], lr, msi, kernel).item()}