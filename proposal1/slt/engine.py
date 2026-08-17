"""Training, full-scene inference and evaluation for SLT."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .losses import SLTLoss
from .model import SLTNet
from proposal1.daetf.data import FusionPatchDataset, estimate_srf
from proposal1.daetf.degrade import FixedDegradation
from proposal1.daetf.io_utils import find_pairs
from proposal1.daetf.metrics import evaluate_arrays


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup:
        return cfg.lr * step / max(cfg.warmup, 1)
    t_total = max(cfg.iters - cfg.warmup, 1)
    n_restarts = getattr(cfg, "n_restarts", 1)
    t_i = t_total // max(n_restarts, 1)
    t = (step - cfg.warmup) % max(t_i, 1) / max(t_i, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def tiled_inference(model: SLTNet, lr: torch.Tensor, msi: torch.Tensor,
                    scale: int, tile_hr: int = 256, overlap: int = 32) -> torch.Tensor:
    """Hann-weighted overlapping tiles so full scenes fit in memory."""
    model.eval()
    bsz, _, h_hr, w_hr = msi.shape
    tile_lr = min(tile_hr // scale, lr.shape[2], lr.shape[3])
    tile_hr = tile_lr * scale
    step_lr = max(tile_lr - overlap // scale, 1)
    out = torch.zeros(bsz, model.cfg.bands, h_hr, w_hr, device=lr.device,
                      dtype=torch.float32)
    wsum = torch.zeros(bsz, 1, h_hr, w_hr, device=lr.device, dtype=torch.float32)
    win1d = torch.hann_window(tile_hr, periodic=False, device=lr.device).clamp_min(1e-3)
    win = (win1d[:, None] * win1d[None, :])[None, None]

    ys = list(range(0, max(lr.shape[2] - tile_lr, 0) + 1, step_lr))
    xs = list(range(0, max(lr.shape[3] - tile_lr, 0) + 1, step_lr))
    if ys[-1] + tile_lr < lr.shape[2]:
        ys.append(lr.shape[2] - tile_lr)
    if xs[-1] + tile_lr < lr.shape[3]:
        xs.append(lr.shape[3] - tile_lr)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + tile_lr, x0 + tile_lr
            hy0, hx0 = y0 * scale, x0 * scale
            pred = model(lr[:, :, y0:y1, x0:x1],
                         msi[:, :, hy0:hy0 + tile_hr, hx0:hx0 + tile_hr])["out"].float()
            w = win[..., :pred.shape[-2], :pred.shape[-1]]
            out[:, :, hy0:hy0 + tile_hr, hx0:hx0 + tile_hr] += pred * w
            wsum[:, :, hy0:hy0 + tile_hr, hx0:hx0 + tile_hr] += w
    return (out / wsum.clamp_min(1e-6)).clamp(0, 1)


@torch.no_grad()
def evaluate_dataset(model: SLTNet, root: str, cfg: Config, split: str = "Test",
                     device: str = "cuda", limit: Optional[int] = None,
                     tile_hr: int = 256, verbose: bool = True,
                     return_rows: bool = False,
                     srf: Optional[np.ndarray] = None):
    """Full-scene evaluation through the unified metric module.

    Every scene also reports ``lr_consistency`` (max|D(ŷ) - X| / max|X| under
    the fixed evaluation operator), per the protocol audit.
    """
    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    from proposal1.daetf.data import SceneCache
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation.from_config(cfg).to(device)
    if srf is None:
        srf = estimate_srf(root, split, cfg)
    srf_w = torch.from_numpy(srf.T.reshape(cfg.msi_bands, cfg.bands, 1, 1)
                             .astype(np.float32)).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": [],
                     "lr_consistency": []}

    for stem, hp, rp in pairs:
        hsi, rgb = cache.get(stem, hp, rp)
        h = (hsi.shape[1] // cfg.scale) * cfg.scale
        w = (hsi.shape[2] // cfg.scale) * cfg.scale
        gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(device)
        msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(device)
        lr = degrade(gt)
        pred = tiled_inference(model, lr, msi, cfg.scale, tile_hr=tile_hr)
        m = evaluate_arrays(pred[0].cpu().numpy().transpose(1, 2, 0),
                            gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
        re = degrade(pred.float())
        m["lr_consistency"] = float(
            ((re - lr).abs().max() / lr.abs().max().clamp_min(1e-12)).item())
        rows.append((stem, m))
        for k, v in m.items():
            agg[k].append(v)
        if verbose:
            print(f"  {stem:<24} PSNR={m['psnr']:7.3f}  SSIM={m['ssim']:.4f}  "
                  f"SAM={m['sam']:6.3f}  ERGAS={m['ergas']:8.3f}  "
                  f"LRcons={m['lr_consistency']:.2e}")
        del gt, msi, lr, pred
        if device == "cuda":
            torch.cuda.empty_cache()

    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    if verbose:
        print(f"  {'MEAN':<24} PSNR={mean['psnr']:7.3f}  SSIM={mean['ssim']:.4f}  "
              f"SAM={mean['sam']:6.3f}  ERGAS={mean['ergas']:8.3f}  "
              f"LRcons={mean['lr_consistency']:.2e}")
    if return_rows:
        return mean, [{"scene": s, **m} for s, m in rows]
    return mean


def train(cfg: Config, device: str = "cuda", log_fn=print) -> Tuple[SLTNet, Dict]:
    cfg.resolve(verbose=False)
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    log_fn("estimating SRF from the training pairs ...")
    srf = estimate_srf(cfg.source_root, "Train", cfg)

    train_set = FusionPatchDataset(cfg.source_root, "Train", cfg, train=True,
                                   srf=srf, length=cfg.iters * cfg.batch)
    loader = DataLoader(train_set, batch_size=cfg.batch, shuffle=False,
                        num_workers=cfg.workers, pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=cfg.workers > 0)

    model = SLTNet(cfg).to(device)
    crit = SLTLoss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5,
                            betas=(0.9, 0.99))
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_fn(f"SLT ({cfg.name}): {model.n_params() / 1e6:.2f} M parameters, "
           f"manifold={cfg.manifold} geodesic={cfg.geodesic} "
           f"msi_guide={cfg.use_msi_guide}")

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    def update_ema(m, ema_m, decay):
        with torch.no_grad():
            for p, ema_p in zip(m.parameters(), ema_m.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    history: Dict = {"iter": [], "loss": [], "val": [], "cfg": cfg.to_dict()}
    best, t0 = -1e9, time.time()

    ckpt_path = os.path.join(cfg.out_dir, f"{cfg.name}_checkpoint.pth")
    start_step = 1
    if os.path.exists(ckpt_path):
        log_fn(f"Resuming from {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck["ema_model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"] + 1
        best = ck.get("best", -1e9)
        history = ck.get("history", history)

    model.train()
    for step, batch in enumerate(loader, start=start_step):
        if step > cfg.iters:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg)

        lr_hsi = batch["lr"].to(device, non_blocking=True)
        msi = batch["msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        if step % cfg.grad_accum == 1 or cfg.grad_accum == 1:
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(lr_hsi, msi)
            loss, logs = crit(out, gt)
        scaler.scale(loss / cfg.grad_accum).backward()

        if step % cfg.grad_accum == 0 or step == cfg.iters:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            update_ema(model, ema_model, cfg.ema_decay)

        if step % cfg.log_every == 0:
            rate = step / (time.time() - t0)
            eta = (cfg.iters - step) / max(rate, 1e-6) / 60
            log_fn(f"it {step:6d}/{cfg.iters}  loss {logs['total']:.4f}  "
                   f"{' '.join(f'{k} {v:.4f}' for k, v in logs.items())}  "
                   f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.2f} it/s  "
                   f"eta {eta:.0f}m")
            history["iter"].append(step)
            history["loss"].append(logs["total"])
            torch.save({"model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "best": best, "history": history},
                       ckpt_path)

        if step % cfg.val_every == 0 or step == cfg.iters:
            m = evaluate_dataset(ema_model, cfg.source_root, cfg, "Test",
                                 device, limit=cfg.val_scenes, verbose=False)
            log_fn(f"  [val@{step}] PSNR {m['psnr']:.3f}  SAM {m['sam']:.3f}  "
                   f"ERGAS {m['ergas']:.3f}")
            history["val"].append({"iter": step, **m})
            if m["psnr"] > best:
                best = m["psnr"]
                torch.save({"model": ema_model.state_dict(),
                            "cfg": cfg.to_dict(), "srf": srf, "val": m},
                           os.path.join(cfg.out_dir, f"{cfg.name}_best.pth"))
            model.train()

    torch.save({"model": ema_model.state_dict(), "cfg": cfg.to_dict(),
                "srf": srf}, os.path.join(cfg.out_dir, f"{cfg.name}_final.pth"))
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return model, history


def load_checkpoint(path: str, device: str = "cuda") -> Tuple[SLTNet, Config, np.ndarray]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    model = SLTNet(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["srf"]