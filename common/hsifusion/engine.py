"""Training, tiled inference, evaluation and test-time adaptation.

Model-agnostic: anything with `forward(lr_hsi, msi) -> {"out": tensor}` and a
`.cfg` attribute can be trained and scored here. That contract is what lets all
four proposals share one protocol, one metric implementation and one set of
result tables.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import FusionPatchDataset, SceneCache, estimate_srf
from .degrade import FixedDegradation
from .io_utils import find_pairs
from .metrics import evaluate_arrays
from .checkpoint import CheckpointManager


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, cfg) -> float:
    if step < cfg.warmup:
        return cfg.lr * step / max(cfg.warmup, 1)
    t = (step - cfg.warmup) / max(cfg.iters - cfg.warmup, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------- inference
@torch.no_grad()
def tiled_inference(model: nn.Module, lr: torch.Tensor, msi: torch.Tensor,
                    scale: int, bands: Optional[int] = None,
                    tile_hr: int = 256, overlap: int = 32) -> torch.Tensor:
    """Hann-weighted overlapping tiles, so full scenes fit in 16 GB and tile
    seams do not appear in the output."""
    model.eval()
    bands = bands or getattr(model, "cfg", None).bands
    bsz, _, h_hr, w_hr = msi.shape
    tile_lr, ov_lr = tile_hr // scale, overlap // scale
    tile_lr = min(tile_lr, lr.shape[2], lr.shape[3])
    tile_hr = tile_lr * scale
    step_lr = max(tile_lr - ov_lr, 1)
    out = torch.zeros(bsz, bands, h_hr, w_hr, device=lr.device, dtype=torch.float32)
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
            hy0, hx0, hy1, hx1 = y0 * scale, x0 * scale, y1 * scale, x1 * scale
            pred = model(lr[:, :, y0:y1, x0:x1],
                         msi[:, :, hy0:hy1, hx0:hx1])["out"].float()
            w = win[..., :pred.shape[-2], :pred.shape[-1]]
            out[:, :, hy0:hy1, hx0:hx1] += pred * w
            wsum[:, :, hy0:hy1, hx0:hx1] += w
    return (out / wsum.clamp_min(1e-6)).clamp(0, 1)


@torch.no_grad()
def evaluate_dataset(model: nn.Module, root: str, cfg, split: str = "Test",
                     device: str = "cuda", limit: Optional[int] = None,
                     tile_hr: int = 256, verbose: bool = True,
                     return_rows: bool = False):
    """Full-scene evaluation through the unified metric module."""
    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation(cfg.scale, cfg.blur_ksize, cfg.eval_sigma).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": []}

    for stem, hp, rp in pairs:
        hsi, rgb = cache.get(stem, hp, rp)
        h = (hsi.shape[1] // cfg.scale) * cfg.scale
        w = (hsi.shape[2] // cfg.scale) * cfg.scale
        gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(device)
        msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(device)
        lr = degrade(gt)
        pred = tiled_inference(model, lr, msi, cfg.scale, cfg.bands, tile_hr=tile_hr)
        m = evaluate_arrays(pred[0].cpu().numpy().transpose(1, 2, 0),
                            gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
        rows.append({"scene": stem, **m})
        for k, v in m.items():
            agg[k].append(v)
        if verbose:
            print(f"  {stem:<24} PSNR={m['psnr']:7.3f}  SSIM={m['ssim']:.4f}  "
                  f"SAM={m['sam']:6.3f}  ERGAS={m['ergas']:8.3f}")
        del gt, msi, lr, pred
        if device == "cuda":
            torch.cuda.empty_cache()

    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    if verbose:
        print(f"  {'MEAN':<24} PSNR={mean['psnr']:7.3f}  SSIM={mean['ssim']:.4f}  "
              f"SAM={mean['sam']:6.3f}  ERGAS={mean['ergas']:8.3f}")
    return (mean, rows) if return_rows else mean


# -------------------------------------------------------------------- training
def train(cfg, build_model: Callable[[object], nn.Module],
          build_loss: Callable[[object, np.ndarray], nn.Module],
          device: str = "cuda", align_target: bool = False,
          feature_fn: Optional[Callable] = None, log_fn=print
          ) -> Tuple[nn.Module, Dict]:
    """Train any model that follows the (lr_hsi, msi) -> {'out': ...} contract.

    `align_target` draws unlabelled target-domain patches alongside and expects
    `feature_fn(model, lr, msi)` to return pooled features for an MMD penalty.
    No target ground truth is ever read, so cross-domain evaluation stays honest.
    """
    cfg.resolve(verbose=False)
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    log_fn("estimating SRF from the training pairs ...")
    srf = estimate_srf(cfg.source_root, "Train", cfg)
    log_fn(f"SRF shape {srf.shape}, column sums {srf.sum(0).round(3).tolist()}")

    train_set = FusionPatchDataset(cfg.source_root, "Train", cfg, train=True, srf=srf,
                                   length=cfg.iters * cfg.batch)
    loader = DataLoader(train_set, batch_size=cfg.batch, shuffle=False,
                        num_workers=cfg.workers, pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=cfg.workers > 0)

    tgt_loader = None
    if align_target and cfg.target_root and feature_fn is not None:
        tgt_set = FusionPatchDataset(cfg.target_root, "Train", cfg, train=True, srf=srf,
                                     length=cfg.iters * cfg.batch)
        tgt_loader = iter(DataLoader(tgt_set, batch_size=cfg.batch, shuffle=False,
                                     num_workers=max(1, cfg.workers // 2), drop_last=True))
        log_fn(f"domain alignment enabled against {cfg.target_root} (unlabelled)")

    model = build_model(cfg).to(device)
    crit = build_loss(cfg, srf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5,
                            betas=(0.9, 0.99))
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_params = n_params(model)
    log_fn(f"{cfg.name}: {total_params / 1e6:.2f} M parameters")

    history: Dict[str, list] = {"iter": [], "loss": [], "val": [],
                                "cfg": cfg.to_dict(), "params": total_params}
    best, t0 = -1e9, time.time()

    ckpt_mgr = CheckpointManager(cfg.out_dir, cfg.name, device=device)
    ckpt_mgr.print_status(log_fn)

    resume_result = ckpt_mgr.maybe_resume(model, opt, scaler, log_fn=log_fn)
    if isinstance(resume_result, tuple):
        start_step, best, history = resume_result
    else:
        start_step = resume_result

    model.train()
    for step, batch in enumerate(loader, start=start_step):
        if step > cfg.iters:
            break

        # Kaggle time guard: save and exit if <10min remaining
        if ckpt_mgr._time_remaining() < 600:
            log_fn(f"\n[KAGGLE] <10min remaining — saving checkpoint at step {step} and exiting")
            ckpt_mgr.save(model, opt, scaler, step, best, history, force=True)
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg)

        lr_hsi = batch["lr"].to(device, non_blocking=True)
        msi = batch["msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        kernel = batch["kernel"].to(device, non_blocking=True)
        extra = {"deg_gt": batch["deg"].to(device, non_blocking=True)}

        if tgt_loader is not None:
            tb = next(tgt_loader)
            with torch.amp.autocast("cuda", enabled=use_amp):
                extra["tgt_feat"] = feature_fn(model, tb["lr"].to(device),
                                               tb["msi"].to(device))

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(lr_hsi, msi)
            loss, logs = crit(out, gt, lr_hsi, msi, model, kernel=kernel, **extra)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % cfg.log_every == 0:
            rate = step / (time.time() - t0)
            eta = (cfg.iters - step) / max(rate, 1e-6) / 60
            log_fn(f"it {step:6d}/{cfg.iters}  loss {logs['total']:.4f}  "
                   f"char {logs.get('char', 0):.4f}  sam {logs.get('sam', 0):.4f}  "
                   f"spat {logs.get('spat', 0):.4f}  spec {logs.get('spec', 0):.4f}  "
                   f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.2f} it/s  eta {eta:.0f}m")
            history["iter"].append(step)
            history["loss"].append(logs["total"])

            ckpt_mgr.save(model, opt, scaler, step, best, history, every=2000)

        if step % cfg.val_every == 0 or step == cfg.iters:
            m = evaluate_dataset(model, cfg.source_root, cfg, "Test", device,
                                 limit=cfg.val_scenes, verbose=False)
            log_fn(f"  [val@{step}] PSNR {m['psnr']:.3f}  SAM {m['sam']:.3f}  "
                   f"ERGAS {m['ergas']:.3f}")
            history["val"].append({"iter": step, **m})
            if m["psnr"] > best:
                best = m["psnr"]
                ckpt_mgr.save_best(model, cfg, srf, m, log_fn=log_fn)
            model.train()

    ckpt_mgr.save_final(model, cfg, srf, total_params, log_fn=log_fn)
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return model, history


# ------------------------------------------------------- test-time adaptation
@torch.no_grad()
def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_time_adapt(model: nn.Module, lr: torch.Tensor, msi: torch.Tensor,
                    crit: nn.Module, steps: int = 30, lr_rate: float = 5e-5,
                    restore: bool = True,
                    param_filter: Tuple[str, ...] = ("deg", "film", "gate", "fdrm")
                    ) -> torch.Tensor:
    """Self-supervised adaptation on one unlabelled scene, using only the
    physics terms. Adapting a subset of parameters keeps it cheap and stable."""
    state = clone_state(model) if restore else None
    model.train()
    params = [p for n, p in model.named_parameters()
              if any(k in n for k in param_filter)]
    if not params:                       # model exposes no conditioning params
        params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr_rate)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(lr, msi)
        loss, _ = crit(out, None, lr, msi, model, supervised=False)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(lr, msi)["out"].clamp(0, 1)
    if state is not None:
        model.load_state_dict(state)
    return pred


@torch.no_grad()
def evaluate_with_tta(model: nn.Module, root: str, cfg, crit: nn.Module,
                      split: str = "Test", device: str = "cuda",
                      steps: int = 30, limit: Optional[int] = None,
                      tile_hr: int = 256, verbose: bool = True):
    """Cross-domain evaluation where each scene is adapted before scoring.

    Every scene restarts from the same trained weights: adapting cumulatively
    would make the result depend on the order the scenes are listed in and let
    information leak between test scenes.
    """
    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation(cfg.scale, cfg.blur_ksize, cfg.eval_sigma).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": []}
    base_state = clone_state(model)

    for stem, hp, rp in pairs:
        model.load_state_dict(base_state)
        hsi, rgb = cache.get(stem, hp, rp)
        h = (hsi.shape[1] // cfg.scale) * cfg.scale
        w = (hsi.shape[2] // cfg.scale) * cfg.scale
        ch, cw = min(h, tile_hr * 2), min(w, tile_hr * 2)
        oy, ox = (h - ch) // 2, (w - cw) // 2
        gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(device)
        msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(device)
        lr = degrade(gt)
        crop_lr = lr[:, :, oy // cfg.scale:(oy + ch) // cfg.scale,
                     ox // cfg.scale:(ox + cw) // cfg.scale]
        crop_msi = msi[:, :, oy:oy + ch, ox:ox + cw]
        with torch.enable_grad():
            test_time_adapt(model, crop_lr, crop_msi, crit, steps=steps, restore=False)
        pred = tiled_inference(model, lr, msi, cfg.scale, cfg.bands, tile_hr=tile_hr)
        m = evaluate_arrays(pred[0].cpu().numpy().transpose(1, 2, 0),
                            gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
        rows.append({"scene": stem, **m})
        for k, v in m.items():
            agg[k].append(v)
        if verbose:
            print(f"  {stem:<24} PSNR={m['psnr']:7.3f}  SSIM={m['ssim']:.4f}  "
                  f"SAM={m['sam']:6.3f}  ERGAS={m['ergas']:8.3f}")
        del gt, msi, lr, pred
        if device == "cuda":
            torch.cuda.empty_cache()

    model.load_state_dict(base_state)
    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    if verbose:
        print(f"  {'MEAN (TTA)':<24} PSNR={mean['psnr']:7.3f}  SSIM={mean['ssim']:.4f}  "
              f"SAM={mean['sam']:6.3f}  ERGAS={mean['ergas']:8.3f}")
    return mean, rows
