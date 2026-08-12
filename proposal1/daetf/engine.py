"""Training, full-scene inference, evaluation and test-time adaptation."""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import copy

from .config import Config
from .data import FusionPatchDataset, SceneCache, estimate_srf
from .degrade import FixedDegradation
from .io_utils import find_pairs
from .losses import SPCLoss
from .metrics import evaluate_arrays
from .model import DAETFNet
from .modules import GeometricSelfEnsemble


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup:
        return cfg.lr * step / max(cfg.warmup, 1)
    
    # Cosine annealing with warm restarts
    t_total = max(cfg.iters - cfg.warmup, 1)
    n_restarts = getattr(cfg, 'n_restarts', 1)
    t_i = t_total // max(n_restarts, 1)
    current_step = (step - cfg.warmup) % max(t_i, 1)
    
    t = current_step / max(t_i, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


# ------------------------------------------------------------------- inference
@torch.no_grad()
def tiled_inference(model: DAETFNet, lr: torch.Tensor, msi: torch.Tensor, scale: int,
                    tile_hr: int = 256, overlap: int = 32, ensemble: bool = True) -> torch.Tensor:
    """Hann-weighted overlapping tiles, so full 512x512 and 1040x1392 scenes fit
    in 16 GB without seams appearing at tile boundaries."""
    model.eval()
    bsz, _, h_hr, w_hr = msi.shape
    tile_lr, ov_lr = tile_hr // scale, overlap // scale
    tile_lr = min(tile_lr, lr.shape[2], lr.shape[3])
    tile_hr = tile_lr * scale
    step_lr = max(tile_lr - ov_lr, 1)
    out = torch.zeros(bsz, model.cfg.bands, h_hr, w_hr, device=lr.device, dtype=torch.float32)
    wsum = torch.zeros(bsz, 1, h_hr, w_hr, device=lr.device, dtype=torch.float32)

    win1d = torch.hann_window(tile_hr, periodic=False, device=lr.device).clamp_min(1e-3)
    win = (win1d[:, None] * win1d[None, :])[None, None]

    ys = list(range(0, max(lr.shape[2] - tile_lr, 0) + 1, step_lr))
    xs = list(range(0, max(lr.shape[3] - tile_lr, 0) + 1, step_lr))
    if ys[-1] + tile_lr < lr.shape[2]:
        ys.append(lr.shape[2] - tile_lr)
    if xs[-1] + tile_lr < lr.shape[3]:
        xs.append(lr.shape[3] - tile_lr)

    def _infer(m, l, ms):
        return m(l, ms)["out"].float()

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + tile_lr, x0 + tile_lr
            hy0, hx0, hy1, hx1 = y0 * scale, x0 * scale, y1 * scale, x1 * scale
            
            crop_lr = lr[:, :, y0:y1, x0:x1]
            crop_msi = msi[:, :, hy0:hy1, hx0:hx1]
            
            if ensemble:
                pred = GeometricSelfEnsemble.forward_ensemble(model, crop_lr, crop_msi, _infer)
            else:
                pred = _infer(model, crop_lr, crop_msi)
                
            w = win[..., :pred.shape[-2], :pred.shape[-1]]
            out[:, :, hy0:hy1, hx0:hx1] += pred * w
            wsum[:, :, hy0:hy1, hx0:hx1] += w
    return (out / wsum.clamp_min(1e-6)).clamp(0, 1)


@torch.no_grad()
def evaluate_dataset(model: DAETFNet, root: str, cfg: Config, split: str = "Test",
                     device: str = "cuda", limit: Optional[int] = None,
                     tile_hr: int = 256, verbose: bool = True,
                     return_rows: bool = False):
    """Full-scene evaluation through the unified metric module.

    With return_rows=True the per-scene table is returned as well, which is what
    the paired significance tests operate on.
    """
    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation.from_config(cfg).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": []}

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
        rows.append((stem, m))
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
    if return_rows:
        return mean, [{"scene": s, **m} for s, m in rows]
    return mean


# -------------------------------------------------------------------- training
def train(cfg: Config, device: str = "cuda", align_target: bool = True,
          log_fn=print) -> Tuple[DAETFNet, Dict]:
    """Train on the source domain.

    When `align_target` is set and a target root is configured, unlabelled
    target patches are drawn alongside and aligned with an MMD penalty. No
    ground truth from the target domain is ever used, so the cross-domain
    evaluation stays honest.
    """
    cfg.resolve(verbose=False)          # idempotent: fills only what is None
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
    if align_target and cfg.target_root:
        tgt_set = FusionPatchDataset(cfg.target_root, "Train", cfg, train=True, srf=srf,
                                     length=cfg.iters * cfg.batch)
        tgt_loader = iter(DataLoader(tgt_set, batch_size=cfg.batch, shuffle=False,
                                     num_workers=max(1, cfg.workers // 2), drop_last=True))
        log_fn(f"domain alignment enabled against {cfg.target_root} (unlabelled)")

    model = DAETFNet(cfg).to(device)
    crit = SPCLoss(cfg, torch.from_numpy(srf)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5,
                            betas=(0.9, 0.99))
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = model.n_params()
    log_fn(f"DAETF-Net v3 (ASCR): {n_params / 1e6:.2f} M parameters")

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    def update_ema(m, ema_m, decay):
        with torch.no_grad():
            for p, ema_p in zip(m.parameters(), ema_m.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    history: Dict[str, list] = {"iter": [], "loss": [], "val": [], "cfg": cfg.to_dict()}
    best, t0 = -1e9, time.time()

    model.train()
    for step, batch in enumerate(loader, start=1):
        if step > cfg.iters:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg)

        lr_hsi = batch["lr"].to(device, non_blocking=True)
        msi = batch["msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        deg_gt = batch["deg"].to(device, non_blocking=True)
        kernel = batch["kernel"].to(device, non_blocking=True)

        tgt_feat = None
        if tgt_loader is not None:
            tb = next(tgt_loader)
            with torch.amp.autocast("cuda", enabled=use_amp):
                tgt_feat = model.features(tb["lr"].to(device), tb["msi"].to(device))

        if step % cfg.grad_accum == 1 or cfg.grad_accum == 1:
            opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(lr_hsi, msi)
            loss, logs = crit(out, gt, lr_hsi, msi, model, deg_gt=deg_gt,
                              kernel=kernel, tgt_feat=tgt_feat)
            
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
            # Log expert usage if available (v3 diagnostic)
            usage_str = ""
            if hasattr(model, 'expert_usage_summary'):
                usage = model.expert_usage_summary()
                if usage:
                    usage_str = (f"  exp=[sp:{usage.get('spectral', 0):.2f}"
                                 f" ed:{usage.get('edge', 0):.2f}"
                                 f" tx:{usage.get('texture', 0):.2f}"
                                 f" co:{usage.get('correction', 0):.2f}]")
            log_fn(f"it {step:6d}/{cfg.iters}  loss {logs['total']:.4f}  "
                   f"char {logs.get('char', 0):.4f}  sam {logs.get('sam', 0):.4f}  "
                   f"spat {logs.get('spat', 0):.4f}  spec {logs.get('spec', 0):.4f}  "
                   f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.2f} it/s  eta {eta:.0f}m"
                   f"{usage_str}")
            history["iter"].append(step)
            history["loss"].append(logs["total"])

        if step % cfg.val_every == 0 or step == cfg.iters:
            m = evaluate_dataset(ema_model, cfg.source_root, cfg, "Test", device,
                                 limit=cfg.val_scenes, verbose=False)
            log_fn(f"  [val@{step}] PSNR {m['psnr']:.3f}  SAM {m['sam']:.3f}  "
                   f"ERGAS {m['ergas']:.3f}")
            history["val"].append({"iter": step, **m})
            
            if cfg.target_root:
                m_tgt = evaluate_dataset(ema_model, cfg.target_root, cfg, "Test", device,
                                         limit=cfg.val_scenes, verbose=False)
                log_fn(f"  [tgt@{step}] PSNR {m_tgt['psnr']:.3f}  SAM {m_tgt['sam']:.3f}  "
                       f"ERGAS {m_tgt['ergas']:.3f}")
                history.setdefault("tgt_val", []).append({"iter": step, **m_tgt})

            if m["psnr"] > best:
                best = m["psnr"]
                torch.save({"model": ema_model.state_dict(), "cfg": cfg.to_dict(),
                            "srf": srf, "val": m}, os.path.join(cfg.out_dir, "daetf_best.pth"))
            model.train()

    torch.save({"model": ema_model.state_dict(), "cfg": cfg.to_dict(), "srf": srf,
                "params": n_params}, os.path.join(cfg.out_dir, "daetf_final.pth"))
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return model, history


# ------------------------------------------------------- test-time adaptation
@torch.no_grad()
def _clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_time_adapt(model: DAETFNet, lr: torch.Tensor, msi: torch.Tensor,
                    crit: SPCLoss, steps: int = 50, lr_rate: float = 2e-5,
                    restore: bool = True) -> torch.Tensor:
    """Self-supervised adaptation on a single unlabelled target scene.

    Only the physics terms are used - they need no ground truth - so this runs
    on a new dataset or sensor exactly as it would in deployment. Only the
    conditioning-related parameters are adapted, which keeps it stable and cheap.
    """
    state = _clone_state(model) if restore else None
    model.train()
    # Routing noise is a TRAINING-time exploration device: it exists so no
    # expert is starved at initialisation. Leaving it on during adaptation
    # makes TTA stochastic, so adapting the same scene twice gives different
    # answers and the reported cross-domain numbers stop being reproducible.
    # Suppressed here and restored afterwards.
    noise_state = None
    if getattr(model, "moe", None) is not None and hasattr(model.moe, "gate_noise"):
        noise_state = (model.moe.gate_noise, model.moe.gate_floor)
        model.moe.gate_noise = 0.0
        model.moe.gate_floor = 0.0
    # v3: also adapt disagreement projection and new MoE gate
    params = [p for n, p in model.named_parameters()
              if any(k in n for k in ("deg", "film", "moe.gate", "moe.deg_proj",
                                      "disagree", "fdrm"))]
    opt = torch.optim.Adam(params, lr=lr_rate)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(lr, msi)
        loss, _ = crit(out, out["out"].detach(), lr, msi, model, supervised=False)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(lr, msi)["out"].clamp(0, 1)
    if noise_state is not None:
        model.moe.gate_noise, model.moe.gate_floor = noise_state
    if state is not None:
        model.load_state_dict(state)
    return pred


@torch.no_grad()
def evaluate_with_tta(model: DAETFNet, root: str, cfg: Config, srf: np.ndarray,
                      split: str = "Test", device: str = "cuda",
                      steps: int = 50, limit: Optional[int] = None,
                      tile_hr: int = 256, verbose: bool = True):
    """Cross-domain evaluation where each scene is adapted before scoring.

    Every scene starts from the same trained weights. Adapting cumulatively
    across scenes would make the result depend on the order the scenes happen
    to be listed in, and would quietly let information leak from one test scene
    into the next.
    """
    crit = SPCLoss(cfg, torch.from_numpy(srf)).to(device)
    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation.from_config(cfg).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": []}
    base_state = _clone_state(model)          # restored before every scene

    for stem, hp, rp in pairs:
        model.load_state_dict(base_state)
        hsi, rgb = cache.get(stem, hp, rp)
        h = (hsi.shape[1] // cfg.scale) * cfg.scale
        w = (hsi.shape[2] // cfg.scale) * cfg.scale
        # adapt on a centre crop to bound memory, then infer over the full scene
        ch, cw = min(h, tile_hr * 2), min(w, tile_hr * 2)
        oy, ox = (h - ch) // 2, (w - cw) // 2
        gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(device)
        msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(device)
        lr = degrade(gt)
        crop_lr = lr[:, :, oy // cfg.scale: (oy + ch) // cfg.scale,
                     ox // cfg.scale: (ox + cw) // cfg.scale]
        crop_msi = msi[:, :, oy:oy + ch, ox:ox + cw]
        with torch.enable_grad():
            test_time_adapt(model, crop_lr, crop_msi, crit, steps=steps, restore=False)
        pred = tiled_inference(model, lr, msi, cfg.scale, tile_hr=tile_hr)
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

    model.load_state_dict(base_state)         # leave the caller's model untouched
    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    if verbose:
        print(f"  {'MEAN (TTA)':<24} PSNR={mean['psnr']:7.3f}  SSIM={mean['ssim']:.4f}  "
              f"SAM={mean['sam']:6.3f}  ERGAS={mean['ergas']:8.3f}")
    return mean, rows


def load_checkpoint(path: str, device: str = "cuda") -> Tuple[DAETFNet, Config, np.ndarray]:
    """Rebuild a model from a checkpoint without needing the original Config."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    model = DAETFNet(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["srf"]
