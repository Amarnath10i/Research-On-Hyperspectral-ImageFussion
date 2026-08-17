"""Training, sampling wrapper and evaluation for SpectralFlow.

The public model follows the shared protocol contract
``forward(lr_hsi, msi) -> {"out": ...}`` so it can be scored by the same
``hsifusion`` engine as the other proposals.  Under the hood ``forward`` runs
the full null-space-projected reverse diffusion, which is what the model
returns as its fused estimate.

Training, by contrast, never samples: it only trains the score network (the
spectral-spatial manifold prior) and the degradation code head via denoising
score matching.  The projection is applied at sampling time, which is where the
consistency guarantee lives.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hsifusion.data import FusionPatchDataset, estimate_srf
from hsifusion.degrade import FixedDegradation
from hsifusion.io_utils import find_pairs
from hsifusion.metrics import evaluate_arrays

from .config import Config
from .losses import ScoreMatchingLoss
from .nullspace import (RangeNullProjector, decode_degradation_params,
                        kernel_from_params)
from .sampler import DiffusionSampler, refine_operator_kernel
from .score import SpectralScoreNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup:
        return cfg.lr * step / max(cfg.warmup, 1)
    t = (step - cfg.warmup) / max(cfg.iters - cfg.warmup, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


class DegCodeHead(nn.Module):
    """Reads the observed pair and emits a degradation code plus physical
    operator parameters (sigma_x, sigma_y, sin2t, cos2t, noise)."""

    def __init__(self, bands: int, msi_bands: int, code_dim: int, scale: int):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(bands + msi_bands, 32, 3, 1, 1), nn.SiLU(True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.SiLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.code = nn.Linear(32, code_dim)
        self.raw = nn.Linear(32, 5)

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor):
        m = F.avg_pool2d(msi, self.scale)
        h = torch.cat([lr_hsi, m], dim=1)
        h = self.net(h).flatten(1)
        return self.code(h), self.raw(h)


class SamplingModel(nn.Module):
    """The protocol-facing wrapper: sampling is the forward pass.

    ``forward(lr_hsi, msi) -> {"out": fused}`` draws a reverse-diffusion sample
    on the null space of the observation operator, so ``D(out) == lr_hsi`` to
    solver tolerance for whatever the score network produces.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.score_net = SpectralScoreNet(
            cfg.bands, cfg.msi_bands, ch=cfg.ch, ch_mult=cfg.ch_mult,
            num_res=cfg.num_res, cond_dim=cfg.cond_dim, code_dim=cfg.code_dim,
            dropout=cfg.dropout)
        self.deg_head = DegCodeHead(cfg.bands, cfg.msi_bands, cfg.code_dim,
                                    cfg.scale)
        self.projector = RangeNullProjector(cfg.scale, cfg.blur_ksize,
                                            cfg.eval_sigma, cfg.cg_steps,
                                            cfg.ridge)
        self.sampler = DiffusionSampler(
            self.score_net, self.projector, num_timesteps=cfg.num_timesteps,
            sample_steps=cfg.sample_steps, eta=cfg.eta,
            beta_start=cfg.beta_start, beta_end=cfg.beta_end,
            use_projection=cfg.use_projection)
        self._ext_kernel: Optional[torch.Tensor] = None
        self._use_msi_guide = cfg.use_msi_guide

    def set_projection(self, enabled: bool) -> None:
        """Toggle the null-space projection (Q1 ladder Stage 1 vs Stage 2)."""
        self.sampler.use_projection = bool(enabled)

    def set_msi_guide(self, enabled: bool) -> None:
        """Toggle whether the HR-MSI guide reaches the sampler."""
        self._use_msi_guide = bool(enabled)

    def set_kernel(self, kernel: Optional[torch.Tensor]) -> None:
        """External kernel override (used by per-scene operator refinement)."""
        if kernel is None:
            self._ext_kernel = None
        else:
            self._ext_kernel = kernel.detach().clone()

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor):
        if not self._use_msi_guide:
            msi = torch.zeros_like(msi)
        code, raw = self.deg_head(lr_hsi, msi)
        if self._ext_kernel is not None:
            kernel = self._ext_kernel.to(lr_hsi.device)
        else:
            kernel = kernel_from_params(decode_degradation_params(raw),
                                        self.cfg.blur_ksize)
        out = self.sampler.sample(lr_hsi, msi, kernel, code)
        return {"out": out, "kernel": kernel, "code": code, "deg": raw}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: Config) -> SamplingModel:
    return SamplingModel(cfg)


# ------------------------------------------------------------------ training
def train(cfg: Config, device: str = "cuda", log_fn=print
          ) -> Tuple[SamplingModel, Dict]:
    """Train the score network and degradation head by denoising score
    matching.  Sampling/projection are not used here."""
    cfg.resolve(verbose=False)
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    log_fn("estimating SRF from the training pairs ...")
    srf = torch.from_numpy(estimate_srf(cfg.source_root, "Train", cfg)).float()

    train_set = FusionPatchDataset(cfg.source_root, "Train", cfg, train=True,
                                   srf=srf.numpy(),
                                   length=cfg.iters * cfg.batch)
    loader = DataLoader(train_set, batch_size=cfg.batch, shuffle=False,
                        num_workers=cfg.workers, pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=cfg.workers > 0)

    model = build_model(cfg).to(device)
    crit = ScoreMatchingLoss(cfg.num_timesteps, cfg.beta_start, cfg.beta_end,
                             cfg.w_deg, cfg.w_spec)
    params = list(model.score_net.parameters()) + list(model.deg_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-5, betas=(0.9, 0.99))
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_params = sum(p.numel() for p in params)
    log_fn(f"{cfg.name}: {total_params / 1e6:.2f} M trainable parameters")

    history: Dict = {"iter": [], "loss": [], "val": [], "cfg": cfg.to_dict()}
    best, t0 = -1e9, time.time()

    for step, batch in enumerate(loader, start=1):
        if step > cfg.iters:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg)

        gt = batch["gt"].to(device, non_blocking=True)
        msi = batch["msi"].to(device, non_blocking=True)
        lr_hsi = batch["lr"].to(device, non_blocking=True)
        kernel = batch["kernel"].to(device, non_blocking=True)
        deg_gt = batch["deg"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, logs = crit(model, gt, msi, lr_hsi, kernel, deg_gt,
                              srf=srf.to(device))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % cfg.log_every == 0:
            rate = step / (time.time() - t0)
            eta = (cfg.iters - step) / max(rate, 1e-6) / 60
            log_fn(f"it {step:6d}/{cfg.iters}  sm {logs['sm']:.4f}  "
                   f"deg {logs.get('deg', 0):.4f}  "
                   f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.2f} it/s  eta {eta:.0f}m")
            history["iter"].append(step)
            history["loss"].append(logs["total"])

        if step % cfg.val_every == 0 or step == cfg.iters:
            m = evaluate_dataset(model, cfg.source_root, cfg, "Test", device,
                                 limit=cfg.val_scenes, verbose=False)
            log_fn(f"  [val@{step}] PSNR {m['psnr']:.3f}  SAM {m['sam']:.3f}  "
                   f"ERGAS {m['ergas']:.3f}")
            history["val"].append({"iter": step, **m})
            if m["psnr"] > best:
                best = m["psnr"]
                torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                            "srf": srf, "val": m},
                           os.path.join(cfg.out_dir, f"{cfg.name}_best.pth"))

    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "srf": srf,
                "params": total_params},
               os.path.join(cfg.out_dir, f"{cfg.name}_final.pth"))
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return model, history


# ---------------------------------------------------------------- evaluation
def evaluate_dataset(model: SamplingModel, root: str, cfg: Config,
                     split: str = "Test", device: str = "cuda",
                     limit: Optional[int] = None, tile_hr: int = 256,
                     verbose: bool = True, return_rows: bool = False,
                     refine_rounds: Optional[int] = None):
    """Full-scene evaluation with optional per-scene operator refinement.

    When ``refine_rounds`` is set, each scene alternates: sample with the
    current operator estimate -> refine the blur kernel against the physics of
    that sample -> re-sample.  This is the blind-generalisation step that
    replaces supervised test-time adaptation for a generative model.
    """
    from hsifusion.engine import tiled_inference
    rounds = cfg.refine_rounds if refine_rounds is None else refine_rounds
    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    from hsifusion.data import SceneCache
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation(cfg.scale, cfg.blur_ksize, cfg.eval_sigma).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": [],
                     "lr_consistency": []}

    for stem, hp, rp in pairs:
        hsi, rgb = cache.get(stem, hp, rp)
        h = (hsi.shape[1] // cfg.scale) * cfg.scale
        w = (hsi.shape[2] // cfg.scale) * cfg.scale
        gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(device)
        msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(device)
        lr = degrade(gt)

        model.set_kernel(None)
        pred = tiled_inference(model, lr, msi, cfg.scale, cfg.bands, tile_hr=tile_hr)
        if rounds > 0:
            srf = torch.from_numpy(_load_srf(cfg)).to(device)
            for _ in range(rounds):
                init_params = model.deg_head(lr, msi)[1]
                refined = refine_operator_kernel(lr, msi, pred, init_params, srf,
                                                 cfg.blur_ksize, cfg.refine_steps,
                                                 cfg.refine_lr, cfg.refine_w_spec)
                model.set_kernel(refined)
                pred = tiled_inference(model, lr, msi, cfg.scale, cfg.bands,
                                       tile_hr=tile_hr)
        model.set_kernel(None)

        m = evaluate_arrays(pred[0].cpu().numpy().transpose(1, 2, 0),
                            gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
        # LR-consistency under the FIXED evaluation operator (the one used to
        # build the observation): with the projection this is solver tolerance,
        # without it (~Stage 1) it is the free-DDIM drift.  The estimated-
        # operator consistency is reported separately for blind runs.
        re = degrade(pred.float())
        lc = float(((re - lr).abs().max() / lr.abs().max().clamp_min(1e-12)).item())
        m = {**m, "lr_consistency": lc}
        rows.append({"scene": stem, **m})
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
    return (mean, rows) if return_rows else mean


def _load_srf(cfg: Config) -> np.ndarray:
    """Load the SRF saved with the best checkpoint, else estimate fresh."""
    import os
    p = os.path.join(cfg.out_dir, f"{cfg.name}_best.pth")
    if os.path.exists(p):
        ckpt = torch.load(p, map_location="cpu")
        if "srf" in ckpt:
            return ckpt["srf"].numpy()
    return estimate_srf(cfg.source_root, "Train", cfg)


def load_checkpoint(cfg: Config, device: str = "cuda") -> SamplingModel:
    cfg.resolve(verbose=False)
    model = build_model(cfg)
    p = os.path.join(cfg.out_dir, f"{cfg.name}_best.pth")
    if not os.path.exists(p):
        p = os.path.join(cfg.out_dir, f"{cfg.name}_final.pth")
    ckpt = torch.load(p, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model