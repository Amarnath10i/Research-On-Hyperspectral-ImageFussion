"""Training, one-step sampling wrapper and evaluation for ConsistentFlow.

The public model follows the shared protocol contract
``forward(lr_hsi, msi) -> {"out": ...}``; under the hood ``forward`` runs the
distilled one-step consistency map followed by the null-space projection, so
``D(out) == lr_hsi`` to solver tolerance at ~regressor cost.

Training is **consistency distillation** (Song et al.): the student
consistency map is pushed toward its own EMA along teacher (P5 score net)
trajectories, collapsing the multi-step sampler into a single-step map.  The
projection participates in training, so the map learns to output points on the
consistent set rather than negotiating with the observation.
"""

from __future__ import annotations

import copy
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

from spectralflow.nullspace import (RangeNullProjector,
                                    decode_degradation_params,
                                    kernel_from_params)
from spectralflow.sampler import LinearNoiseSchedule
from spectralflow.score import SpectralScoreNet

from .config import Config
from .sampler import ConsistencySampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DegCodeHead(nn.Module):
    """Reads the observed pair and emits a degradation code plus physical
    operator parameters (sigma_x, sigma_y, sin2t, cos2t, noise).  Same role as
    in P5; kept local so the import surface of this package is small."""

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


class ConsistencyModel(nn.Module):
    """The protocol-facing wrapper: one-step consistency sampling IS the
    forward pass.  ``D(out) == lr_hsi`` to solver tolerance because the output
    is re-projected onto the consistent set after the map."""

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
        self.sampler = ConsistencySampler(
            self.score_net, self.projector, num_timesteps=cfg.num_timesteps,
            sample_steps=cfg.sample_steps,
            beta_start=cfg.beta_start, beta_end=cfg.beta_end,
            use_projection=cfg.use_projection)
        self._ext_kernel: Optional[torch.Tensor] = None

    def set_projection(self, enabled: bool) -> None:
        """Toggle the null-space projection (control arm for the ablation)."""
        self.sampler.use_projection = bool(enabled)

    def set_kernel(self, kernel: Optional[torch.Tensor]) -> None:
        self._ext_kernel = (None if kernel is None
                            else kernel.detach().clone())

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor):
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


def build_model(cfg: Config) -> ConsistencyModel:
    return ConsistencyModel(cfg)


# ------------------------------------------------------ consistency distillation
def consistent_output(x0: torch.Tensor, lr_hsi: torch.Tensor, kernel,
                      projector: RangeNullProjector) -> torch.Tensor:
    """D_pinv(X) + P_perp(clamp(x0)) - map the estimate onto the consistent set."""
    hw = (x0.shape[-2], x0.shape[-1])
    return projector.pinv(lr_hsi, hw, kernel) \
        + projector.project_null(x0.clamp(0, 1), kernel)


def to_x0_estimate(net: SpectralScoreNet, y_t: torch.Tensor, msi: torch.Tensor,
                   code: torch.Tensor, t: torch.Tensor,
                   schedule: LinearNoiseSchedule) -> torch.Tensor:
    a = schedule.a[t].to(y_t.dtype).reshape(-1, 1, 1, 1)
    sq1 = (1 - schedule.a[t]).to(y_t.dtype).clamp_min(1e-8).sqrt().reshape(-1, 1, 1, 1)
    eps = net(y_t, msi, code, t)
    return (y_t - sq1 * eps) / a.clamp_min(1e-8)


def consistency_distill_loss(student: SpectralScoreNet,
                             target: SpectralScoreNet,
                             teacher: SpectralScoreNet,
                             schedule: LinearNoiseSchedule,
                             projector: RangeNullProjector,
                             gt: torch.Tensor, msi: torch.Tensor, lr: torch.Tensor,
                             code: torch.Tensor, kernel: torch.Tensor,
                             use_projection: bool = True) -> torch.Tensor:
    """CD loss: push f_phi(y_{t+1}) toward f_phi^- (y_t) along a teacher
    trajectory, both mapped onto the consistent set.  The target net is the
    EMA copy; the teacher is the P5 score net that defines the trajectory."""
    b = gt.shape[0]
    t = torch.randint(1, schedule.T, (b,), device=gt.device)
    tp = torch.clamp(t + 1, max=schedule.T)

    a_next = schedule.a[tp].to(gt.dtype).reshape(-1, 1, 1, 1)
    a_t = schedule.a[t].to(gt.dtype).reshape(-1, 1, 1, 1)
    sq1_next = (1 - a_next).clamp_min(1e-8).sqrt()
    sq1_t = (1 - a_t).clamp_min(1e-8).sqrt()

    eps = torch.randn_like(gt)
    y_next = torch.sqrt(a_next) * gt + sq1_next * eps
    # teacher one-step estimate at t+1, then DDIM step down to t
    eps_teacher = teacher(y_next, msi, code, tp)
    x0_next = (y_next - sq1_next * eps_teacher) / a_next.clamp_min(1e-8)
    y_t = torch.sqrt(a_t) * x0_next + sq1_t * eps_teacher

    f_next = to_x0_estimate(student, y_next, msi, code, tp, schedule)
    f_t = to_x0_estimate(target, y_t, msi, code, t, schedule)
    if use_projection:
        f_next = consistent_output(f_next, lr, kernel, projector)
        f_t = consistent_output(f_t, lr, kernel, projector)
    return F.mse_loss(f_next, f_t.detach())


def cosine_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup:
        return cfg.lr * step / max(cfg.warmup, 1)
    t = (step - cfg.warmup) / max(cfg.iters - cfg.warmup, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


# ------------------------------------------------------------------ training
def train(cfg: Config, device: str = "cuda", log_fn=print) -> Tuple[ConsistencyModel, Dict]:
    """Distill the consistency map from a (P5-architecture) teacher.

    The teacher is initialised identically to the student, so this is a
    self-distillation: the CD objective collapses the multi-step trajectory the
    teacher defines into a single-step map.
    """
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
    schedule = LinearNoiseSchedule(cfg.num_timesteps, cfg.beta_start,
                                   cfg.beta_end).to(device)
    teacher = copy.deepcopy(model.score_net).to(device).eval()
    target = copy.deepcopy(model.score_net).to(device)
    for p in target.parameters():
        p.requires_grad = False

    params = list(model.score_net.parameters()) + list(model.deg_head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-5, betas=(0.9, 0.99))
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_fn(f"{cfg.name}: {sum(p.numel() for p in params) / 1e6:.2f} M "
           f"trainable parameters, sample_steps={cfg.sample_steps}")

    history: Dict = {"iter": [], "loss": [], "cfg": cfg.to_dict()}
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        if step > cfg.iters:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg)

        gt = batch["gt"].to(device, non_blocking=True)
        msi = batch["msi"].to(device, non_blocking=True)
        lr = batch["lr"].to(device, non_blocking=True)
        kernel = batch["kernel"].to(device, non_blocking=True)
        code = model.deg_head(lr, msi)[0]

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = consistency_distill_loss(
                model.score_net, target, teacher, schedule, model.projector,
                gt, msi, lr, code, kernel, use_projection=cfg.use_projection)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():          # EMA target network
            for p, tp in zip(model.score_net.parameters(), target.parameters()):
                tp.data.mul_(cfg.ema_decay).add_(p.data, alpha=1 - cfg.ema_decay)

        if step % cfg.log_every == 0:
            rate = step / (time.time() - t0)
            log_fn(f"it {step:6d}/{cfg.iters}  cd {loss.item():.5f}  "
                   f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.2f} it/s")
            history["iter"].append(step)
            history["loss"].append(loss.item())

    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "srf": srf},
               os.path.join(cfg.out_dir, f"{cfg.name}_final.pth"))
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return model, history


# ---------------------------------------------------------------- evaluation
def evaluate_dataset(model: ConsistencyModel, root: str, cfg: Config,
                     split: str = "Test", device: str = "cuda",
                     limit: Optional[int] = None, tile_hr: int = 256,
                     verbose: bool = True, return_rows: bool = False):
    """Full-scene evaluation.  Every scene reports LR-consistency under the
    fixed evaluation operator: with the projection it is solver tolerance in a
    single forward pass - the claim this proposal exists to make."""
    from hsifusion.engine import tiled_inference
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
        model.set_kernel(None)

        m = evaluate_arrays(pred[0].cpu().numpy().transpose(1, 2, 0),
                            gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
        re = degrade(pred.float())
        m["lr_consistency"] = float(
            ((re - lr).abs().max() / lr.abs().max().clamp_min(1e-12)).item())
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


def load_checkpoint(cfg: Config, device: str = "cuda") -> ConsistencyModel:
    cfg.resolve(verbose=False)
    model = build_model(cfg)
    p = os.path.join(cfg.out_dir, f"{cfg.name}_final.pth")
    ckpt = torch.load(p, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model
