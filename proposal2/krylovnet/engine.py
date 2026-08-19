"""Training / evaluation engine for KrylovNet (mirrors proposal1.slt)."""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from proposal1.daetf.data import FusionPatchDataset, estimate_srf
from proposal1.daetf.degrade import FixedDegradation
from proposal1.daetf.metrics import evaluate_arrays

from .config import Config
from .losses import KrylovLoss
from .model import KrylovNet


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)


def _make_model(cfg: Config, device: torch.device, srf: np.ndarray) -> KrylovNet:
    model = KrylovNet(cfg).to(device)
    model.set_srf(torch.from_numpy(srf.astype(np.float32)))
    return model


def _lr_schedule(it: int, cfg: Config) -> float:
    if it < cfg.warmup:
        return cfg.lr * (it + 1) / cfg.warmup
    t = (it - cfg.warmup) / max(1, cfg.iters - cfg.warmup)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))



class _TupleAdapter(torch.utils.data.Dataset):
    """Bridge between the shared dataset and this engine's training loop.

    FusionPatchDataset yields a dict; the loop below consumes
    (lr, msi, gt, kernel, name). Adapting here keeps the shared data pipeline
    as the single source of truth for the protocol - the alternative, a
    private dataset per proposal, is how the ten baselines in existing/ ended
    up incomparable.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        return d["lr"], d["msi"], d["gt"], d["kernel"], d["name"]


def train(cfg: Config, device: Optional[str] = None) -> dict:
    cfg.resolve()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.seed)

    srf = estimate_srf(cfg.source_root, "Train", cfg)
    model = _make_model(cfg, device, srf)
    loss_fn = KrylovLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay)

    ds = _TupleAdapter(FusionPatchDataset(
        cfg.source_root, "Train", cfg, train=True, srf=srf,
        length=max(cfg.iters * cfg.batch, 1)))
    dl = DataLoader(ds, batch_size=cfg.batch, shuffle=True,
                    num_workers=cfg.workers, pin_memory=device.type == "cuda",
                    persistent_workers=(cfg.workers > 0 and device.type == "cuda"))
    it = iter(dl)

    os.makedirs(cfg.out_dir, exist_ok=True)
    step = 0
    best = {"psnr": -1.0}
    history = {"psnr": [], "ssim": [], "sam": [], "ergas": [], "loss": []}

    t0 = time.time()
    model.train()
    for it_idx in range(cfg.iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        lr_t, msi, gt, kernel, _ = batch
        lr_t, msi, gt, kernel = (x.to(device) for x in (lr_t, msi, gt, kernel))
        if device.type == "cpu":
            kernel = kernel.float()

        for p in optimizer.param_groups:
            p["lr"] = _lr_schedule(it_idx, cfg)

        with autocast(enabled=cfg.amp and device.type == "cuda"):
            pred = model(lr_t, msi, kernel)
            loss = loss_fn(pred, gt, lr_t, msi, kernel, model)
        total = loss["loss"] / cfg.grad_accum
        scaler.scale(total).backward()
        if (it_idx + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        if it_idx % cfg.log_every == 0:
            history["loss"].append(loss["loss"].item())
            print(f"[{it_idx}/{cfg.iters}] loss={loss['loss'].item():.4f} "
                  f"phys={loss['phys'].item():.3f} spec={loss['spec'].item():.3f} "
                  f"res={loss['res'].item():.4f} "
                  f"({time.time() - t0:.0f}s)")

        if (it_idx + 1) % cfg.val_every == 0 or it_idx == cfg.iters - 1:
            ema.apply_to(model)
            res = evaluate_dataset(model, cfg, device)
            print(f"[val @ {it_idx}] " +
                  " ".join(f"{k}={v:.4f}" for k, v in res.items()))
            for k in ("psnr", "ssim", "sam", "ergas"):
                history[k].append(res[k])
            if res["psnr"] > best["psnr"]:
                best = res
                ckpt = {
                    "cfg": cfg.to_dict(), "state": model.state_dict(),
                    "best": best, "step": it_idx,
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))
                print("[save] best.pt")
            ema.apply_to(model)

    history["best"] = best
    print(f"[done] best {best}")
    return history


def evaluate_dataset(model: nn.Module, cfg: Config, device: torch.device,
                     mode: str = "full") -> dict:
    """Delegate to the shared evaluator.

    The previous implementation here was written against a different data
    format entirely - it called find_pairs(source_root, target_root), unpacked
    its 3-tuples as pairs, loaded scenes with np.load(...)["arr_0"] when the
    datasets ship .mat, and constructed FixedDegradation with the arguments in
    the wrong order. It had never been executed.

    Delegating is also the point of the shared library: every proposal is then
    scored by one metric implementation under one degradation at one scale
    factor. A private evaluator per proposal is exactly how the ten published
    baselines in existing/ became mutually incomparable.

    KrylovNet.forward(lr, msi, kernel=None) -> {"out": ...} already satisfies
    the shared (lr, msi) -> {"out": ...} contract, so no adapter is needed.
    """
    from proposal1.daetf.engine import evaluate_dataset as _shared_eval

    was_training = model.training
    model.eval()
    try:
        res = _shared_eval(
            model, cfg.source_root, cfg, "Test", str(device),
            limit=(cfg.val_scenes if mode == "quick" else None),
            verbose=False)
    finally:
        if was_training:
            model.train()
    return res


def tiled_inference(model: nn.Module, lr: np.ndarray, msi: np.ndarray,
                    scale: int, bands: int, msi_bands: int,
                    srf: np.ndarray, device: torch.device,
                    tile: int = 256, batch: int = 8,
                    kernel: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = lr.shape[:2]
    H, W = h * scale, w * scale
    out = np.zeros((bands, H, W), dtype=np.float32)
    tiles = [(y, min(y + tile, H)) for y in range(0, H, tile)]
    tiles = [(y0, y1) for y0, y1 in tiles if y1 - y0 > 0]
    for i in range(0, len(tiles), batch):
        ts = tiles[i:i + batch]
        lr_b = torch.zeros(batch, bands, h, w, device=device)
        msi_b = torch.zeros(batch, msi_bands, h, w, device=device)
        coords = []
        for j, (y0, y1) in enumerate(ts):
            x0, x1 = y0, y1
            lr_p = lr[y0 // scale:x1 // scale, x0 // scale:x1 // scale, :]
            msi_p = msi[y0 // scale:x1 // scale, x0 // scale:x1 // scale, :]
            coords.append((y0, y1))
            lr_b[j, :, :lr_p.shape[0], :lr_p.shape[1]] = \
                torch.from_numpy(lr_p.transpose(2, 0, 1)).float().to(device)
            msi_b[j, :, :msi_p.shape[0], :msi_p.shape[1]] = \
                torch.from_numpy(msi_p.transpose(2, 0, 1)).float().to(device)
        with torch.no_grad():
            pred = model(lr_b, msi_b, None if kernel is None else
                         torch.from_numpy(kernel).float().to(device))
            for j, (y0, y1) in enumerate(coords):
                out[:, y0:y1, y0:y1] = pred["out"][j, :, :y1 - y0, :y1 - y0].cpu().numpy()
    return out.transpose(1, 2, 0)


def load_checkpoint(path: str, cfg: Config, device: torch.device) -> KrylovNet:
    ckpt = torch.load(path, map_location=device)
    srf = estimate_srf(cfg.source_root, "Train", cfg)
    model = _make_model(cfg, device, srf)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model