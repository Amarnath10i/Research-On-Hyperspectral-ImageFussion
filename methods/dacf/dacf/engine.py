"""DACF engine: training, adaptation, evaluation, and data loading."""

from __future__ import annotations

import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .losses import DACFLoss


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_mat(path: str) -> np.ndarray:
    """Load .mat file with scipy, fallback to h5py for v7.3."""
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        for k in data:
            if not k.startswith("_"):
                arr = data[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    return arr.astype(np.float32)
    except Exception:
        pass
    import h5py
    with h5py.File(path, "r") as f:
        for k in f:
            if not k.startswith("_"):
                arr = np.array(f[k])
                if arr.ndim == 3:
                    return arr.astype(np.float32)
                # Handle structured arrays
                if hasattr(arr, "dtype") and arr.dtype.names:
                    for name in arr.dtype.names:
                        sub = np.array(arr[name])
                        if sub.ndim == 3:
                            return sub.astype(np.float32)
    raise ValueError(f"Cannot load HSI from {path}")


def load_paviau(path: str) -> list[np.ndarray]:
    """Load PaviaU dataset (103 bands, single scene → patch split)."""
    mat_files = glob.glob(os.path.join(path, "**", "*.mat"), recursive=True)
    hsi_list = []
    for f in mat_files:
        try:
            arr = load_mat(f)
        except Exception as e:
            print(f"  Skipping {os.path.basename(f)}: {e}")
            continue
        if arr.ndim == 3 and min(arr.shape) > 10:
            # Detect HWC and transpose
            if arr.shape[-1] < arr.shape[0] and arr.shape[-1] < arr.shape[1]:
                arr = np.transpose(arr, (2, 0, 1))
            mx = float(arr.max())
            if mx > 1:
                arr = arr / mx
            hsi_list.append(arr.astype(np.float32))
            print(f"  Loaded {os.path.basename(f)}: {arr.shape}")
    return hsi_list


def load_chikusei(path: str) -> list[np.ndarray]:
    """Load Chikusei dataset (128 bands, single scene)."""
    mat_files = glob.glob(os.path.join(path, "**", "*.mat"), recursive=True)
    hsi_list = []
    for f in mat_files:
        try:
            arr = load_mat(f)
        except Exception as e:
            print(f"  Skipping {os.path.basename(f)}: {e}")
            continue
        if arr.ndim == 3 and min(arr.shape) > 10:
            if arr.shape[-1] < arr.shape[0] and arr.shape[-1] < arr.shape[1]:
                arr = np.transpose(arr, (2, 0, 1))
            mx = float(arr.max())
            if mx > 1:
                arr = arr / mx
            hsi_list.append(arr.astype(np.float32))
            print(f"  Loaded {os.path.basename(f)}: {arr.shape}")
    return hsi_list


def make_srf(bands: int, msi_bands: int = 3) -> torch.Tensor:
    """Create Gaussian spectral response function."""
    centers = np.linspace(0, bands - 1, msi_bands)
    srf = np.zeros((msi_bands, bands), dtype=np.float32)
    sigma = bands / (2 * msi_bands)
    for i, c in enumerate(centers):
        srf[i] = np.exp(-0.5 * ((np.arange(bands) - c) / sigma) ** 2)
    srf = srf / srf.sum(axis=1, keepdims=True)
    return torch.from_numpy(srf)


def gaussian_kernel2d(k: int, sx: float, sy: float) -> torch.Tensor:
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 / (2 * sx**2) + yy**2 / (2 * sy**2)))
    return kernel / kernel.sum()


def blur_downsample(x: torch.Tensor, kernel: torch.Tensor, scale: int) -> torch.Tensor:
    B, C, H, W = x.shape
    k = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1).to(x.device, x.dtype)
    pad = kernel.shape[0] // 2
    out = []
    for _ in range(max(1, B // 4)):
        blurred = F.conv2d(x.reshape(B * C, 1, H, W), k.unsqueeze(1), padding=pad, groups=C)
        out.append(blurred[:, :, ::scale, ::scale])
    return torch.cat(out, dim=1).squeeze(0)


def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute PSNR, SSIM, SAM, ERGAS."""
    pred_np = pred.detach().cpu().numpy().astype(np.float64)
    target_np = target.detach().cpu().numpy().astype(np.float64)

    # PSNR
    mse = np.mean((pred_np - target_np) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    # SSIM (simplified)
    mu_p, mu_t = pred_np.mean(), target_np.mean()
    sig_p = pred_np.std()
    sig_t = target_np.std()
    sig_pt = np.mean((pred_np - mu_p) * (target_np - mu_t))
    c1, c2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)) / \
           ((mu_p**2 + mu_t**2 + c1) * (sig_p**2 + sig_t**2 + c2))

    # SAM
    p = pred_np.reshape(pred_np.shape[0], -1)
    t = target_np.reshape(target_np.shape[0], -1)
    cos = np.sum(p * t, axis=0) / (np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0) + 1e-10)
    sam = np.mean(np.arccos(np.clip(cos, -1 + 1e-7, 1 - 1e-7))) * 180 / np.pi

    # ERGAS
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2, axis=(1, 2)))
    ergas = 100 * np.sqrt(np.mean((rmse / (np.mean(target_np, axis=(1, 2)) + 1e-10)) ** 2))

    return {"PSNR": psnr, "SSIM": ssim, "SAM": sam, "ERGAS": ergas}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DACFDataset(torch.utils.data.Dataset):
    """Patch-based dataset for DACF training."""

    def __init__(self, hsi_list: list[np.ndarray], bands: int, msi_bands: int,
                 srf: torch.Tensor, scale: int = 4, patch: int = 64,
                 train: bool = True, length: int = 8000,
                 sigma_range: tuple = (0.5, 3.0), noise_range: tuple = (0.0, 0.05)):
        self.hsi_list = hsi_list
        self.bands = bands
        self.scale = scale
        self.patch = patch
        self.train = train
        self.length = length if train else len(hsi_list)
        self.srf = srf
        self.sigma_range = sigma_range
        self.noise_range = noise_range

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.train:
            si = np.random.randint(len(self.hsi_list))
            hsi = self.hsi_list[si]
            _, h, w = hsi.shape
            top = np.random.randint(0, h - self.patch + 1)
            left = np.random.randint(0, w - self.patch + 1)
            gt = torch.from_numpy(hsi[:, top:top + self.patch, left:left + self.patch])
            # Augmentation
            if np.random.random() < 0.5:
                gt = torch.flip(gt, [-1])
            if np.random.random() < 0.5:
                gt = torch.flip(gt, [-2])
            k = np.random.randint(4)
            if k:
                gt = torch.rot90(gt, k, (-2, -1))
        else:
            si = idx % len(self.hsi_list)
            hsi = self.hsi_list[si]
            _, h, w = hsi.shape
            p = (min(h, w) // self.scale) * self.scale
            gt = torch.from_numpy(hsi[:, :p, :p])

        # Random degradation
        sx = np.random.uniform(*self.sigma_range)
        sy = sx if np.random.random() > 0.5 else np.random.uniform(*self.sigma_range)
        kernel = gaussian_kernel2d(9, sx, sy)
        lr = blur_downsample(gt, kernel, self.scale)

        noise = np.random.uniform(*self.noise_range) if self.train else 0.0
        if noise > 0:
            lr = (lr + torch.randn_like(lr) * noise).clamp(0, 1)

        msi = torch.einsum("mb,bhw->mhw", self.srf, gt).clamp(0, 1)

        return {"lr": lr, "msi": msi, "gt": gt, "kernel": kernel}


# ---------------------------------------------------------------------------
# Phase 1: Flow matching training
# ---------------------------------------------------------------------------

def train(model, dataset: DACFDataset, cfg: dict, device: str,
          name: str = "dacf") -> dict:
    """Phase 1: train flow matching with degradation randomization."""
    srf = dataset.srf.to(device)
    loss_fn = DACFLoss(
        lambda_flow=cfg.get("lambda_flow", 1.0),
        lambda_cycle=cfg.get("lambda_cycle", 0.5),
        lambda_srf=cfg.get("lambda_srf", 0.3),
        lambda_spectral=cfg.get("lambda_spectral", 0.1),
    )

    dl = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch"], shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    os.makedirs(f"{name}_out", exist_ok=True)
    best_psnr = -1
    history = {"iter": [], "loss": [], "psnr": [], "ssim": [], "sam": [], "ergas": []}

    t0 = time.time()
    model.train()
    it = iter(dl)

    for step in range(cfg["iters"]):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        lr_t = batch["lr"].to(device, non_blocking=True)
        msi = batch["msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        t = torch.rand(lr_t.shape[0], device=device)

        # LR schedule
        warmup = cfg.get("warmup", 500)
        if step < warmup:
            lr_now = cfg["lr"] * (step + 1) / warmup
        else:
            frac = (step - warmup) / max(1, cfg["iters"] - warmup)
            lr_now = 1e-6 + 0.5 * (cfg["lr"] - 1e-6) * (1 + math.cos(math.pi * frac))
        for p in opt.param_groups:
            p["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            pred = model.training_step(lr_t, msi, gt, t)
            # Also compute a 2-step sample for cycle loss
            with torch.no_grad():
                code = pred["code"]
                fused_2step = model.flow.sample(lr_t, code, steps=2)
                fused_2step = model.projector.project(fused_2step - lr_t) + lr_t
            loss = loss_fn.phase1(pred, gt, lr_t, msi, fused_2step, srf,
                                  model.projector)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % cfg.get("log_every", 100) == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / max(elapsed, 1e-6)
            eta = (cfg["iters"] - step - 1) / max(rate, 1e-6) / 60
            history["iter"].append(step)
            history["loss"].append(loss.item())
            print(f"[{step:6d}/{cfg['iters']}] loss={loss.item():.4f} "
                  f"lr={lr_now:.2e} {rate:.2f}it/s eta={eta:.0f}m")

        if (step + 1) % cfg.get("val_every", 2000) == 0 or step == cfg["iters"] - 1:
            model.eval()
            psnrs, ssims, sams, ergases = [], [], [], []
            n_val = min(len(dataset), cfg.get("val_scenes", 8))
            for vi in range(n_val):
                b = DACFDataset(dataset.hsi_list, dataset.bands,
                                dataset.msi_bands if hasattr(dataset, 'msi_bands') else 3,
                                dataset.srf, train=False)
                bt = b[vi]
                lr_v = bt["lr"].unsqueeze(0).to(device)
                msi_v = bt["msi"].unsqueeze(0).to(device)
                gt_v = bt["gt"].unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(lr_v, msi_v)["out"]
                m = all_metrics(out[0], gt_v[0])
                psnrs.append(m["PSNR"]); ssims.append(m["SSIM"])
                sams.append(m["SAM"]); ergases.append(m["ERGAS"])
            mean_p = np.mean(psnrs)
            print(f"  [val] PSNR={mean_p:.3f} SSIM={np.mean(ssims):.4f} "
                  f"SAM={np.mean(sams):.3f} ERGAS={np.mean(ergases):.3f}")
            history["psnr"].append(mean_p)
            history["ssim"].append(np.mean(ssims))
            history["sam"].append(np.mean(sams))
            history["ergas"].append(np.mean(ergases))
            if mean_p > best_psnr:
                best_psnr = mean_p
                torch.save({"state": model.state_dict(), "best_psnr": best_psnr},
                           f"{name}_out/best.pth")
                print(f"  [save] best.pth (PSNR={best_psnr:.3f})")
            model.train()
            if device == "cuda":
                torch.cuda.empty_cache()

    torch.save({"state": model.state_dict(), "best_psnr": best_psnr},
               f"{name}_out/final.pth")
    history["best_psnr"] = best_psnr
    print(f"[done] best PSNR={best_psnr:.3f}")
    return history


# ---------------------------------------------------------------------------
# Phase 2: Self-supervised adaptation
# ---------------------------------------------------------------------------

def adapt(model, lr_hsi: torch.Tensor, msi: torch.Tensor,
          srf: torch.Tensor, device: str, iters: int = 100,
          lr: float = 1e-3) -> None:
    """Phase 2: self-supervised adaptation on a single observation pair.

    Only updates the degradation encoder. Flow stays frozen.
    Enforces D(fused) ≈ LR-HSI and S(fused) ≈ MSI.
    """
    loss_fn = DACFLoss()
    # Only optimize encoder parameters
    opt = torch.optim.Adam(model.encoder.parameters(), lr=lr)

    model.train()
    model.flow.eval()
    model.projector.eval()

    for step in range(iters):
        opt.zero_grad(set_to_none=True)
        code = model.encode(lr_hsi, msi)
        fused = model.flow.sample(lr_hsi, code, steps=model.flow_steps)
        fused = model.projector.project(fused - lr_hsi) + lr_hsi
        loss = loss_fn.phase2(fused, lr_hsi, msi, srf, model.projector)
        loss.backward()
        opt.step()

    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(model, hsi_list: list[np.ndarray], srf: torch.Tensor,
                     device: str, patch_size: int = 256,
                     adapt_enabled: bool = True) -> dict:
    """Evaluate DACF on a dataset, optionally with Phase 2 adaptation."""
    model.eval()
    psnrs, ssims, sams, ergases = [], [], [], []
    srf_t = srf.to(device)

    for i, gt_np in enumerate(hsi_list):
        _, h, w = gt_np.shape
        p = min(patch_size, h, w)
        gt_t = torch.from_numpy(gt_np[:, :p, :p]).unsqueeze(0).to(device)
        lr_t = blur_downsample(gt_t, gaussian_kernel2d(9, 1.2, 1.2), model.scale)
        msi_t = torch.einsum("mb,bhw->mhw", srf_t, gt_t[0]).clamp(0, 1).unsqueeze(0)

        if device == "cuda":
            torch.cuda.empty_cache()

        # Phase 2 adaptation
        if adapt_enabled:
            adapt(model, lr_t, msi_t, srf_t, device, iters=100, lr=1e-3)

        with torch.no_grad():
            out = model(lr_t, msi_t)["out"]
        m = all_metrics(out[0], gt_t[0])
        psnrs.append(m["PSNR"]); ssims.append(m["SSIM"])
        sams.append(m["SAM"]); ergases.append(m["ERGAS"])
        print(f"  scene {i:2d}: PSNR={m['PSNR']:7.3f} SSIM={m['SSIM']:.4f} "
              f"SAM={m['SAM']:6.3f} ERGAS={m['ERGAS']:8.3f}")
        del gt_t, lr_t, msi_t, out
        if device == "cuda":
            torch.cuda.empty_cache()

    mean = {"PSNR": np.mean(psnrs), "SSIM": np.mean(ssims),
            "SAM": np.mean(sams), "ERGAS": np.mean(ergases)}
    print(f"  MEAN:     PSNR={mean['PSNR']:7.3f} SSIM={mean['SSIM']:.4f} "
          f"SAM={mean['SAM']:6.3f} ERGAS={mean['ERGAS']:8.3f}")
    return mean
