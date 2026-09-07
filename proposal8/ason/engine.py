"""ASON training and evaluation engine."""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .losses import ASONLoss, charbonnier


def train(model, dataset, cfg: dict, device: str, name: str = "ason") -> dict:
    """Train ASON model.

    Args:
        model: ASONNet instance
        dataset: HSIFusionDataset with .srf attribute
        cfg: dict with batch, iters, lr, warmup, val_every, log_every, val_scenes
        device: 'cuda' or 'cpu'
        name: output directory prefix

    Returns:
        history dict with iter, loss, psnr, ssim, sam, ergas, best_psnr
    """
    from ..io_utils import all_metrics, HSIFusionDataset, blur_downsample, gaussian_kernel2d

    srf = dataset.srf
    loss_fn = ASONLoss()
    dl = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch"], shuffle=True,
                                      num_workers=0, pin_memory=(device == "cuda"))
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
        kernel = batch["kernel"].to(device, non_blocking=True)
        t = torch.rand(lr_t.shape[0], device=device)

        # LR schedule
        warmup, iters = cfg.get("warmup", 1000), cfg["iters"]
        if step < warmup:
            lr_now = cfg["lr"] * (step + 1) / warmup
        else:
            frac = (step - warmup) / max(1, iters - warmup)
            lr_now = 1e-6 + 0.5 * (cfg["lr"] - 1e-6) * (1 + math.cos(math.pi * frac))
        for p in opt.param_groups:
            p["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            pred = model.training_step(lr_t, msi, gt, t, kernel)
            sampled = model.sample(lr_t, msi, kernel=kernel, steps=2, code=pred["code"])
            loss = loss_fn(pred, gt, lr_t, msi, srf, sampled)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step % cfg.get("log_every", 200) == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / max(elapsed, 1e-6)
            eta = (cfg["iters"] - step - 1) / max(rate, 1e-6) / 60
            history["iter"].append(step)
            history["loss"].append(loss.item())
            print(f"[{step:6d}/{cfg['iters']}] loss={loss.item():.4f} "
                  f"lr={lr_now:.2e} {rate:.2f}it/s eta={eta:.0f}m")

        if (step + 1) % cfg.get("val_every", 3000) == 0 or step == cfg["iters"] - 1:
            model.eval()
            psnrs, ssims, sams, ergases = [], [], [], []
            for vi in range(min(len(dataset), cfg.get("val_scenes", 4))):
                b = HSIFusionDataset(dataset.hsi_list, dataset.bands,
                                     dataset.msi_bands if hasattr(dataset, 'msi_bands') else 3,
                                     dataset.srf, train=False)
                bt = b[vi]
                lr_v = bt["lr"].unsqueeze(0).to(device)
                msi_v = bt["msi"].unsqueeze(0).to(device)
                gt_v = bt["gt"].unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(lr_v, msi_v)["out"]
                m = all_metrics(out[0], gt_v[0])
                psnrs.append(m["PSNR"])
                ssims.append(m["SSIM"])
                sams.append(m["SAM"])
                ergases.append(m["ERGAS"])
            mean_p = np.mean(psnrs)
            print(f"  [val] PSNR={mean_p:.3f} SSIM={np.mean(ssims):.4f} "
                  f"SAM={np.mean(sams):.3f} ERGAS={np.mean(ergases):.3f}")
            history["psnr"].append(mean_p)
            history["ssim"].append(np.mean(ssims))
            history["sam"].append(np.mean(sams))
            history["ergas"].append(np.mean(ergases))
            if mean_p > best_psnr:
                best_psnr = mean_p
                torch.save({"state": model.state_dict(), "cfg": cfg, "srf": srf.cpu(),
                            "best": {"psnr": mean_p}},
                           f"{name}_out/best.pth")
                print(f"  [save] best.pth (PSNR={best_psnr:.3f})")
            model.train()
            if device == "cuda":
                torch.cuda.empty_cache()

    torch.save({"state": model.state_dict(), "best_psnr": best_psnr}, f"{name}_out/final.pth")
    history["best_psnr"] = best_psnr
    print(f"[done] best PSNR={best_psnr:.3f}")
    return history


def evaluate_dataset(model, hsi_list, srf, device: str, patch_size: int = 256) -> dict:
    """Evaluate ASON on a dataset.

    Args:
        model: trained ASONNet
        hsi_list: list of np arrays [C, H, W]
        srf: [msi_bands, bands] tensor
        device: 'cuda' or 'cpu'
        patch_size: crop size to avoid OOM on large scenes

    Returns:
        dict with mean PSNR, SSIM, SAM, ERGAS
    """
    from ..io_utils import all_metrics, blur_downsample, gaussian_kernel2d

    model.eval()
    psnrs, ssims, sams, ergases = [], [], [], []
    srf_t = srf.to(device)

    for i, gt_np in enumerate(hsi_list):
        _, h, w = gt_np.shape
        p = min(patch_size, h, w)
        gt_t = torch.from_numpy(gt_np[:, :p, :p]).unsqueeze(0).to(device)
        lr_t = blur_downsample(gt_t, gaussian_kernel2d(9, 1.2, 1.2), model.scale)
        msi_t = torch.einsum("mb,bhw->mhw", srf_t, gt_t[0]).clamp(0, 1).unsqueeze(0)
        torch.cuda.empty_cache() if device == "cuda" else None
        with torch.no_grad():
            out = model(lr_t, msi_t)["out"]
        m = all_metrics(out[0], gt_t[0])
        psnrs.append(m["PSNR"])
        ssims.append(m["SSIM"])
        sams.append(m["SAM"])
        ergases.append(m["ERGAS"])
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
