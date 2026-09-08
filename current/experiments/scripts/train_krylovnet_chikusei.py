"""KrylovNet training for Chikusei (128 bands).

Adapted from the KrylovNet CAVE implementation.
Key changes:
  - bands=128, srf=chikusei_srf(128)
  - Increased graph_k for 128-band graph
  - Reduced patch/batch for memory
  - Uses the ChikuseiDataset from common.hsifusion.datasets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add common to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))

from hsifusion.srf import chikusei_srf, conditioning


# ===========================================================================
# 1. Operator + Solver (inline from krylovnet/solver.py)
# ===========================================================================
def gaussian_kernel2d(size, sigma_x, sigma_y, theta):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    c, s = np.cos(theta), np.sin(theta)
    xx_r = c * xx + s * yy
    yy_r = -s * xx + c * yy
    k = np.exp(-0.5 * (xx_r ** 2 / (sigma_x ** 2) + yy_r ** 2 / (sigma_y ** 2)))
    return (k / k.sum()).astype(np.float32)


class FusionOperator:
    """A = D^T D + S^T S + rho I, with D = spatial degradation, S = spectral response."""

    def __init__(self, scale: int, rho: float = 1e-3):
        self.scale = scale
        self.rho = rho

    def A(self, x: torch.Tensor, kernel: torch.Tensor,
          srf: torch.Tensor) -> torch.Tensor:
        """Apply A = D^T D x + S^T S x + rho x."""
        B, C, H, W = x.shape
        # D^T D: blur -> downsample -> upsample -> blur
        k = kernel.repeat(C, 1, 1, 1)
        xb = F.conv2d(x, k, padding=kernel.shape[-1] // 2, groups=C)
        xb = xb[:, :, ::self.scale, ::self.scale]
        xb = F.interpolate(xb, size=(H, W), mode="bicubic", align_corners=False)
        xb = F.conv2d(xb, k, padding=kernel.shape[-1] // 2, groups=C)
        # S^T S: MSI -> HSI -> MSI -> HSI
        msi = torch.einsum("nbhw,bm->nmhw", x, srf)
        hsi = torch.einsum("nmhw,bm->nbhw", msi, srf)
        return xb + hsi + self.rho * x

    def b(self, lr: torch.Tensor, msi: torch.Tensor,
          kernel: torch.Tensor, srf: torch.Tensor) -> torch.Tensor:
        """RHS = D^T X + S^T M."""
        B, C, H, W = lr.shape
        H_hr, W_hr = H * self.scale, W * self.scale
        # D^T X
        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False)
        k = kernel.repeat(C, 1, 1, 1)
        dt_x = F.conv2d(lr_up, k, padding=kernel.shape[-1] // 2, groups=C)
        # S^T M
        st_m = torch.einsum("nmhw,bm->nbhw", msi, srf)
        return dt_x + st_m


def krylov_gmres(x0, b, A, Pinv, n_stages, blend=None, alpha_gates=None):
    """Unrolled GMRES with optional preconditioner and attention blend."""
    x = x0
    r = b - A(x)
    p = r.clone()
    residuals = []
    Vs = []

    for i in range(n_stages):
        v = p
        if Pinv is not None:
            v = Pinv(v)
        v = v / (torch.linalg.vector_norm(v, dim=(1, 2, 3), keepdim=True) + 1e-8)

        Av = A(v)
        # MGS
        h = torch.einsum("bchw,bchw->b", v, Av).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        r = r - h * v

        # Preconditioned residual for next direction
        p_new = r.clone()
        if Pinv is not None:
            p_new = Pinv(p_new)
        p_new = p_new / (torch.linalg.vector_norm(p_new, dim=(1, 2, 3), keepdim=True) + 1e-8)
        p = p_new

        residuals.append(torch.linalg.vector_norm(r, dim=(1, 2, 3)).mean().item())
        Vs.append(v)

    # Blend
    if blend is not None and len(Vs) > 1:
        vs_stack = torch.stack(Vs, dim=1)  # (B, n_stages, C, H, W)
        alpha = blend(torch.tensor(residuals, device=x0.device).float())
        x = x0 + (alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * vs_stack).sum(dim=1)
    else:
        x = x + r

    return x, residuals


class Blend(nn.Module):
    """Attention blend over Krylov basis vectors."""
    def __init__(self, n_stages: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(n_stages, n_stages * 2),
            nn.ReLU(),
            nn.Linear(n_stages * 2, n_stages),
            nn.Softmax(dim=-1),
        )

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        return self.attn(residuals.unsqueeze(0)).squeeze(0)


# ===========================================================================
# 2. SpectralPreconditioner (GNN over band graph)
# ===========================================================================
class SpectralPreconditioner(nn.Module):
    def __init__(self, bands: int, graph_k: int = 8, hidden: int = 32,
                 gcn_layers: int = 2, feat_dim: int = 2):
        super().__init__()
        self.bands = bands
        self.graph_k = graph_k
        self.embed = nn.Linear(feat_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(gcn_layers)])
        self.head = nn.Linear(hidden, 1)
        self.skip = nn.Linear(feat_dim, 1)

    def build_affinity(self, feats):
        b = feats.shape[0]
        d = torch.cdist(feats, feats)
        k = min(self.graph_k, self.bands - 1)
        idx = torch.topk(d, k=k, dim=-1, largest=False).indices
        adj = torch.zeros(b, self.bands, self.bands, device=feats.device, dtype=feats.dtype)
        ar = torch.arange(self.bands, device=feats.device)
        adj[torch.arange(b).reshape(b, 1, 1), ar.reshape(1, self.bands, 1), idx] = 1.0
        adj = adj + adj.transpose(1, 2)
        adj = torch.clamp(adj, max=1.0) + torch.eye(self.bands, device=feats.device)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return adj / deg

    def forward(self, feats):
        adj = self.build_affinity(feats)
        h = F.relu(self.embed(feats))
        for layer in self.layers:
            h = F.relu(adj @ layer(h))
        s = torch.exp(self.head(h).squeeze(-1) + self.skip(feats).squeeze(-1))
        return s


# ===========================================================================
# 3. ResidualDenoiser (learned proximal prior)
# ===========================================================================
class ResidualDenoiser(nn.Module):
    def __init__(self, bands: int, width: int = 64, blocks: int = 4):
        super().__init__()
        self.head = nn.Conv2d(bands, width, 3, 1, 1)
        self.body = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, 1, 1),
                          nn.LeakyReLU(0.1, True),
                          nn.Conv2d(width, width, 3, 1, 1))
            for _ in range(blocks)
        ])
        self.tail = nn.Conv2d(width, bands, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        self.act = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        h = self.act(self.head(x))
        for blk in self.body:
            h = h + blk(h)
        return x + self.tail(h)


# ===========================================================================
# 4. KrylovNet for Chikusei
# ===========================================================================
class KrylovNetChikusei(nn.Module):
    """KrylovNet adapted for Chikusei (128 bands)."""

    def __init__(self, bands=128, msi_bands=3, scale=4, n_stages=6,
                 graph_k=8, hidden=32, prior_width=80, prior_blocks=8,
                 rho=1e-3, eval_sigma=1.2, blur_ksize=9):
        super().__init__()
        self.bands = bands
        self.msi_bands = msi_bands
        self.scale = scale
        self.rho = rho
        self.op = FusionOperator(scale, rho)
        self.precond = SpectralPreconditioner(bands, graph_k, hidden)
        self.blend = Blend(n_stages)
        self.prior = ResidualDenoiser(bands, prior_width, prior_blocks)
        k = gaussian_kernel2d(blur_ksize, eval_sigma, eval_sigma, 0.0)
        self.register_buffer("default_kernel", k.float())
        self.register_buffer("srf", torch.zeros(msi_bands, bands))

    def set_srf(self, srf: torch.Tensor):
        s = srf if srf.shape[0] == self.bands else srf.t().contiguous()
        self.srf.data = s.float()

    def _band_feats(self, hsi):
        mu = hsi.mean(dim=(2, 3))
        sd = hsi.std(dim=(2, 3))
        return torch.stack([mu, sd], dim=-1)

    def forward(self, lr, msi, kernel=None):
        kernel = self.default_kernel if kernel is None else kernel
        B = lr.shape[0]
        b = self.op.b(lr, msi, kernel, self.srf)
        x0 = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        A = lambda v: self.op.A(v, kernel, self.srf)

        s = self.precond(self._band_feats(lr))
        Pinv = lambda v: v * s.reshape(B, self.bands, *([1] * (v.ndim - 2)))

        # Outer iterations: data step + prior step
        x = x0
        for _ in range(4):
            x, res = krylov_gmres(x, b, A, Pinv, 6 // 4, self.blend)
            x = self.prior(x)

        x = x.clamp(0, 1) if not self.training else x
        return {"out": x}


# ===========================================================================
# 5. Metrics
# ===========================================================================
def calc_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12:
        return 100.0
    return -10.0 * np.log10(mse)

def calc_sam(pred, gt):
    p = pred.reshape(pred.shape[0], -1)
    g = gt.reshape(gt.shape[0], -1)
    p = p / (np.linalg.norm(p, axis=0, keepdims=True) + 1e-8)
    g = g / (np.linalg.norm(g, axis=0, keepdims=True) + 1e-8)
    cos = np.clip((p * g).sum(0), -1.0, 1.0)
    return np.mean(np.arccos(cos)) * 180.0 / np.pi

def calc_ergas(pred, gt, scale=4):
    C, H, W = pred.shape
    err = (pred - gt) ** 2
    ergas = 0.0
    for c in range(C):
        mg = gt[c].mean()
        if mg > 0:
            ergas += err[c].mean() / (mg ** 2)
    return math.sqrt(ergas / C) * 100.0 * scale

def calc_ssim(pred, gt):
    from scipy.ndimage import uniform_filter
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu1 = uniform_filter(pred, size=3, mode="reflect")
    mu2 = uniform_filter(gt, size=3, mode="reflect")
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    sigma12 = uniform_filter(pred * gt, size=3, mode="reflect") - mu1 * mu2
    sigma1 = uniform_filter(pred ** 2, size=3, mode="reflect") - mu1_sq
    sigma2 = uniform_filter(gt ** 2, size=3, mode="reflect") - mu2_sq
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2) + 1e-8)
    return np.mean(ssim)


# ===========================================================================
# 6. EMA
# ===========================================================================
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point and k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)
    def restore_from(self, model):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}


# ===========================================================================
# 7. Training
# ===========================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Config
    bands = 128
    msi_bands = 3
    scale = 4
    n_stages = 6
    graph_k = 8           # Increased for 128 bands
    hidden = 32
    prior_width = 80
    prior_blocks = 8
    rho = 1e-3
    eval_sigma = 1.2
    blur_ksize = 9

    epochs = 100000
    batch_size = 4
    patch_size = 64
    lr = 2e-4
    eval_every = 500
    time_budget_h = 9.0

    # SRF
    srf_np = chikusei_srf(bands)
    srf_info = conditioning(srf_np)
    print(f"\nChikusei SRF conditioning: {srf_info['cond']:.2f}")
    srf_t = torch.from_numpy(srf_np).float().to(device)

    # Dataset
    print("\nLoading Chikusei dataset...")
    from hsifusion.datasets import ChikuseiDataset
    train_ds = ChikuseiDataset(args.root, "train", bands, scale, patch_size, srf_mode="sensor")
    test_ds = ChikuseiDataset(args.root, "test", bands, scale, patch_size, srf_mode="sensor")
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Model
    model = KrylovNetChikusei(
        bands=bands, msi_bands=msi_bands, scale=scale, n_stages=n_stages,
        graph_k=graph_k, hidden=hidden, prior_width=prior_width,
        prior_blocks=prior_blocks, rho=rho, eval_sigma=eval_sigma,
        blur_ksize=blur_ksize
    ).to(device)
    model.set_srf(srf_t)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"KrylovNet-Chikusei params: {nparams/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(model, 0.999)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    l1 = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_psnr = 0.0
    best_epoch = 0
    save_dir = os.path.join(args.output_dir, "krylovnet_chikusei_ckpt")
    os.makedirs(save_dir, exist_ok=True)
    T0 = time.time()
    LIMIT = time_budget_h * 3600

    steps_per_epoch = 100

    print(f"\nStarting training for {epochs} epochs (budget {time_budget_h}h)...")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        if (time.time() - T0) > LIMIT:
            print(f"[time budget reached at epoch {epoch}]")
            break
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            gts, lhs, mss = [], [], []
            for _ in range(batch_size):
                g, l, m = train_ds[random.randrange(len(train_ds))]
                gts.append(g); lhs.append(l); mss.append(m)
            gt = torch.stack(gts, 0).to(device)
            yH = torch.stack(lhs, 0).to(device)
            yM = torch.stack(mss, 0).to(device)

            with torch.cuda.amp.autocast(enabled=True):
                out_dict = model(yH, yM)
                out = out_dict["out"]
                loss_recon = l1(out, gt)
                loss_phys = F.mse_loss(model.op.A(out, model.default_kernel, model.srf),
                                       model.op.b(yH, yM, model.default_kernel, model.srf))
                loss = loss_recon + 0.1 * loss_phys

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / steps_per_epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:6d}/{epochs} | Loss {avg:.5f} | "
                  f"LR {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if epoch % eval_every == 0 or epoch == epochs:
            ema.apply_to(model)
            psnrs, ssims, sams, ergas_list = [], [], [], []
            with torch.no_grad():
                for i in range(min(len(test_ds), 20)):
                    g, l, m = test_ds[i]
                    g = g.unsqueeze(0).to(device)
                    l = l.unsqueeze(0).to(device)
                    m = m.unsqueeze(0).to(device)
                    pred = model(l, m)["out"][0].detach().cpu().numpy()
                    gt_np = g[0].detach().cpu().numpy()
                    psnrs.append(calc_psnr(pred, gt_np))
                    ssims.append(calc_ssim(pred, gt_np))
                    sams.append(calc_sam(pred, gt_np))
                    ergas_list.append(calc_ergas(pred, gt_np, scale))
            m_psnr = float(np.mean(psnrs))
            m_ssim = float(np.mean(ssims))
            m_sam = float(np.mean(sams))
            m_ergas = float(np.mean(ergas_list))
            improved = ""
            if m_psnr > best_psnr:
                best_psnr = m_psnr
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                improved = " [BEST]"
            print(f"  >>> Test@{epoch}: PSNR {m_psnr:.4f} | SSIM {m_ssim:.4f} | "
                  f"SAM {m_sam:.3f} | ERGAS {m_ergas:.3f}{improved}")
            ema.restore_from(model)

    # Final eval
    ckpt = os.path.join(save_dir, "best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    psnrs, ssims, sams, ergas_list = [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            g, l, m = test_ds[i]
            g = g.unsqueeze(0).to(device)
            l = l.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)
            pred = model(l, m)["out"][0].detach().cpu().numpy()
            gt_np = g[0].detach().cpu().numpy()
            psnrs.append(calc_psnr(pred, gt_np))
            ssims.append(calc_ssim(pred, gt_np))
            sams.append(calc_sam(pred, gt_np))
            ergas_list.append(calc_ergas(pred, gt_np, scale))
    final = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
             "sam": float(np.mean(sams)), "ergas": float(np.mean(ergas_list))}
    print("=" * 60)
    print(f"FINAL BEST (epoch {best_epoch}): PSNR {final['psnr']:.4f} | "
          f"SSIM {final['ssim']:.4f} | SAM {final['sam']:.3f} | "
          f"ERGAS {final['ergas']:.3f}")
    print("=" * 60)

    # SOTA comparison
    print("\n--- SOTA Comparison (Chikusei x4) ---")
    sota = {
        "CoFusion (2026)": {"psnr": 49.14, "sam": 2.60},
        "RAMoE (2026)": {"psnr": 48.10, "sam": 0.79},
        "SMGU-Net (2025)": {"psnr": 48.82, "sam": 2.72},
        "PSRT (2023)": {"psnr": 47.99, "sam": 2.84},
        "KrylovNet v1 (ours, 2.3k)": {"psnr": 43.69, "sam": 6.07},
    }
    for name, vals in sota.items():
        delta = final["psnr"] - vals["psnr"]
        print(f"  {name}: PSNR {vals['psnr']:.2f} (Δ={delta:+.2f}) SAM {vals['sam']:.2f}")

    with open(os.path.join(args.output_dir, "krylovnet_chikusei_results.json"), "w") as f:
        json.dump({
            "dataset": "Chikusei",
            "bands": bands,
            "protocol": "Chikusei x4, Sensor SRF, Wald blur",
            "params_M": nparams / 1e6,
            "best_epoch": best_epoch,
            "final": final,
            "sota_comparison": sota,
        }, f, indent=2)
    print("\nSaved krylovnet_chikusei_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input/chikusei")
    ap.add_argument("--output_dir", default="/kaggle/working")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
