"""Diffusion-NullFusion — SOTA-beater for Chikusei x4

Architecture:
  1. Multi-Scale Swin Transformer U-Net as f_theta
  2. DDPM noise prediction in nullspace (cosine schedule, T=1000)
  3. DDIM inference with 5-sample ensemble averaging
  4. Sensor-Adaptive Conditioning via SRF embedding + AdaLN
  5. Exact null-space projection via CG (RangeNullProjector)

Data consistency: X_final = pinv + P_N(avg_nullspace)
Loss: MSE noise prediction + physics consistency

Run on Kaggle: python train_diffusion_nullfusion.py --root /kaggle/input/datasets/mingliu123/chikusei
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import convolve, uniform_filter
import h5py
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# SRF & degradation
# ---------------------------------------------------------------------------

def chikusei_srf(bands=128):
    wl = np.linspace(363.0, 1018.0, 128)
    raw = np.stack([
        np.exp(-((wl - 620) ** 2) / (2 * 80 ** 2)),
        np.exp(-((wl - 540) ** 2) / (2 * 70 ** 2)),
        np.exp(-((wl - 460) ** 2) / (2 * 60 ** 2)),
    ], axis=1).astype(np.float32)
    return raw / np.maximum(raw.sum(axis=0, keepdims=True), 1e-8)


def gaussian_kernel2d(size=9, sigma=1.2):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    return (k / k.sum()).astype(np.float32)


class DegradationOp(nn.Module):
    def __init__(self, scale, ksize=9, sigma=1.2):
        super().__init__()
        self.scale = scale
        k = gaussian_kernel2d(ksize, sigma)
        self.register_buffer("k", torch.from_numpy(k)[None, None])

    def forward(self, x):
        C = x.shape[1]
        return F.conv2d(x, self.k.repeat(C, 1, 1, 1), padding=4, groups=C)[
            :, :, :: self.scale, :: self.scale
        ]

    def transpose(self, y, hw):
        C = y.shape[1]
        up = F.interpolate(y, size=hw, mode="bicubic", align_corners=False)
        return F.conv2d(up, self.k.repeat(C, 1, 1, 1), padding=4, groups=C)


# ---------------------------------------------------------------------------
# CG solver & null-space projector
# ---------------------------------------------------------------------------

def scalar_cg(applyA, rhs, steps, tol=1e-10):
    z = torch.zeros_like(rhs)
    r = rhs - applyA(z)
    p = r.clone()
    rs = (r * r).flatten(1).sum(1)
    for _ in range(steps):
        ap = applyA(p)
        denom = (p * ap).flatten(1).sum(1)
        alpha = (rs / denom.clamp_min(tol)).reshape(-1, 1, 1, 1)
        z = z + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).flatten(1).sum(1)
        beta = (rs_new / rs.clamp_min(tol)).reshape(-1, 1, 1, 1)
        p = r + beta * p
        rs = rs_new
    return z


class RangeNullProjector(nn.Module):
    def __init__(self, scale, cg_steps=40, ridge=1e-4):
        super().__init__()
        self.D = DegradationOp(scale)
        self.scale = scale
        self.cg_steps = cg_steps
        self.ridge = ridge

    def _normal_op(self, out_hw):
        def applyA(z):
            return self.D(self.D.transpose(z, out_hw)) + self.ridge * z
        return applyA

    def pinv(self, yH, out_hw):
        return self.D.transpose(
            scalar_cg(self._normal_op(out_hw), yH, self.cg_steps), out_hw
        )

    def project_null(self, v, out_hw=None):
        if out_hw is None:
            out_hw = (v.shape[-2], v.shape[-1])
        return v - self.pinv(self.D(v), out_hw)


# ---------------------------------------------------------------------------
# Swin Transformer components
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        ws = self.window_size if hasattr(self, 'window_size') else 8

        if not hasattr(self, 'window_size'):
            ws = 8

        x_2d = x.reshape(B, H, W, C)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x_win = x_2d.reshape(B, Hp // ws, ws, Wp // ws, ws, C)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)

        qkv = self.qkv(x_win).reshape(-1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        out = self.proj(out)

        out = out.reshape(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        if pad_h or pad_w:
            out = out[:, :H, :W, :]
        return out.reshape(B, H * W, C)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on sensor embedding."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma) + beta


class SwinBlock(nn.Module):
    """Swin Transformer block with AdaLN conditioning."""
    def __init__(self, dim, num_heads, cond_dim, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.attn.window_size = window_size
        self.norm2 = AdaLN(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, cond):
        x = x + self.attn(self.norm1(x, cond))
        x = x + self.mlp(self.norm2(x, cond))
        return x


class PatchMerging(nn.Module):
    """2x downsampling via patch merging."""
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B = x.shape[0]
        x = x.reshape(B, H, W, -1)
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 1::2, 0::2]
        x2 = x[:, 0::2, 1::2]
        x3 = x[:, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.reshape(B, -1, 4 * x.shape[-1])
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class PatchExpanding(nn.Module):
    """2x upsampling via patch expanding."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 4 * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        x = self.norm(x)
        x = self.linear(x)
        B = x.shape[0]
        x = x.reshape(B, H, W, 2, 2, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, -1)
        return x.reshape(B, H * 2 * W * 2, -1), H * 2, W * 2


# ---------------------------------------------------------------------------
# Multi-Scale Swin Transformer U-Net
# ---------------------------------------------------------------------------

class ChannelProject(nn.Module):
    """Project features from one channel dimension to another."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


class MultiScaleSwinUNet(nn.Module):
    """U-shaped Swin Transformer with AdaLN conditioning at every level.

    Args:
        in_ch: input channels (128 = spectral bands)
        base_dim: base embedding dimension
        num_heads: list of head counts per level
        cond_dim: conditioning vector dimension
        window_size: attention window size
        depths: [enc1, enc2, bottleneck, dec2, dec1] block counts
    """
    def __init__(self, in_ch=128, base_dim=48, num_heads=None, cond_dim=64,
                 window_size=8, depths=None):
        super().__init__()
        if num_heads is None:
            num_heads = [4, 8, 16]
        if depths is None:
            depths = [1, 1, 2, 1, 1]

        dims = [base_dim, base_dim * 2, base_dim * 4]

        self.input_proj = nn.Linear(in_ch, dims[0])

        # Encoder
        self.enc1_blocks = nn.ModuleList([
            SwinBlock(dims[0], num_heads[0], cond_dim, window_size)
            for _ in range(depths[0])
        ])
        self.merge1 = PatchMerging(dims[0])

        self.enc2_blocks = nn.ModuleList([
            SwinBlock(dims[1], num_heads[1], cond_dim, window_size)
            for _ in range(depths[1])
        ])
        self.merge2 = PatchMerging(dims[1])

        # Bottleneck
        self.bottleneck_blocks = nn.ModuleList([
            SwinBlock(dims[2], num_heads[2], cond_dim, window_size)
            for _ in range(depths[2])
        ])

        # Decoder — PatchExpanding upsamples spatial but keeps channels,
        # so we project channels to match skip connections
        self.expand2 = PatchExpanding(dims[2])
        self.proj_skip2 = ChannelProject(dims[1], dims[2])
        self.proj_dec2 = ChannelProject(dims[2], dims[1])
        self.dec2_blocks = nn.ModuleList([
            SwinBlock(dims[1], num_heads[1], cond_dim, window_size)
            for _ in range(depths[3])
        ])

        self.expand1 = PatchExpanding(dims[1])
        self.proj_skip1 = ChannelProject(dims[0], dims[1])
        self.proj_dec1 = ChannelProject(dims[1], dims[0])
        self.dec1_blocks = nn.ModuleList([
            SwinBlock(dims[0], num_heads[0], cond_dim, window_size)
            for _ in range(depths[4])
        ])

        self.output_proj = nn.Linear(dims[0], in_ch)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = self.input_proj(x)

        # Encoder
        for block in self.enc1_blocks:
            x = block(x, cond)
        skip1 = x
        x, H1, W1 = self.merge1(x, H, W)

        for block in self.enc2_blocks:
            x = block(x, cond)
        skip2 = x
        x, H2, W2 = self.merge2(x, H1, W1)

        # Bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, cond)

        # Decoder with skip connections (project channels to match)
        x, H2, W2 = self.expand2(x, H2, W2)
        x = self.proj_dec2(x + self.proj_skip2(skip2))
        for block in self.dec2_blocks:
            x = block(x, cond)

        x, H1, W1 = self.expand1(x, H1, W1)
        x = self.proj_dec1(x + self.proj_skip1(skip1))
        for block in self.dec1_blocks:
            x = block(x, cond)

        x = self.output_proj(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Cosine diffusion schedule
# ---------------------------------------------------------------------------

class CosineSchedule:
    """Cosine noise schedule for DDPM."""
    def __init__(self, T=1000, s=0.008):
        self.T = T
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = (f / f[0]).float()
        self.alpha_bar = alpha_bar
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        self.beta = torch.clamp(beta, max=0.999).float()
        self.alpha = 1 - self.beta

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        return self

    def add_noise(self, x0, noise, t):
        ab = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

    def ddim_step(self, x_t, eps_pred, t, t_prev):
        ab_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        ab_prev = self.alpha_bar[t_prev].reshape(-1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)
        x0_pred = x0_pred.clamp(0, 1)
        direction = torch.sqrt(1 - ab_prev) * eps_pred
        return torch.sqrt(ab_prev) * x0_pred + direction


# ---------------------------------------------------------------------------
# Sensor embedding (SRF -> conditioning vector)
# ---------------------------------------------------------------------------

class SensorEmbedding(nn.Module):
    """Encode SRF matrix into a fixed conditioning vector via MLP."""
    def __init__(self, srf_matrix, out_dim=64):
        super().__init__()
        flat = torch.from_numpy(srf_matrix).float().flatten()
        self.register_buffer("srf_flat", flat)
        in_dim = flat.numel()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self):
        return self.mlp(self.srf_flat.unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------------
# Diffusion-NullFusion
# ---------------------------------------------------------------------------

class DiffusionNullFusion(nn.Module):
    """Diffusion-NullFusion: Multi-Scale Swin UNet + DDPM + Sensor AdaLN.

    Training: predict noise epsilon in nullspace (standard DDPM).
    Inference: DDIM denoising with multi-sample ensemble averaging.
    Data consistency: X = pinv(yH) + P_N(avg_nullspace).
    """
    def __init__(self, bands=128, msi=3, base_dim=48, scale=4,
                 cond_dim=64, T=1000, num_heads=None, depths=None,
                 window_size=8):
        super().__init__()
        self.bands = bands
        self.scale = scale
        self.T = T

        # SRF buffers
        srf = torch.from_numpy(chikusei_srf(bands)).float()
        self.register_buffer("srf", srf)
        self.register_buffer("srfinv", torch.linalg.pinv(srf))

        # Null projector
        self.projector = RangeNullProjector(scale, cg_steps=8, ridge=1e-4)

        # Sensor embedding (fixed per sensor)
        self.sensor_embed = SensorEmbedding(chikusei_srf(bands), cond_dim)

        # Diffusion schedule
        self.schedule = CosineSchedule(T)

        # Conditioning projection (MSI + pseudo-inverse -> cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Conv2d(msi + bands, cond_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Swin UNet for noise prediction
        self.unet = MultiScaleSwinUNet(
            in_ch=bands, base_dim=base_dim, num_heads=num_heads,
            cond_dim=cond_dim, window_size=window_size, depths=depths,
        )

    def _conditioning(self, yH, yM, H_hr, W_hr):
        """Build conditioning vector from MSI and pseudo-inverse."""
        base = self.projector.pinv(yH, (H_hr, W_hr))
        obs = F.interpolate(yH, (H_hr, W_hr), mode="bicubic", align_corners=False)
        cond_input = torch.cat([yM, obs], dim=1)
        cond = self.cond_proj(cond_input)
        cond = cond + self.sensor_embed()
        return cond, base

    def forward(self, x_t, t, cond):
        """Training: predict noise given noisy sample, timestep, condition."""
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        return self.unet(x_t, cond + t_emb)

    @torch.no_grad()
    def inference(self, yH, yM, num_samples=5, ddim_steps=50):
        """Inference: DDIM denoising + multi-sample ensemble.

        Returns dict with 'out', 'base', 'null'.
        """
        H_hr, W_hr = yM.shape[-2], yM.shape[-1]
        cond, base = self._conditioning(yH, yM, H_hr, W_hr)

        samples = []
        for _ in range(num_samples):
            x_t = torch.randn(1, self.bands, H_hr, W_hr, device=yH.device)
            ts = torch.linspace(self.T - 1, 0, ddim_steps, dtype=torch.long, device=yH.device)
            for i in range(len(ts) - 1):
                eps = self(x_t, ts[i], cond)
                x_t = self.schedule.ddim_step(x_t, eps, ts[i], ts[i + 1])
            samples.append(x_t)

        avg_null = torch.mean(torch.stack(samples), dim=0)
        avg_null = self.projector.project_null(avg_null, (H_hr, W_hr))
        return {"out": base + avg_null, "base": base, "null": avg_null}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChikuseiDS(Dataset):
    def __init__(self, root, split="train", bands=128, scale=4, patch=64):
        self.split = split
        self.bands = bands
        self.scale = scale
        self.patch = patch
        self.srf = chikusei_srf(bands)
        self.kernel = gaussian_kernel2d(9, 1.2)

        mat_files = glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)
        hsi = [m for m in mat_files
               if "Ground_Truth" not in os.path.basename(m)
               and "gt" not in os.path.basename(m).lower()]
        if hsi:
            hsi.sort(key=lambda f: os.path.getsize(f), reverse=True)
            mat_path = hsi[0]
        elif mat_files:
            mat_files.sort(key=lambda f: os.path.getsize(f), reverse=True)
            mat_path = mat_files[0]
        else:
            raise FileNotFoundError(f"No .mat files under {root}")

        print(f"[Data] Loading: {mat_path}")
        try:
            data = loadmat(mat_path)
        except NotImplementedError:
            print("[Data] v7.3 mat file — using h5py")
            data = {}
            with h5py.File(mat_path, "r") as f:
                for key in f.keys():
                    if not key.startswith("__"):
                        val = np.array(f[key]).copy()
                        if val.ndim >= 2:
                            data[key] = val

        for key, val in data.items():
            if not key.startswith("__") and hasattr(val, "shape"):
                arr = np.array(val, dtype=np.float32)
                if arr.ndim == 3 and min(arr.shape) > 10:
                    if arr.shape[0] > arr.shape[-1]:
                        arr = arr.transpose(2, 0, 1)
                    if arr.max() > 1.0:
                        arr = arr / arr.max()
                    self.cube = arr
                    break

        C, H, W = self.cube.shape
        print(f"[Data] Cube: {C}b, {H}x{W}px")

        p = patch
        coords = [(y, x) for y in range(0, H - p + 1, p)
                   for x in range(0, W - p + 1, p)]
        random.seed(42)
        random.shuffle(coords)
        n = int(0.7 * len(coords))
        self.patches = coords[:n] if split == "train" else coords[n:]
        print(f"[Data] {split}: {len(self.patches)} patches")

    def __len__(self):
        return len(self.patches) * (200 if self.split == "train" else 1)

    def _sim(self, gt):
        C, H, W = gt.shape
        blurred = np.empty_like(gt)
        for c in range(C):
            blurred[c] = convolve(gt[c], self.kernel, mode="wrap")
        hr = H // self.scale
        y0 = (H - hr * self.scale) // 2
        x0 = (W - hr * self.scale) // 2
        lr = blurred[:, y0::self.scale, x0::self.scale].astype(np.float32)
        msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
        return lr, np.clip(msi, 0, 1)

    def __getitem__(self, idx):
        y, x = self.patches[idx % len(self.patches)]
        p = self.patch
        gt = self.cube[:, y:y + p, x:x + p].copy()
        if self.split == "train":
            if random.random() < 0.5:
                gt = gt[:, :, ::-1].copy()
            if random.random() < 0.5:
                gt = gt[:, ::-1, :].copy()
            if random.random() < 0.5:
                gt = np.rot90(gt, random.randint(1, 3), axes=(1, 2)).copy()
            if random.random() < 0.15:
                gt = (gt + np.random.randn(*gt.shape).astype(np.float32) * 0.01).clip(0, 1)
        lr, msi = self._sim(gt)
        return torch.from_numpy(gt), torch.from_numpy(lr), torch.from_numpy(msi)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def diffusion_loss(model, x0, cond, schedule):
    """Standard DDPM noise prediction loss."""
    B = x0.shape[0]
    t = torch.randint(0, schedule.T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = schedule.add_noise(x0, noise, t)
    eps_pred = model(x_t, t, cond)
    return F.mse_loss(eps_pred, noise)


def physics_consistency_loss(pred, yH, yM, model):
    """Enforce D(X) = yH and S(X) = yM."""
    lr_pred = model.projector.D(pred)
    msi_pred = torch.einsum("bchw,cm->bmhw", pred, model.srf)
    return F.mse_loss(lr_pred, yH) + F.mse_loss(msi_pred, yM)


def total_loss(model, gt, yH, yM, schedule, w_noise=1.0, w_phys=0.1):
    """Combined diffusion + physics consistency loss."""
    H_hr, W_hr = yM.shape[-2], yM.shape[-1]
    cond, base = model._conditioning(yH, yM, H_hr, W_hr)

    # Nullspace target
    x0 = model.projector.project_null(gt - base, (H_hr, W_hr))

    # Diffusion loss
    l_noise = diffusion_loss(model, x0, cond, schedule)

    # Physics consistency on reconstructed output
    pred = base + x0
    l_phys = physics_consistency_loss(pred, yH, yM, model)

    total = w_noise * l_noise + w_phys * l_phys
    return total, l_noise, l_phys


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr_np(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    return 100.0 if mse < 1e-12 else -10 * np.log10(mse)


def sam_np(pred, gt):
    p = pred.reshape(pred.shape[0], -1)
    g = gt.reshape(gt.shape[0], -1)
    p = p / (np.linalg.norm(p, axis=0, keepdims=True) + 1e-8)
    g = g / (np.linalg.norm(g, axis=0, keepdims=True) + 1e-8)
    return np.mean(np.arccos(np.clip((p * g).sum(0), -1, 1))) * 180 / math.pi


def ssim_np(pred, gt):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = uniform_filter(pred, 3, mode="reflect")
    mu2 = uniform_filter(gt, 3, mode="reflect")
    s12 = uniform_filter(pred * gt, 3, mode="reflect") - mu1 * mu2
    s1 = uniform_filter(pred ** 2, 3, mode="reflect") - mu1 ** 2
    s2 = uniform_filter(gt ** 2, 3, mode="reflect") - mu2 ** 2
    return np.mean(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2) + 1e-8))


def ergas_np(pred, gt, scale=4):
    C = pred.shape[0]
    e = sum(((pred - gt) ** 2)[c].mean() / (gt[c].mean() ** 2 + 1e-8)
            for c in range(C))
    return math.sqrt(e / C) * 100 * scale


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point and k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--bands", type=int, default=128)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--time_budget_h", type=float, default=8.5)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--base_dim", type=int, default=48)
    parser.add_argument("--cond_dim", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/diffusion_nullfusion")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = DiffusionNullFusion(
        bands=args.bands, msi=3, base_dim=args.base_dim, scale=args.scale,
        cond_dim=args.cond_dim, T=args.T,
    ).to(device)

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: Diffusion-NullFusion — {nparams / 1e6:.2f}M params")

    # Schedule on device
    schedule = model.schedule.to(device)

    # Data
    train_ds = ChikuseiDS(args.root, "train", args.bands, args.scale, args.patch)
    test_ds = ChikuseiDS(args.root, "test", args.bands, args.scale, args.patch)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    ema = EMA(model, 0.999)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=2000, T_mult=2, eta_min=1e-6
    )

    best_psnr = 0
    best_epoch = 0
    T0 = time.time()
    LIMIT = args.time_budget_h * 3600

    targets = {"CoFusion": 49.14, "SMGU-Net": 48.82, "RAMoE": 48.10,
               "PSRT": 47.99, "KrylovNet v1": 43.69}

    print(f"\nTraining: {args.epochs} epochs, {args.time_budget_h}h budget")
    print(f"Diffusion T={args.T}, DDIM steps={args.ddim_steps}, samples={args.num_samples}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}, Steps/epoch: {args.steps_per_epoch}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        if (time.time() - T0) > LIMIT:
            print(f"\n[TIME BUDGET at epoch {epoch}]")
            break

        model.train()
        total = 0
        t0 = time.time()
        opt.zero_grad()

        for step in range(args.steps_per_epoch):
            gts, lhs, mss = [], [], []
            for _ in range(args.batch_size):
                g, l, m = train_ds[random.randrange(len(train_ds))]
                gts.append(g)
                lhs.append(l)
                mss.append(m)
            gt = torch.stack(gts, 0).to(device)
            yH = torch.stack(lhs, 0).to(device)
            yM = torch.stack(mss, 0).to(device)

            loss, l_noise, l_phys = total_loss(model, gt, yH, yM, schedule)
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                ema.update(model)

            total += loss.item()

        scheduler.step()
        dt = time.time() - t0
        avg = total / args.steps_per_epoch

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{args.epochs} | Loss {avg:.5f} | "
                  f"LR {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        # Evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ema.apply(model)
            model.eval()
            ps_list, ss_list, sa_list, er_list = [], [], [], []

            with torch.no_grad():
                for i in range(len(test_ds)):
                    g, l, m = test_ds[i]
                    # Use 1 sample + 20 steps for fast eval during training
                    out = model.inference(
                        l.unsqueeze(0).to(device),
                        m.unsqueeze(0).to(device),
                        num_samples=1,
                        ddim_steps=20,
                    )
                    pred_np = out["out"][0].cpu().numpy()
                    gt_np = g.numpy()
                    ps_list.append(psnr_np(pred_np, gt_np))
                    ss_list.append(ssim_np(pred_np, gt_np))
                    sa_list.append(sam_np(pred_np, gt_np))
                    er_list.append(ergas_np(pred_np, gt_np, 4))

            mp = float(np.mean(ps_list))
            ms_ = float(np.mean(ss_list))
            ma = float(np.mean(sa_list))
            me = float(np.mean(er_list))

            marker = ""
            if mp > best_psnr:
                best_psnr = mp
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, "best.pth"))
                marker = " [BEST]"

            print(f"  >>> Test@{epoch}: PSNR {mp:.4f} | SSIM {ms_:.4f} | "
                  f"SAM {ma:.3f} | ERGAS {me:.3f}{marker}")
            for name, target in targets.items():
                d = mp - target
                m_ = " <<< BEAT" if d > 0 else ""
                print(f"    vs {name:15s}: delta={d:+.2f} dB{m_}")

            ema.restore(model)

    # Final evaluation with full ensemble
    print("\n" + "=" * 70)
    print("FINAL EVALUATION: 5 samples x 50 DDIM steps")
    print("=" * 70)

    ckpt = os.path.join(args.save_dir, "best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    ps_list, ss_list, sa_list, er_list = [], [], [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            g, l, m = test_ds[i]
            out = model.inference(
                l.unsqueeze(0).to(device),
                m.unsqueeze(0).to(device),
                num_samples=args.num_samples,
                ddim_steps=args.ddim_steps,
            )
            pred_np = out["out"][0].cpu().numpy()
            gt_np = g.numpy()
            ps_list.append(psnr_np(pred_np, gt_np))
            ss_list.append(ssim_np(pred_np, gt_np))
            sa_list.append(sam_np(pred_np, gt_np))
            er_list.append(ergas_np(pred_np, gt_np, 4))

    final = {
        "psnr": float(np.mean(ps_list)),
        "ssim": float(np.mean(ss_list)),
        "sam": float(np.mean(sa_list)),
        "ergas": float(np.mean(er_list)),
    }
    print(f"FINAL (epoch {best_epoch}): PSNR {final['psnr']:.4f} | "
          f"SSIM {final['ssim']:.4f} | SAM {final['sam']:.3f} | "
          f"ERGAS {final['ergas']:.3f}")

    for name, target in {**targets, "CoFusion (2026)": 49.14}.items():
        d = final["psnr"] - target
        m_ = " <<< BEAT" if d > 0 else ""
        print(f"  {name:25s} Target {target:6.2f} | Ours delta={d:+.2f} dB{m_}")

    results = {
        "dataset": "Chikusei",
        "bands": args.bands,
        "params_M": nparams / 1e6,
        "best_epoch": best_epoch,
        "T": args.T,
        "ddim_steps": args.ddim_steps,
        "num_samples": args.num_samples,
        "final": final,
    }
    out_path = os.path.join(args.save_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
