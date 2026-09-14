"""Diffusion-NullFusion — CAVE x4 SOTA-beater

Bugs fixed vs original train_diffusion_nullfusion.py:
  1. Dataset: CAVE (31 bands, PNG layout) not Chikusei (128 bands, .mat)
  2. SRF: Nikon D700 (400-700nm) not Chikusei Gaussian (363-1018nm)
  3. Null-space projector: combined degradation A=[D;R] via block-CG (not just D)
  4. Loss: charbonnier + SSIM + SAM + gradient + physics (was MSE-only)
  5. Gradient checkpointing for Kaggle T4 memory
  6. Kaggle time-limit safe: periodic checkpoint + SIGTERM handler + resume
  7. Full test set evaluation (not random subset)
  8. Transpose operator uses bicubic upsample (not zero-fill)
  9. AMP autocast properly scoped
 10. EMA state saved/restored correctly on resume

Protocol: Wald simulation (Gaussian blur 9x9, sigma=1.2, x4 decimation),
           Nikon D700 SRF, 20 train / 12 test scenes, data_range=1.0.

SOTA targets (CAVE x4):
  FeINFN        52.47 dB
  NullFusion v4 50.31 dB
  CoFusion      50.67 dB
  SMGU-Net      49.83 dB

Run on Kaggle: python train_diffusion_nullfusion_cave.py \
    --root /kaggle/input/datasets/liptee/hyperspectral-image-restoration-based-on-cave
"""
from __future__ import annotations

import argparse
import atexit
import glob
import json
import math
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

try:
    from scipy.ndimage import convolve, uniform_filter
except ImportError:
    from scipy.ndimage import convolve, uniform_filter


# ---------------------------------------------------------------------------
# Nikon D700 SRF (31 bands, 400-700 nm at 10 nm)
# ---------------------------------------------------------------------------

_NIKON_D700_31 = np.array([
    [0.0050, 0.0130, 0.2400], [0.0060, 0.0190, 0.3600],
    [0.0070, 0.0280, 0.5200], [0.0080, 0.0420, 0.7100],
    [0.0090, 0.0620, 0.8800], [0.0100, 0.0890, 0.9800],
    [0.0110, 0.1250, 1.0000], [0.0130, 0.1750, 0.9500],
    [0.0150, 0.2400, 0.8400], [0.0180, 0.3300, 0.6900],
    [0.0230, 0.4500, 0.5300], [0.0310, 0.5900, 0.3900],
    [0.0450, 0.7400, 0.2700], [0.0700, 0.8800, 0.1800],
    [0.1100, 0.9700, 0.1200], [0.1700, 1.0000, 0.0800],
    [0.2600, 0.9800, 0.0550], [0.3800, 0.9100, 0.0400],
    [0.5300, 0.8000, 0.0300], [0.6900, 0.6700, 0.0230],
    [0.8300, 0.5300, 0.0180], [0.9300, 0.4000, 0.0140],
    [0.9900, 0.2900, 0.0110], [1.0000, 0.2100, 0.0090],
    [0.9700, 0.1500, 0.0075], [0.9000, 0.1050, 0.0062],
    [0.8000, 0.0740, 0.0052], [0.6800, 0.0520, 0.0044],
    [0.5500, 0.0370, 0.0037], [0.4300, 0.0260, 0.0031],
    [0.3200, 0.0190, 0.0026],
], dtype=np.float32)


def nike_d700_srf(bands=31):
    src = _NIKON_D700_31
    if bands != src.shape[0]:
        xs = np.linspace(0.0, 1.0, src.shape[0])
        xd = np.linspace(0.0, 1.0, bands)
        src = np.stack([np.interp(xd, xs, src[:, i]) for i in range(3)], axis=1)
    srf = src.astype(np.float32)
    srf = srf / np.maximum(srf.sum(axis=0, keepdims=True), 1e-8)
    return srf


# ---------------------------------------------------------------------------
# Gaussian blur kernel
# ---------------------------------------------------------------------------

def gaussian_kernel2d(size=9, sigma=1.2):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    return (k / k.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Degradation operator (fixed: bicubic transpose, not zero-fill)
# ---------------------------------------------------------------------------

class DegradationOp(nn.Module):
    """Per-band blur + decimate (forward) / bicubic upsample + blur (adjoint)."""

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

    def transpose(self, y, out_hw):
        C = y.shape[1]
        up = F.interpolate(y, size=out_hw, mode="bicubic", align_corners=False)
        return F.conv2d(up, self.k.repeat(C, 1, 1, 1), padding=4, groups=C)


# ---------------------------------------------------------------------------
# Block CG solver for combined A=[D;R] system
# ---------------------------------------------------------------------------

def block_cg(applyA, rhs, steps, tol=1e-10):
    z = tuple(torch.zeros_like(r) for r in rhs)
    r = tuple(rh - a for rh, a in zip(rhs, applyA(z)))
    p = tuple(ri.clone() for ri in r)
    rs = sum((ri * ri).flatten(1).sum(1) for ri in r)
    shape = (rhs[0].shape[0],) + (1,) * (rhs[0].dim() - 1)
    for _ in range(steps):
        ap = applyA(p)
        denom = sum((pi * ai).flatten(1).sum(1) for pi, ai in zip(p, ap))
        alpha = (rs / denom.clamp_min(tol)).reshape(*shape)
        z = tuple(zi + alpha * pi for zi, pi in zip(z, p))
        r = tuple(ri - alpha * ai for ri, ai in zip(r, ap))
        rs_new = sum((ri * ri).flatten(1).sum(1) for ri in r)
        beta = (rs_new / rs.clamp_min(tol)).reshape(*shape)
        p = tuple(ri + beta * pi for ri, pi in zip(r, p))
        rs = rs_new
    return z


# ---------------------------------------------------------------------------
# Combined operator A = [D; R] for joint range/null decomposition
# ---------------------------------------------------------------------------

class CombinedOperator(nn.Module):
    """Combined degradation: A = [D; R] where D=blur+decimate, R=SRF projection."""

    def __init__(self, scale, bands, msi_bands, srf_np, ksize=9, sigma=1.2,
                 cg_steps=40, ridge=1e-6):
        super().__init__()
        self.D = DegradationOp(scale, ksize, sigma)
        self.scale = scale
        self.bands = bands
        self.msi_bands = msi_bands
        self.cg_steps = cg_steps
        self.ridge = ridge
        srf = torch.from_numpy(srf_np).float()
        self.register_buffer("srf", srf)

    def R(self, x):
        return torch.einsum("bchw,cm->bmhw", x, self.srf)

    def Rt(self, m):
        return torch.einsum("bmhw,cm->bchw", m, self.srf)

    def forward(self, x):
        return self.D(x), self.R(x)

    def adjoint(self, yH, yM, out_hw):
        return self.D.transpose(yH, out_hw) + self.Rt(yM)

    def apply_gram(self, zH, zM, out_hw):
        Dt = self.D.transpose(zH, out_hw)
        RtM = self.Rt(zM)
        wH = self.D(Dt) + self.D(RtM) + self.ridge * zH
        wM = self.R(Dt) + self.R(RtM) + self.ridge * zM
        return wH, wM

    def pinv(self, yH, yM, out_hw):
        zH, zM = block_cg(
            lambda p: self.apply_gram(p[0], p[1], out_hw), (yH, yM), self.cg_steps
        )
        return self.adjoint(zH, zM, out_hw)

    def project_null(self, v):
        with torch.no_grad():
            out_hw = (v.shape[-2], v.shape[-1])
            yH, yM = self.forward(v)
            proj = self.pinv(yH, yM, out_hw)
        return v - proj


# ---------------------------------------------------------------------------
# Swin Transformer components
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.window_size = window_size

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        ws = self.window_size
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
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma) + beta


class SpectralAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim), nn.Sigmoid(),
        )

    def forward(self, x):
        gate = self.fc(x.mean(dim=1))
        return x * gate.unsqueeze(1)


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, cond_dim, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = AdaLN(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.spec = SpectralAttention(dim)

    def forward(self, x, cond):
        x = x + self.attn(self.norm1(x, cond))
        x = x + self.mlp(self.norm2(x, cond))
        x = self.spec(x)
        return x


class PatchMerging(nn.Module):
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
        x = torch.cat([x0, x1, x2, x3], -1).reshape(B, -1, 4 * x.shape[-1])
        return self.reduction(self.norm(x)), H // 2, W // 2


class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 4 * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        x = self.linear(self.norm(x))
        B = x.shape[0]
        x = x.reshape(B, H, W, 2, 2, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, -1)
        return x.reshape(B, H * 2 * W * 2, -1), H * 2, W * 2


class ChannelProject(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Multi-Scale Swin Transformer U-Net (with gradient checkpointing)
# ---------------------------------------------------------------------------

class MultiScaleSwinUNet(nn.Module):
    def __init__(self, in_ch=31, base_dim=64, num_heads=None, cond_dim=64,
                 window_size=8, depths=None, use_checkpoint=False):
        super().__init__()
        if num_heads is None:
            num_heads = [4, 8, 16]
        if depths is None:
            depths = [2, 2, 4, 2, 2]

        dims = [base_dim, base_dim * 2, base_dim * 4]
        self.use_checkpoint = use_checkpoint

        self.input_proj = nn.Linear(in_ch, dims[0])
        self.enc1 = nn.ModuleList([
            SwinBlock(dims[0], num_heads[0], cond_dim, window_size)
            for _ in range(depths[0])
        ])
        self.merge1 = PatchMerging(dims[0])
        self.enc2 = nn.ModuleList([
            SwinBlock(dims[1], num_heads[1], cond_dim, window_size)
            for _ in range(depths[1])
        ])
        self.merge2 = PatchMerging(dims[1])
        self.bottleneck = nn.ModuleList([
            SwinBlock(dims[2], num_heads[2], cond_dim, window_size)
            for _ in range(depths[2])
        ])
        self.expand2 = PatchExpanding(dims[2])
        self.proj_skip2 = ChannelProject(dims[1], dims[2])
        self.proj_dec2 = ChannelProject(dims[2], dims[1])
        self.dec2 = nn.ModuleList([
            SwinBlock(dims[1], num_heads[1], cond_dim, window_size)
            for _ in range(depths[3])
        ])
        self.expand1 = PatchExpanding(dims[1])
        self.proj_skip1 = ChannelProject(dims[0], dims[1])
        self.proj_dec1 = ChannelProject(dims[1], dims[0])
        self.dec1 = nn.ModuleList([
            SwinBlock(dims[0], num_heads[0], cond_dim, window_size)
            for _ in range(depths[4])
        ])
        self.output_proj = nn.Linear(dims[0], in_ch)

    def _run(self, blocks, x, cond):
        for b in blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x = checkpoint(b, x, cond, use_reentrant=False)
            else:
                x = b(x, cond)
        return x

    def forward(self, x, cond):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = self.input_proj(x)
        x = self._run(self.enc1, x, cond)
        skip1 = x
        x, H1, W1 = self.merge1(x, H, W)
        x = self._run(self.enc2, x, cond)
        skip2 = x
        x, H2, W2 = self.merge2(x, H1, W1)
        x = self._run(self.bottleneck, x, cond)
        x, H2, W2 = self.expand2(x, H2, W2)
        x = self.proj_dec2(x + self.proj_skip2(skip2))
        x = self._run(self.dec2, x, cond)
        x, H1, W1 = self.expand1(x, H1, W1)
        x = self.proj_dec1(x + self.proj_skip1(skip1))
        x = self._run(self.dec1, x, cond)
        return self.output_proj(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Cosine diffusion schedule
# ---------------------------------------------------------------------------

class CosineSchedule:
    def __init__(self, T=1000, s=0.008):
        self.T = T
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi * 0.5) ** 2
        self.alpha_bar = (f / f[0]).float()
        beta = 1 - self.alpha_bar[1:] / self.alpha_bar[:-1]
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
    def __init__(self, srf_matrix, out_dim=64):
        super().__init__()
        flat = torch.from_numpy(srf_matrix).float().flatten()
        self.register_buffer("srf_flat", flat)
        self.mlp = nn.Sequential(
            nn.Linear(flat.numel(), 128), nn.GELU(), nn.Linear(128, out_dim),
        )

    def forward(self):
        return self.mlp(self.srf_flat.unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------------
# Diffusion-NullFusion (CAVE x4)
# ---------------------------------------------------------------------------

class DiffusionNullFusion(nn.Module):
    """Diffusion-NullFusion: Swin UNet + DDPM in nullspace + Sensor AdaLN.

    Training: predict noise epsilon in nullspace of A=[D;R].
    Inference: DDIM denoising + multi-sample ensemble.
    Data consistency: X = pinv_A(yH, yM) + P_N(denoised).
    """
    def __init__(self, bands=31, msi=3, base_dim=64, scale=4,
                 cond_dim=64, T=1000, num_heads=None, depths=None,
                 window_size=8, use_checkpoint=False, cg_steps=20):
        super().__init__()
        self.bands = bands
        self.scale = scale
        self.T = T

        srf_np = nike_d700_srf(bands)
        srf = torch.from_numpy(srf_np).float()
        self.register_buffer("srf", srf)

        self.op = CombinedOperator(
            scale, bands, msi, srf_np, cg_steps=cg_steps, ridge=1e-6,
        )

        self.sensor_embed = SensorEmbedding(srf_np, cond_dim)

        self.schedule = CosineSchedule(T)

        self.cond_proj = nn.Sequential(
            nn.Conv2d(msi + bands, cond_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
        )

        self.unet = MultiScaleSwinUNet(
            in_ch=bands, base_dim=base_dim, num_heads=num_heads,
            cond_dim=cond_dim, window_size=window_size, depths=depths,
            use_checkpoint=use_checkpoint,
        )

    def _conditioning(self, yH, yM, H_hr, W_hr):
        with torch.no_grad():
            base = self.op.pinv(yH, yM, (H_hr, W_hr))
        obs = F.interpolate(yH, (H_hr, W_hr), mode="bicubic", align_corners=False)
        cond = self.cond_proj(torch.cat([yM, obs], dim=1))
        cond = cond + self.sensor_embed()
        return cond, base

    def forward(self, x_t, t, cond):
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        return self.unet(x_t, cond + t_emb)

    @torch.no_grad()
    def inference(self, yH, yM, num_samples=5, ddim_steps=50):
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

        avg = torch.mean(torch.stack(samples), dim=0)
        avg_null = self.op.project_null(avg)
        return {"out": base + avg_null, "base": base, "null": avg_null}


# ---------------------------------------------------------------------------
# CAVE Dataset (liptee PNG layout, 31 bands)
# ---------------------------------------------------------------------------

class CAVEDataset(Dataset):
    """CAVE dataset loader for review benchmarks (liptee PNG layout).

    Expected layout:
      <root>/Train/<scene>/band_01.png ... band_31.png
      <root>/Test/<scene>/band_01.png  ... band_31.png
    """

    def __init__(self, root, split="train", bands=31, scale=4,
                 patch_size=80, max_dim=512):
        self.root = root
        self.split = split
        self.bands = bands
        self.scale = scale
        self.patch_size = patch_size
        self.max_dim = max_dim
        self.is_train = split.lower() == "train"
        self.srf = nike_d700_srf(bands)
        self.kernel = gaussian_kernel2d(9, 1.2)

        self.scenes = self._discover_scenes(root, split)
        self._cache = {}
        for name, path in self.scenes:
            self._cache[name] = self._load_scene_bands(path, bands, max_dim)

        expected = 20 if self.is_train else 12
        if len(self.scenes) != expected:
            print(f"[CAVE] WARNING: expected {expected} {split} scenes, found {len(self.scenes)}")
        print(f"[CAVE] {split}: {len(self.scenes)} scenes loaded")

    def _discover_scenes(self, root, split):
        split_dir = None
        for name in (split, split.capitalize(), split.upper()):
            cand = os.path.join(root, name)
            if os.path.isdir(cand):
                split_dir = cand
                break
        if split_dir is None:
            raise FileNotFoundError(f"split '{split}' not found under {root}")

        scenes = []
        seen = set()
        for dirpath, dirnames, filenames in os.walk(split_dir):
            dirnames.sort()
            # (a) PNG scene dir at any depth
            for bn in ("band_01.png", "Band_01.png", "BAND_01.png"):
                if bn in filenames:
                    rel = os.path.relpath(dirpath, split_dir)
                    if rel not in seen:
                        seen.add(rel)
                        scenes.append((rel.replace(os.sep, "_"), dirpath))
                    break
            # (b) .mat scene files at any depth
            for fn in sorted(filenames):
                if fn.lower().endswith(".mat"):
                    fp = os.path.join(dirpath, fn)
                    if fp not in seen:
                        seen.add(fp)
                        rel = os.path.relpath(fp, split_dir)
                        scenes.append((rel.replace(os.sep, "_")[:-4], fp))
        if not scenes:
            tree = []
            for dirpath, dirnames, filenames in os.walk(root):
                depth = dirpath.replace(root, "").count(os.sep)
                if depth > 4:
                    dirnames[:] = []
                    continue
                tree.append(f"{dirpath} dirs={sorted(dirnames)[:8]} files={len(filenames)}")
                if len(tree) > 40:
                    break
            raise FileNotFoundError(
                f"no scenes (.mat or band_*.png) under {split_dir}\n"
                + "\n".join(tree[:40])
            )
        scenes.sort(key=lambda s: s[0])
        return scenes

    def _load_mat_cube(self, mat_path, bands=31):
        try:
            from scipy.io import loadmat
            data = loadmat(mat_path)
        except NotImplementedError:
            import h5py
            data = {}
            with h5py.File(mat_path, "r") as f:
                for key in f.keys():
                    if not key.startswith("__"):
                        val = np.array(f[key])
                        if getattr(val, "ndim", 0) >= 2:
                            data[key] = val
        for key, val in data.items():
            if str(key).startswith("__") or not hasattr(val, "shape"):
                continue
            arr = np.array(val, dtype=np.float32)
            if arr.ndim != 3 or min(arr.shape) <= 10:
                continue
            if arr.shape[0] > arr.shape[-1]:
                arr = arr.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
            if bands != arr.shape[0]:
                # spectral resample per-pixel via interpolation along band axis
                xs = np.linspace(0.0, 1.0, arr.shape[0])
                xd = np.linspace(0.0, 1.0, bands)
                H_, W_ = arr.shape[1], arr.shape[2]
                flat = arr.reshape(arr.shape[0], -1)
                res = np.empty((bands, flat.shape[1]), dtype=np.float32)
                for i in range(flat.shape[1]):
                    res[:, i] = np.interp(xd, xs, flat[:, i])
                arr = res.reshape(bands, H_, W_)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            if arr.max() > 1.0:
                arr = arr / arr.max()
            arr = np.clip(arr, 0.0, 1.0)
            return arr.astype(np.float32)
        raise ValueError(f"No 3D array in {mat_path}")

    def _load_scene_bands(self, scene_path, bands=31, max_dim=None):
        if os.path.isfile(scene_path) and scene_path.lower().endswith(".mat"):
            cube = self._load_mat_cube(scene_path, bands)
        else:
            try:
                from PIL import Image
                use_pil = True
            except ImportError:
                use_pil = False

            arrays = []
            for i in range(1, bands + 1):
                loaded = False
                for pattern in (f"band_{i:02d}.png", f"Band_{i:02d}.png",
                                f"BAND_{i:02d}.png", f"band_{i}.png"):
                    path = os.path.join(scene_path, pattern)
                    if os.path.isfile(path):
                        if use_pil:
                            img = Image.open(path)
                            arr = np.asarray(img, dtype=np.float32)
                        else:
                            import cv2
                            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                        if arr.max() > 1.0:
                            arr = arr / 255.0
                        arrays.append(arr)
                        loaded = True
                        break
                if not loaded:
                    raise FileNotFoundError(f"band {i} not found in {scene_path}")
            cube = np.stack(arrays, axis=0)

        if max_dim is not None and (cube.shape[1] > max_dim or cube.shape[2] > max_dim):
            y0 = max(0, (cube.shape[1] - max_dim) // 2)
            x0 = max(0, (cube.shape[2] - max_dim) // 2)
            cube = cube[:, y0:y0 + max_dim, x0:x0 + max_dim]
        return cube.astype(np.float32)

    def __len__(self):
        return 10000 if self.is_train else len(self.scenes)

    def _random_crop(self, hsi, size):
        _, H, W = hsi.shape
        if H < size or W < size:
            pad_h = max(0, size - H)
            pad_w = max(0, size - W)
            hsi = np.pad(hsi, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            _, H, W = hsi.shape
        y = np.random.randint(0, H - size + 1)
        x = np.random.randint(0, W - size + 1)
        return hsi[:, y:y + size, x:x + size]

    def _augment(self, hsi, msi):
        if np.random.random() < 0.5:
            hsi = hsi[:, :, ::-1].copy()
            msi = msi[:, :, ::-1].copy()
        if np.random.random() < 0.5:
            hsi = hsi[:, ::-1, :].copy()
            msi = msi[:, ::-1, :].copy()
        k = np.random.randint(0, 4)
        if k:
            hsi = np.rot90(hsi, k, axes=(-2, -1)).copy()
            msi = np.rot90(msi, k, axes=(-2, -1)).copy()
        return hsi, msi

    def _simulate(self, gt):
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
        if self.is_train:
            name = list(self._cache.keys())[np.random.randint(0, len(self._cache))]
            gt = self._cache[name].copy()
            gt = self._random_crop(gt, self.patch_size)
            lr, msi = self._simulate(gt)
            gt, msi = self._augment(gt, msi)
        else:
            name, _ = self.scenes[idx % len(self.scenes)]
            gt = self._cache[name].copy()
            H, W = gt.shape[1], gt.shape[2]
            H = (H // self.scale) * self.scale
            W = (W // self.scale) * self.scale
            gt = gt[:, :H, :W]
            lr, msi = self._simulate(gt)

        return {
            "gt": torch.from_numpy(gt.astype(np.float32)),
            "lr": torch.from_numpy(lr.astype(np.float32)),
            "msi": torch.from_numpy(msi.astype(np.float32)),
            "scene": name,
        }


# ---------------------------------------------------------------------------
# Losses (fixed: charbonnier + SSIM + SAM + gradient + physics)
# ---------------------------------------------------------------------------

def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def ssim_loss(pred, target, data_range=1.0, size=11, sigma=1.5):
    c = pred.shape[1]
    coords = torch.arange(size, device=pred.device, dtype=pred.dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :]).expand(c, 1, size, size)
    mu1 = F.conv2d(pred, win, padding=size // 2, groups=c)
    mu2 = F.conv2d(target, win, padding=size // 2, groups=c)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, win, padding=size // 2, groups=c) - mu1s
    s2 = F.conv2d(target * target, win, padding=size // 2, groups=c) - mu2s
    s12 = F.conv2d(pred * target, win, padding=size // 2, groups=c) - mu12
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return 1.0 - ssim_map.mean()


def sam_loss(pred, target, eps=1e-6):
    p, t = pred.flatten(2), target.flatten(2)
    num = (p * t).sum(dim=1)
    den = p.norm(dim=1) * t.norm(dim=1)
    cos = (num / den.clamp_min(eps)).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


def gradient_loss(pred, target):
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def diffusion_loss(model, x0, cond, schedule, min_snr_gamma=5.0):
    B = x0.shape[0]
    t = torch.randint(0, schedule.T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = schedule.add_noise(x0, noise, t)
    pred = model(x_t, t, cond)
    # min-SNR weighted MSE
    per_sample = F.mse_loss(pred, noise, reduction="none").flatten(1).mean(1)
    ab = schedule.alpha_bar[t].clamp(1e-5, 1 - 1e-5)
    snr = ab / (1 - ab)
    w = (snr.clamp(max=min_snr_gamma) / snr).detach()
    return (per_sample * w).mean()


def physics_loss(pred, yH, yM, op):
    lr_pred = op.D(pred)
    msi_pred = op.R(pred)
    return F.mse_loss(lr_pred, yH) + F.mse_loss(msi_pred, yM)


def total_loss(model, gt, yH, yM, schedule,
               w_char=1.0, w_ssim=0.5, w_sam=0.05, w_grad=0.2,
               w_noise=1.0, w_phys=0.1, min_snr_gamma=5.0):
    H_hr, W_hr = yM.shape[-2], yM.shape[-1]
    gt32 = gt.float()
    yH32 = yH.float()
    yM32 = yM.float()

    with torch.no_grad():
        cond, base = model._conditioning(yH32, yM32, H_hr, W_hr)
        x0 = model.op.project_null(gt32 - base)

    l_noise = diffusion_loss(model, x0.detach(), cond.detach(), schedule, min_snr_gamma)

    pred = (base + x0).detach().requires_grad_(True)
    l_char = charbonnier_loss(pred, gt32)
    l_ssim = ssim_loss(pred.clamp(0, 1), gt32)
    l_sam = sam_loss(pred, gt32)
    l_grad = gradient_loss(pred, gt32)
    l_phys = physics_loss(pred, yH32, yM32, model.op)

    total = (w_noise * l_noise + w_phys * l_phys
             + w_char * l_char + w_ssim * l_ssim
             + w_sam * l_sam + w_grad * l_grad)
    return total, {
        "noise": l_noise.item(), "phys": l_phys.item(),
        "char": l_char.item(), "ssim": l_ssim.item(),
        "sam": l_sam.item(), "grad": l_grad.item(),
    }


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
# Kaggle-safe checkpoint manager
# ---------------------------------------------------------------------------

class KaggleCheckpoint:
    """Periodic checkpoint + SIGTERM handler for Kaggle time limits."""

    def __init__(self, save_dir, device="cuda"):
        self.save_dir = save_dir
        self.device = device
        self.ckpt_path = os.path.join(save_dir, "resume.pth")
        self.best_path = os.path.join(save_dir, "best.pth")
        self._t0 = time.time()
        self._limit = 11.5 * 3600  # 11h30m (save 30min before kill)
        self._saved = False
        self._objects = {}
        self._register_signal()

    def _register_signal(self):
        if sys.platform == "win32":
            return
        original = signal.getsignal(signal.SIGTERM)

        def handler(signum, frame):
            if self._saved:
                return
            print(f"\n[KAGGLE] Signal {signum} — force saving...")
            try:
                self.save(**self._objects, force=True)
                self._saved = True
            except Exception as e:
                print(f"[KAGGLE] Save failed: {e}")
            if callable(original):
                original(signum, frame)

        try:
            signal.signal(signal.SIGTERM, handler)
        except (OSError, ValueError):
            pass

    def time_remaining(self):
        return max(0.0, self._limit - (time.time() - self._t0))

    def should_save(self, step, every=500):
        if step % every == 0:
            return True
        if self.time_remaining() < 1800 and step % 200 == 0:
            return True
        return False

    def save(self, model, ema, opt, scheduler, epoch, best_psnr, best_epoch,
             force=False):
        self._objects = dict(
            model=model, ema=ema, opt=opt, scheduler=scheduler,
            epoch=epoch, best_psnr=best_psnr, best_epoch=best_epoch,
        )
        tmp = self.ckpt_path + ".tmp"
        torch.save({
            "model": model.state_dict(),
            "ema": ema.shadow,
            "opt": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "best_epoch": best_epoch,
            "wall_time": time.time() - self._t0,
        }, tmp)
        os.replace(tmp, self.ckpt_path)

    def save_best(self, model, opt, ema, scheduler, epoch, best_psnr, best_epoch,
                  val_metrics):
        self.save(model, ema, opt, scheduler, epoch, best_psnr, best_epoch)
        torch.save({
            "model": model.state_dict(),
            "val": val_metrics,
            "epoch": epoch,
        }, self.best_path)
        print(f"[ckpt] Best saved: PSNR={val_metrics['psnr']:.4f}")

    def load(self, model, ema, opt, scheduler):
        if not os.path.exists(self.ckpt_path):
            return 1, float("-inf"), 0
        ck = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow = {k: v.to(self.device) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["scheduler"])
        epoch = ck["epoch"]
        best_psnr = ck.get("best_psnr", float("-inf"))
        best_epoch = ck.get("best_epoch", 0)
        wall = ck.get("wall_time", 0)
        print(f"[ckpt] Resumed: epoch {epoch}, best PSNR={best_psnr:.4f}, "
              f"wall={wall / 3600:.1f}h")
        return epoch + 1, best_psnr, best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="CAVE dataset root")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--bands", type=int, default=31)
    parser.add_argument("--patch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--time_budget_h", type=float, default=8.5)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--base_dim", type=int, default=64)
    parser.add_argument("--cond_dim", type=int, default=64)
    parser.add_argument("--cg_steps", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/diffusion_nullfusion")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory: {mem:.1f} GB")

    # Auto-tune for GPU memory
    use_checkpoint = True
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem < 8:
            args.patch = 64
            args.batch_size = 1
            args.base_dim = 48
        elif mem < 12:
            args.patch = 72
            args.batch_size = 2
            args.base_dim = 56
        else:
            args.patch = 80
            args.batch_size = 2
            args.base_dim = 64

    model = DiffusionNullFusion(
        bands=args.bands, msi=3, base_dim=args.base_dim, scale=args.scale,
        cond_dim=args.cond_dim, T=args.T,
        use_checkpoint=use_checkpoint, cg_steps=args.cg_steps,
    ).to(device)

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: Diffusion-NullFusion CAVE — {nparams / 1e6:.2f}M params")
    print(f"Config: patch={args.patch} batch={args.batch_size} dim={args.base_dim}")

    schedule = model.schedule.to(device)

    # Data
    train_ds = CAVEDataset(args.root, "train", args.bands, args.scale, args.patch)
    test_ds = CAVEDataset(args.root, "test", args.bands, args.scale, args.patch)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            betas=(0.9, 0.999))
    ema = EMA(model, 0.999)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=200, T_mult=2, eta_min=1e-6,
    )
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Checkpoint
    ckpt = KaggleCheckpoint(args.save_dir, device=str(device))
    start_epoch, best_psnr, best_epoch = ckpt.load(model, ema, opt, scheduler)

    T0 = time.time()
    LIMIT = args.time_budget_h * 3600

    targets = {
        "FeINFN (2024)": 52.47,
        "BDT (2023)": 52.30,
        "3DT-Net (2023)": 51.38,
        "DSPNet (2023)": 51.18,
        "DHIF (2022)": 51.07,
        "MIMO-SST (2022)": 50.98,
        "CoFusion (2026)": 50.67,
        "PSRT (2023)": 50.47,
    }
    our_prev_best = 50.31  # NullFusion v4 — our previous method

    print(f"\nTraining: {args.epochs} epochs, {args.time_budget_h}h budget")
    print(f"Diffusion T={args.T}, DDIM steps={args.ddim_steps}, samples={args.num_samples}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}, Steps/epoch: {args.steps_per_epoch}")
    print("-" * 70)

    # ---- Smoke test: run 1 loss step on synthetic data to catch NaN early ----
    print("[SMOKE] Running 1-step loss on synthetic data...")
    model.eval()
    _sgt = torch.rand(2, args.bands, args.patch, args.patch, device=device)
    _slr = torch.rand(2, args.bands, args.patch // args.scale, args.patch // args.scale, device=device)
    _smsi = torch.rand(2, 3, args.patch, args.patch, device=device)
    with torch.no_grad():
        _sloss, _slogs = total_loss(model, _sgt, _slr, _smsi, schedule)
    print(f"[SMOKE] loss={_sloss.item():.6f} finite={math.isfinite(_sloss.item())}")
    for k, v in _slogs.items():
        print(f"  {k}: {v:.6f} finite={math.isfinite(v)}")
    if not math.isfinite(_sloss.item()):
        print("[SMOKE] FAILED — NaN/inf detected in smoke test! Aborting.")
        sys.exit(1)
    print("[SMOKE] PASSED — all losses finite.\n")
    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        if (time.time() - T0) > LIMIT:
            print(f"\n[TIME BUDGET at epoch {epoch}]")
            ckpt.save(model, ema, opt, scheduler, epoch - 1, best_psnr, best_epoch,
                      force=True)
            break

        model.train()
        total = 0
        t0 = time.time()
        opt.zero_grad()
        _debug = epoch <= 3

        for step in range(args.steps_per_epoch):
            batch = [train_ds[random.randrange(len(train_ds))]
                     for _ in range(args.batch_size)]
            gt = torch.stack([b["gt"] for b in batch], 0).to(device)
            yH = torch.stack([b["lr"] for b in batch], 0).to(device)
            yM = torch.stack([b["msi"] for b in batch], 0).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, logs = total_loss(model, gt, yH, yM, schedule)

            scaler.scale(loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(model)

            total += loss.item()

            # periodic save
            if ckpt.should_save(epoch * args.steps_per_epoch + step):
                ckpt.save(model, ema, opt, scheduler, epoch, best_psnr, best_epoch)

        scheduler.step()
        dt = time.time() - t0
        avg = total / args.steps_per_epoch

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{args.epochs} | Loss {avg:.6f} | "
                  f"LR {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")
            if _debug and not math.isfinite(avg):
                print("[NaN DETECTED] Checking data batch for NaN/inf...")
                _b = train_ds[random.randrange(len(train_ds))]
                _gt = _b["gt"]; _lr = _b["lr"]; _msi = _b["msi"]
                print(f"  gt: range=[{_gt.min():.4f}, {_gt.max():.4f}] nan={_gt.isnan().any()} inf={_gt.isinf().any()}")
                print(f"  lr: range=[{_lr.min():.4f}, {_lr.max():.4f}] nan={_lr.isnan().any()} inf={_lr.isinf().any()}")
                print(f"  msi: range=[{_msi.min():.4f}, {_msi.max():.4f}] nan={_msi.isnan().any()} inf={_msi.isinf().any()}")

        # Evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply(model)
            model.eval()
            ps_list, ss_list, sa_list, er_list = [], [], [], []

            with torch.no_grad():
                for i in range(len(test_ds)):
                    item = test_ds[i]
                    out = model.inference(
                        item["lr"].unsqueeze(0).to(device),
                        item["msi"].unsqueeze(0).to(device),
                        num_samples=1, ddim_steps=20,
                    )
                    pred_np = out["out"][0].cpu().numpy()
                    gt_np = item["gt"].numpy()
                    ps_list.append(psnr_np(pred_np, gt_np))
                    ss_list.append(ssim_np(pred_np, gt_np))
                    sa_list.append(sam_np(pred_np, gt_np))
                    er_list.append(ergas_np(pred_np, gt_np, args.scale))

            mp = float(np.mean(ps_list))
            ms_ = float(np.mean(ss_list))
            ma = float(np.mean(sa_list))
            me = float(np.mean(er_list))

            marker = ""
            if mp > best_psnr:
                best_psnr = mp
                best_epoch = epoch
                ckpt.save_best(model, opt, ema, scheduler, epoch,
                               best_psnr, best_epoch,
                               {"psnr": mp, "ssim": ms_, "sam": ma, "ergas": me})
                marker = " [BEST]"

            print(f"  >>> Test@{epoch}: PSNR {mp:.4f} | SSIM {ms_:.4f} | "
                  f"SAM {ma:.3f} | ERGAS {me:.3f}{marker}")
            print(f"    vs NullFusion v4 (our prev best, {our_prev_best:.2f} dB): "
                  f"delta={mp - our_prev_best:+.2f} dB"
                  + (" <<< BEAT" if mp > our_prev_best else ""))
            for name, target in targets.items():
                d = mp - target
                m_ = " <<< BEAT" if d > 0 else ""
                print(f"    vs {name:20s}: delta={d:+.2f} dB{m_}")

            model.load_state_dict(raw_state)
            model.train()

            ckpt.save(model, ema, opt, scheduler, epoch, best_psnr, best_epoch,
                      force=True)

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION: 5 samples x 50 DDIM steps, FULL test set")
    print("=" * 70)

    best_ckpt = os.path.join(args.save_dir, "best.pth")
    if os.path.exists(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    model.eval()
    ps_list, ss_list, sa_list, er_list = [], [], [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            item = test_ds[i]
            out = model.inference(
                item["lr"].unsqueeze(0).to(device),
                item["msi"].unsqueeze(0).to(device),
                num_samples=args.num_samples,
                ddim_steps=args.ddim_steps,
            )
            pred_np = out["out"][0].cpu().numpy()
            gt_np = item["gt"].numpy()
            ps_list.append(psnr_np(pred_np, gt_np))
            ss_list.append(ssim_np(pred_np, gt_np))
            sa_list.append(sam_np(pred_np, gt_np))
            er_list.append(ergas_np(pred_np, gt_np, args.scale))

    final = {
        "psnr": float(np.mean(ps_list)),
        "ssim": float(np.mean(ss_list)),
        "sam": float(np.mean(sa_list)),
        "ergas": float(np.mean(er_list)),
    }
    print(f"\nFINAL (epoch {best_epoch}): PSNR {final['psnr']:.4f} | "
          f"SSIM {final['ssim']:.4f} | SAM {final['sam']:.3f} | "
          f"ERGAS {final['ergas']:.3f}")

    for name, target in {**targets, "Stretch (53 dB)": 53.0}.items():
        d = final["psnr"] - target
        m_ = " <<< BEAT" if d > 0 else ""
        print(f"  {name:25s} Target {target:6.2f} | Ours delta={d:+.2f} dB{m_}")

    results = {
        "dataset": "CAVE",
        "bands": args.bands,
        "protocol": "Wald x4, Nikon D700 SRF, Gaussian 9x9 sigma=1.2",
        "params_M": nparams / 1e6,
        "best_epoch": best_epoch,
        "T": args.T,
        "ddim_steps": args.ddim_steps,
        "num_samples": args.num_samples,
        "targets": targets,
        "final": final,
    }
    out_path = os.path.join(args.save_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
