#!/usr/bin/env python3
"""Self-contained DSPNet training script for CAVE x4 fusion task.

Trains DSPNet (Dual Spatial-spectral Pyramid Network) from scratch on the
CAVE dataset using Wald's protocol simulation. Designed for Kaggle P100 16GB.

Usage:
    python train_dspnet.py
    python train_dspnet.py --root /kaggle/input/datasets/liptee/...
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from einops import rearrange
except ImportError:
    def rearrange(tensor, pattern, **kwargs):
        """Minimal einops rearrange for the patterns used in DSPNet."""
        if "b n (h d) -> b h n d" in pattern:
            h = kwargs["h"]
            b, n, hd = tensor.shape
            d = hd // h
            return tensor.view(b, n, h, d).permute(0, 2, 1, 3)
        raise NotImplementedError(f"rearrange pattern not supported: {pattern}")

# ---------------------------------------------------------------------------
# Nikon D700 Spectral Response Function (31 bands, 400-700 nm at 10 nm)
# ---------------------------------------------------------------------------
NIKON_D700_SRF = np.array([
    [0.0022, 0.0044, 0.0065], [0.0056, 0.0097, 0.0115],
    [0.0120, 0.0180, 0.0190], [0.0230, 0.0310, 0.0300],
    [0.0420, 0.0500, 0.0430], [0.0680, 0.0740, 0.0570],
    [0.1040, 0.1020, 0.0730], [0.1510, 0.1400, 0.0870],
    [0.2020, 0.1810, 0.1010], [0.2520, 0.2190, 0.1110],
    [0.2940, 0.2520, 0.1180], [0.3250, 0.2800, 0.1220],
    [0.3480, 0.3050, 0.1250], [0.3600, 0.3270, 0.1270],
    [0.3660, 0.3450, 0.1280], [0.3650, 0.3590, 0.1280],
    [0.3570, 0.3700, 0.1280], [0.3420, 0.3760, 0.1280],
    [0.3210, 0.3780, 0.1290], [0.2950, 0.3750, 0.1310],
    [0.2640, 0.3680, 0.1360], [0.2300, 0.3560, 0.1450],
    [0.1940, 0.3390, 0.1610], [0.1590, 0.3160, 0.1860],
    [0.1270, 0.2880, 0.2220], [0.0990, 0.2560, 0.2690],
    [0.0760, 0.2210, 0.3260], [0.0570, 0.1840, 0.3820],
    [0.0420, 0.1480, 0.4320], [0.0300, 0.1150, 0.4660],
    [0.0210, 0.0870, 0.4860],
], dtype=np.float32)
NIKON_D700_SRF /= NIKON_D700_SRF.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Gaussian blur kernel
# ---------------------------------------------------------------------------
def make_gaussian_kernel(size=9, sigma=2.0):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return (k / k.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Wald's protocol simulation
# ---------------------------------------------------------------------------
def simulate_wald(hsi, kernel, scale, srf):
    """Blur+decimate HSI -> LR-HSI, HSI x SRF -> HR-MSI."""
    from scipy.ndimage import convolve as scipy_convolve

    C, H, W = hsi.shape
    blurred = np.empty_like(hsi)
    for c in range(C):
        blurred[c] = scipy_convolve(hsi[c], kernel, mode="wrap")
    hr, wr = H // scale, W // scale
    y0 = (H - hr * scale) // 2
    x0 = (W - wr * scale) // 2
    lr_hsi = blurred[:, y0::scale, x0::scale].astype(np.float32)
    hr_msi = np.einsum("chw,cm->mhw", hsi, srf).astype(np.float32)
    hr_msi = np.clip(hr_msi, 0.0, 1.0)
    return lr_hsi, hr_msi


# ---------------------------------------------------------------------------
# CAVE Dataset (liptee PNG layout)
# ---------------------------------------------------------------------------
def _load_mat(mat_path):
    """Load .mat file, return first real array as float32."""
    from scipy.io import loadmat
    data = loadmat(mat_path)
    for key, val in data.items():
        if not key.startswith('_') and hasattr(val, 'shape'):
            arr = np.array(val, dtype=np.float32)
            if arr.ndim >= 2 and min(arr.shape) > 1:
                if arr.ndim == 3 and arr.shape[2] > arr.shape[0]:
                    arr = arr.transpose(2, 0, 1)
                return arr
    raise ValueError(f"No valid array in {mat_path}")


def _discover_scenes(root, split):
    split_dir = None
    for name in (split, split.capitalize(), split.upper()):
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            split_dir = candidate
            break
    if split_dir is None:
        raise FileNotFoundError(f"split '{split}' not found under {root}")
    scenes = []
    for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        mat_files = [f for f in os.listdir(entry.path) if f.endswith('.mat')]
        if mat_files:
            scenes.append((entry.name, entry.path))
            continue
        for band_name in ("band_01.png", "Band_01.png", "BAND_01.png"):
            if os.path.isfile(os.path.join(entry.path, band_name)):
                scenes.append((entry.name, entry.path))
                break
    if not scenes:
        raise FileNotFoundError(f"no scenes found under {split_dir}")
    return scenes


def _load_scene_bands(scene_dir, bands=31, max_dim=None):
    mat_files = [f for f in os.listdir(scene_dir) if f.endswith('.mat')]
    if mat_files:
        cube = _load_mat(os.path.join(scene_dir, mat_files[0]))
        if max_dim is not None and (cube.shape[1] > max_dim or cube.shape[2] > max_dim):
            y0 = max(0, (cube.shape[1] - max_dim) // 2)
            x0 = max(0, (cube.shape[2] - max_dim) // 2)
            cube = cube[:, y0:y0 + max_dim, x0:x0 + max_dim]
        return cube

    try:
        from PIL import Image
        _use_pil = True
    except ImportError:
        import cv2 as _cv2
        _use_pil = False
    arrays = []
    for i in range(1, bands + 1):
        loaded = False
        for pattern in (f"band_{i:02d}.png", f"Band_{i:02d}.png",
                        f"BAND_{i:02d}.png", f"band_{i}.png"):
            path = os.path.join(scene_dir, pattern)
            if os.path.isfile(path):
                if _use_pil:
                    img = Image.open(path)
                    arr = np.asarray(img, dtype=np.float32)
                else:
                    arr = _cv2.imread(path, _cv2.IMREAD_GRAYSCALE).astype(np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
                arrays.append(arr)
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(f"band {i} not found in {scene_dir}")
    cube = np.stack(arrays, axis=0)
    if max_dim is not None and (cube.shape[1] > max_dim or cube.shape[2] > max_dim):
        y0 = max(0, (cube.shape[1] - max_dim) // 2)
        x0 = max(0, (cube.shape[2] - max_dim) // 2)
        cube = cube[:, y0:y0 + max_dim, x0:x0 + max_dim]
    return cube


class CAVEDataset(Dataset):
    """CAVE dataset for DSPNet training/evaluation."""

    def __init__(self, root, split="train", bands=31, scale=4,
                 patch_size=96, max_dim=512):
        self.split = split
        self.bands = bands
        self.scale = scale
        self.patch_size = patch_size
        self.max_dim = max_dim
        self.is_train = split.lower() == "train"
        self.srf = NIKON_D700_SRF
        self.kernel = make_gaussian_kernel(size=9, sigma=2.0)
        self.scenes = _discover_scenes(root, split)
        self._cache = {}
        for name, path in self.scenes:
            self._cache[name] = _load_scene_bands(path, bands, max_dim)

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

    @staticmethod
    def _augment(hsi, msi):
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

    def __getitem__(self, idx):
        if self.is_train:
            name = list(self._cache.keys())[
                np.random.randint(0, len(self._cache))]
            gt = self._cache[name].copy()
            gt = self._random_crop(gt, self.patch_size)
            hr_msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
            hr_msi = np.clip(hr_msi, 0.0, 1.0)
            gt, hr_msi = self._augment(gt, hr_msi)
            lr_hsi, _ = simulate_wald(gt, self.kernel, self.scale, self.srf)
        else:
            name, _ = self.scenes[idx % len(self.scenes)]
            gt = self._cache[name].copy()
            H, W = gt.shape[1], gt.shape[2]
            H = (H // self.scale) * self.scale
            W = (W // self.scale) * self.scale
            gt = gt[:, :H, :W]
            hr_msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
            hr_msi = np.clip(hr_msi, 0.0, 1.0)
            lr_hsi, _ = simulate_wald(gt, self.kernel, self.scale, self.srf)

        return {
            "gt": torch.from_numpy(gt.astype(np.float32)),
            "lr_hsi": torch.from_numpy(lr_hsi.astype(np.float32)),
            "hr_msi": torch.from_numpy(hr_msi.astype(np.float32)),
            "scene_name": name,
        }


# =========================================================================
# DSPNet Model Architecture (inlined from DSPNet.py)
# =========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class MSA(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_qm = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_km = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_vm = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k2 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v2 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k4 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v4 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k8 = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v8 = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescalem = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale2 = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale4 = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescale8 = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_2, x_4, x_8, y):
        b, h, w, c = y.shape
        x2 = x_2.reshape(b, h * w, c)
        x4 = x_4.reshape(b, h * w, c)
        x8 = x_8.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        q_inpm = self.to_qm(y)
        k_inpm = self.to_km(y)
        v_inpm = self.to_vm(y)
        qm, km, vm = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inpm, k_inpm, v_inpm),
        )
        qm = qm.transpose(-2, -1)
        km = km.transpose(-2, -1)
        vm = vm.transpose(-2, -1)
        qm = F.normalize(qm, dim=-1, p=2)
        km = F.normalize(km, dim=-1, p=2)
        attnm = (km @ qm.transpose(-2, -1)) * self.rescalem
        attnm = attnm.softmax(dim=-1)

        k_inp2 = self.to_k2(x2)
        v_inp2 = self.to_v2(x2)
        k2, v2 = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (k_inp2, v_inp2),
        )
        k2 = k2.transpose(-2, -1)
        v2 = v2.transpose(-2, -1)
        k2 = F.normalize(k2, dim=-1, p=2)
        attn2 = (k2 @ qm.transpose(-2, -1)) * self.rescale2
        attn2 = attn2.softmax(dim=-1)

        k_inp4 = self.to_k4(x4)
        v_inp4 = self.to_v4(x4)
        k4, v4 = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (k_inp4, v_inp4),
        )
        k4 = k4.transpose(-2, -1)
        v4 = v4.transpose(-2, -1)
        k4 = F.normalize(k4, dim=-1, p=2)
        attn4 = (k4 @ qm.transpose(-2, -1)) * self.rescale4
        attn4 = attn4.softmax(dim=-1)

        k_inp8 = self.to_k8(x8)
        v_inp8 = self.to_v8(x8)
        k8, v8 = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (k_inp8, v_inp8),
        )
        k8 = k8.transpose(-2, -1)
        v8 = v8.transpose(-2, -1)
        k8 = F.normalize(k8, dim=-1, p=2)
        attn8 = (k8 @ qm.transpose(-2, -1)) * self.rescale8
        attn8 = attn8.softmax(dim=-1)

        x = attnm @ vm + attn2 @ v2 + attn4 @ v4 + attn8 @ v8
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inpm.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        return out_c + out_p


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False,
                      groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MLSIF(nn.Module):
    def __init__(self, dim, dim_head, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ])
            for _ in range(num_blocks)
        ])

    def forward(self, x_2, x_4, x_8, y):
        x_2 = x_2.permute(0, 2, 3, 1)
        x_4 = x_4.permute(0, 2, 3, 1)
        x_8 = x_8.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x_2, x_4, x_8, y) + y
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class SpePyBlock(nn.Module):
    def __init__(self, inchannels, bias=True):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels * 2, 3, 1, 1, groups=2, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannels * 2, inchannels * 2, 3, 1, 1, groups=2,
                      bias=bias),
            nn.LeakyReLU(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels * 2, 3, 1, 1, groups=4, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannels * 2, inchannels * 2, 3, 1, 1, groups=4,
                      bias=bias),
            nn.LeakyReLU(0.2),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels * 2, 3, 1, 1, groups=8, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inchannels * 2, inchannels * 2, 3, 1, 1, groups=8,
                      bias=bias),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x2, x4, x8):
        _, c, _, _ = x2.shape
        if c % 8 != 0:
            x2 = torch.cat(
                (x2, x2[:, c - (8 - c % 8) - 1:c - 1, :, :]), dim=1)
            x4 = torch.cat(
                (x4, x4[:, c - (8 - c % 8) - 1:c - 1, :, :]), dim=1)
            x8 = torch.cat(
                (x8, x8[:, c - (8 - c % 8) - 1:c - 1, :, :]), dim=1)
        return self.conv2(x2), self.conv4(x4), self.conv8(x8)


class SpaPyBlock(nn.Module):
    def __init__(self, inchannels, outchannels, bias=True):
        super().__init__()
        self.scale1 = nn.Conv2d(inchannels, inchannels, 3, 1, 1, bias=bias)
        self.scale1_2 = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=1 / 2),
            nn.Conv2d(inchannels, inchannels, 3, 1, 1, bias=bias),
        )
        self.scale1_4 = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=1 / 4),
            nn.Conv2d(inchannels, inchannels, 3, 1, 1, bias=bias),
        )
        self.channel = ChannelAttention(inchannels * 3)
        self.out = nn.Sequential(
            nn.Conv2d(inchannels * 3, outchannels, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear",
                               align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear",
                               align_corners=True)

    def forward(self, x):
        y1 = self.scale1(x)
        y2 = self.up2(self.scale1_2(x))
        y3 = self.up4(self.scale1_4(x))
        y = torch.cat((y1, y2, y3), dim=1)
        y = self.channel(y) * y
        return self.out(y)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=1 / 2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Downm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=1 / 2),
            SpaPyBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Upm(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.conv(x)


class DSPNet(nn.Module):
    """Dual Spatial-spectral Pyramid Network for HSI fusion."""

    def __init__(self, hschannels, mschannels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        self.spe1 = SpePyBlock(32)
        self.spe2 = SpePyBlock(64)
        self.spe3 = SpePyBlock(128)
        self.ds = nn.Upsample(mode="bilinear", scale_factor=1 / 2)
        self.mls1 = MLSIF(dim=64, num_blocks=1, dim_head=64, heads=64 // 64)
        self.mls2 = MLSIF(dim=128, num_blocks=1, dim_head=64, heads=128 // 64)
        self.mls3 = MLSIF(dim=256, num_blocks=1, dim_head=64, heads=256 // 64)
        self.inc = DoubleConv(hschannels + mschannels, 64)
        self.down1 = Down(64 + 32, 128)
        self.down2 = Down(128 + 64, 256)

        self.up1 = Upm(128 + 256, 128, bilinear)
        self.up2 = Upm(128 + 128 + 64, 64, bilinear)
        self.outc = OutConv(64 + 64 + 32, hschannels)

        self.spa1 = SpaPyBlock(mschannels, 32)
        self.spa2 = Downm(32, 64)
        self.spa3 = Downm(64, 128)

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=4, mode="bicubic",
                          align_corners=False)
        x0 = x
        y1 = self.spa1(y)
        y2 = self.spa2(y1)
        y3 = self.spa3(y2)

        z1_2, z1_4, z1_8 = self.spe1(x, x, x)
        z2_2, z2_4, z2_8 = self.spe2(
            self.ds(z1_2), self.ds(z1_4), self.ds(z1_8))
        z3_2, z3_4, z3_8 = self.spe3(
            self.ds(z2_2), self.ds(z2_4), self.ds(z2_8))

        x1 = self.inc(torch.cat((x, y), dim=1))
        x1 = self.mls1(z1_2, z1_4, z1_8, x1)
        x2 = self.down1(torch.cat((x1, y1), dim=1))
        x2 = self.mls2(z2_2, z2_4, z2_8, x2)
        x3 = self.down2(torch.cat((x2, y2), dim=1))
        x3 = self.mls3(z3_2, z3_4, z3_8, x3)
        x = self.up1(torch.cat((x3, y3), dim=1))
        x = self.up2(torch.cat((x, x2, y2), dim=1))
        logits = self.outc(torch.cat((x, x1, y1), dim=1))

        return logits + x0


# =========================================================================
# Evaluation Metrics
# =========================================================================

def calc_psnr(pred, gt):
    return -10.0 * torch.log10(
        torch.mean((pred - gt) ** 2) + 1e-8).item()


def calc_ssim(pred, gt, channels=31, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ax = torch.arange(window_size, device=pred.device,
                      dtype=pred.dtype) - window_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    _2d = torch.exp(
        -(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2)
    ).unsqueeze(0).unsqueeze(0)
    _2d = _2d / _2d.sum()
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(gt, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(pred * pred, window, padding=pad,
                  groups=channels) - mu1_sq
    s2 = F.conv2d(gt * gt, window, padding=pad, groups=channels) - mu2_sq
    s12 = F.conv2d(pred * gt, window, padding=pad,
                   groups=channels) - mu1_mu2
    cs = (2 * s12 + C2) / (s1 + s2 + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs
    return ssim_map.mean().item()


def calc_sam(pred, gt):
    B, C, H, W = pred.shape
    p = pred.view(B, C, -1)
    g = gt.view(B, C, -1)
    p = F.normalize(p, dim=1)
    g = F.normalize(g, dim=1)
    cos_sim = torch.clamp((p * g).sum(dim=1), -1.0, 1.0)
    sam = torch.acos(cos_sim)
    return sam.mean().item() * (180.0 / math.pi)


def calc_ergas(pred, gt, scale=4):
    B, C, H, W = pred.shape
    err = (pred - gt) ** 2
    ergas = 0.0
    for c in range(C):
        mg = gt[:, c].mean()
        if mg > 1e-8:
            ergas += err[:, c].mean() / (mg ** 2)
    return math.sqrt(ergas / C) * 100.0 * scale


# =========================================================================
# Training
# =========================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"SM: {torch.cuda.get_device_capability(0)}")

    print("\nLoading datasets...")
    train_ds = CAVEDataset(
        args.root, split="train", bands=args.bands, scale=args.scale,
        patch_size=args.patch_size, max_dim=args.max_dim)
    test_ds = CAVEDataset(
        args.root, split="test", bands=args.bands, scale=args.scale,
        patch_size=0, max_dim=args.max_dim)
    print(f"Train scenes: {len(train_ds.scenes)}, "
          f"Test scenes: {len(test_ds.scenes)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    iter_loader = iter(train_loader)

    print("\nBuilding DSPNet...")
    model = DSPNet(args.bands, 3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(layer.weight)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(1, args.epochs, 5)),
        gamma=0.95,
    )

    best_psnr = 0.0
    best_epoch = 0
    save_dir = os.path.join(args.output_dir, "dspnet_checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch: {args.batch_size}, Patch: {args.patch_size}, "
          f"LR: {args.lr}")
    print(f"Eval every {args.eval_every} epochs on "
          f"{len(test_ds.scenes)} test scenes")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            gt = batch["gt"].to(device)
            lr_hsi = batch["lr_hsi"].to(device)
            hr_msi = batch["hr_msi"].to(device)

            output = model(lr_hsi, hr_msi).clip(0, 1)
            loss = criterion(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / args.steps_per_epoch
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} | LR: {lr_now:.2e} | "
                  f"Time: {elapsed:.1f}s")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            psnr_l, ssim_l, sam_l, ergas_l = [], [], [], []

            with torch.no_grad():
                for i in range(len(test_ds.scenes)):
                    item = test_ds[i]
                    gt_e = item["gt"].unsqueeze(0).to(device)
                    lr_e = item["lr_hsi"].unsqueeze(0).to(device)
                    hr_e = item["hr_msi"].unsqueeze(0).to(device)
                    pred = model(lr_e, hr_e).clip(0, 1)
                    psnr_l.append(calc_psnr(pred, gt_e))
                    ssim_l.append(calc_ssim(pred, gt_e))
                    sam_l.append(calc_sam(pred, gt_e))
                    ergas_l.append(calc_ergas(pred, gt_e, args.scale))

            avg_psnr = np.mean(psnr_l)
            avg_ssim = np.mean(ssim_l)
            avg_sam = np.mean(sam_l)
            avg_ergas = np.mean(ergas_l)
            print(f"  [Eval] PSNR: {avg_psnr:.4f} dB | "
                  f"SSIM: {avg_ssim:.4f} | "
                  f"SAM: {avg_sam:.4f} deg | "
                  f"ERGAS: {avg_ergas:.4f}")

            for i, name in enumerate(test_ds.scenes):
                print(f"    {name[0]:30s}  PSNR={psnr_l[i]:.2f}  "
                      f"SSIM={ssim_l[i]:.4f}  SAM={sam_l[i]:.4f}  "
                      f"ERGAS={ergas_l[i]:.4f}")

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                ckpt = os.path.join(save_dir, "dspnet_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "psnr": avg_psnr, "ssim": avg_ssim,
                    "sam": avg_sam, "ergas": avg_ergas,
                }, ckpt)
                print(f"  >> New best! Saved to {ckpt}")

    print("\n" + "=" * 70)
    print(f"Training complete! Best PSNR: {best_psnr:.4f} dB "
          f"at epoch {best_epoch}")
    print("=" * 70)


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DSPNet Training for CAVE x4")
    parser.add_argument(
        "--root", type=str,
        default="/kaggle/input/datasets/liptee/"
                "hyperspectral-image-restoration-based-on-cave",
        help="Dataset root directory")
    parser.add_argument("--bands", type=int, default=31)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--output_dir", type=str,
        default="/kaggle/working",
        help="Directory to save checkpoints")
    args = parser.parse_args()
    train(args)
