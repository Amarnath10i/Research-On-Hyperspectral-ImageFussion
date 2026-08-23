#!/usr/bin/env python3
"""Self-contained FeINFN training script for CAVE x4 fusion task.

Trains FeINFN (NeurIPS 2024) from scratch on the CAVE dataset using
Wald's protocol simulation. Designed for Kaggle P100 16GB GPU.

Usage:
    python train_feinfn.py
    python train_feinfn.py --root /kaggle/input/datasets/liptee/hyperspectral-image-restoration-based-on-cave
"""
import argparse
import copy
import math
import os
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

###############################################################################
# Data Loading (from cave_common.py)
###############################################################################
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


def nike_d700_srf(bands=31, normalise=True):
    src = _NIKON_D700_31
    if bands != src.shape[0]:
        xs = np.linspace(0.0, 1.0, src.shape[0])
        xd = np.linspace(0.0, 1.0, bands)
        src = np.stack([np.interp(xd, xs, src[:, i]) for i in range(3)], axis=1)
    srf = src.astype(np.float32)
    if normalise:
        srf = srf / np.maximum(srf.sum(axis=0, keepdims=True), 1e-8)
    return srf


def make_gaussian_kernel(size=9, sigma=1.2):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def simulate_wald(hsi, kernel, scale, srf):
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


def _load_mat(path):
    """Load a .mat file and return the first real array as float32."""
    import scipy.io as sio
    mat = sio.loadmat(path)
    for k, v in mat.items():
        if not k.startswith("__"):
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim >= 2:
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                elif arr.ndim == 3 and arr.shape[2] in (3, 31):
                    arr = arr.transpose(2, 0, 1)
                return arr
    raise ValueError(f"No valid array in {path}")


def _discover_scenes(root, split):
    """Discover scenes from liptee CAVE dataset structure.
    
    Expects: root/Train/HSI/*.mat and root/Test/HSI/*.mat
    Falls back to root/Train/MONO/*.mat for scene names.
    """
    split_dir = None
    for name in (split, split.capitalize(), split.upper()):
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            split_dir = candidate
            break
    if split_dir is None:
        raise FileNotFoundError(f"split '{split}' not found under {root}")
    
    # Try HSI directory first (standard liptee structure)
    hsi_dir = None
    for name in ("HSI", "hsi", "Hsi"):
        candidate = os.path.join(split_dir, name)
        if os.path.isdir(candidate):
            hsi_dir = candidate
            break
    
    if hsi_dir:
        # liptee structure: Train/HSI/*.mat
        scenes = []
        for f in sorted(os.listdir(hsi_dir)):
            if f.endswith(".mat"):
                stem = os.path.splitext(f)[0]
                mat_path = os.path.join(hsi_dir, f)
                scenes.append((stem, mat_path))
        if scenes:
            return scenes
    
    # Fallback: look for band_*.png (old structure)
    scenes = []
    for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        for band_name in ("band_01.png", "Band_01.png", "BAND_01.png"):
            if os.path.isfile(os.path.join(entry.path, band_name)):
                scenes.append((entry.name, entry.path))
                break
    if not scenes:
        raise FileNotFoundError(
            f"No scenes found under {split_dir}. "
            f"Expected Train/HSI/*.mat or scene dirs with band_*.png"
        )
    return scenes


def _load_scene_bands(scene_path_or_dir, bands=31, max_dim=None):
    """Load HSI cube from .mat file or band_*.png directory."""
    if os.path.isfile(scene_path_or_dir) and scene_path_or_dir.endswith(".mat"):
        cube = _load_mat(scene_path_or_dir)
    else:
        # Fallback: load band_*.png files from directory
        try:
            from PIL import Image
            _use_pil = True
        except ImportError:
            import cv2
            _use_pil = False
        arrays = []
        for i in range(1, bands + 1):
            loaded = False
            for pattern in (f"band_{i:02d}.png", f"Band_{i:02d}.png",
                            f"BAND_{i:02d}.png", f"band_{i}.png"):
                path = os.path.join(scene_path_or_dir, pattern)
                if os.path.isfile(path):
                    if _use_pil:
                        img = Image.open(path)
                        arr = np.asarray(img, dtype=np.float32)
                    else:
                        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    if arr.max() > 1.0:
                        arr = arr / 255.0
                    arrays.append(arr)
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError(f"band {i} not found in {scene_path_or_dir}")
        cube = np.stack(arrays, axis=0)
    
    if max_dim is not None and (cube.shape[1] > max_dim or cube.shape[2] > max_dim):
        y0 = max(0, (cube.shape[1] - max_dim) // 2)
        x0 = max(0, (cube.shape[2] - max_dim) // 2)
        cube = cube[:, y0:y0 + max_dim, x0:x0 + max_dim]
    return cube


class CAVEDataset:
    def __init__(self, root, split="train", bands=31, scale=4,
                 patch_size=80, max_dim=512, srf=None):
        self.root = root
        self.split = split
        self.bands = bands
        self.scale = scale
        self.patch_size = patch_size
        self.max_dim = max_dim
        self.is_train = split.lower() == "train"
        self.srf = srf if srf is not None else nike_d700_srf(bands)
        self.kernel = make_gaussian_kernel(size=9, sigma=1.2)
        self.scenes = _discover_scenes(root, split)
        print(f"[CAVEDataset] Found {len(self.scenes)} {split} scenes: "
              f"{[s[0] for s in self.scenes[:5]]}...")
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
        return hsi[:, y:y + size, x:x + size], y, x

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
            gt, _, _ = self._random_crop(gt, self.patch_size)
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


###############################################################################
# Model Code (from FeINFN.py + fe_block.py)
###############################################################################

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super().__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super().__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act,
                     res_scale=args.res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            m_tail = [
                Upsampler(conv, args.scale[0], n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        return x


def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                       n_colors=1, scale=2, no_upsampling=True, rgb_range=1):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range
    args.n_colors = n_colors
    return EDSR(args)


class hightfre(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, groups=1):
        super().__init__()
        self.groups = groups
        self.inch = in_channels
        self.outch = out_channels
        kernel = torch.tensor([[0, -1, 0],
                               [-1, 1, -1],
                               [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        return F.conv2d(
            x,
            self.kernel[None, None].repeat_interleave(self.inch, dim=0),
            groups=self.inch, padding=1)


class ComplexGaborLayer(nn.Module):
    def __init__(self, omega0=30.0, sigma0=10.0, trainable=True):
        super().__init__()
        self.omega_0 = nn.Parameter(omega0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(sigma0 * torch.ones(1), trainable)

    def forward(self, input):
        input = input.permute(0, -2, -1, 1)
        omega = self.omega_0 * input
        scale = self.scale_0 * input
        return torch.exp(
            1j * omega - scale.abs().square()
        ).permute(0, -1, 1, 2)


class MLP_P(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Sequential(
                nn.Conv2d(lastv, hidden, kernel_size=1, bias=False),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1,
                          bias=False, groups=hidden)))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Sequential(
            nn.Conv2d(lastv, out_dim, kernel_size=1, bias=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1,
                      bias=False, groups=out_dim)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[None]
        if x.size(1) > self.pe.size(1):
            pe = pe.transpose(1, 2)
            pe = F.interpolate(pe, size=(x.size(1)), mode='linear')
            pe = pe.transpose(1, 2)
        pe_x = pe[:, :x.size(1)]
        x = x + pe_x
        return x


class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels, freq_dim=31, hidden_dims=None,
                 omega=30, scale=10.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]
        last_dim_K = in_channels
        last_dim_Q = freq_dim
        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(
                nn.Conv2d(last_dim_K, hidden_dim, 1), nn.ReLU()))
            self.Q.append(nn.Sequential(
                nn.Conv2d(last_dim_Q, hidden_dim, 1),
                ComplexGaborLayer(omega0=omega, sigma0=scale,
                                  trainable=True)))
            last_dim_K = hidden_dim + in_channels
            last_dim_Q = hidden_dim
        self.last_layer = nn.Conv2d(hidden_dims[-1], in_channels - 1, 1)

    def step(self, x, y):
        k = self.K[0](x).real
        q = k * self.Q[0](y)
        q = q.real
        for i in range(1, len(self.K)):
            k = self.K[i](torch.cat([q, x], dim=1)).real
            q = k * self.Q[i](q)
            q = q.real
        q = self.last_layer(q)
        return q

    def forward(self, INR_feat, freq_feat):
        return self.step(INR_feat, freq_feat)


class FourierUnit(nn.Module):
    def __init__(self, feat_dim=128, guide_dim=128, mlp_dim=None,
                 NIR_dim=33, d_model=2):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = [256, 128]
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        imnet_in_dim = feat_dim + guide_dim + 2
        self.imnet1 = MLP(imnet_in_dim, out_dim=NIR_dim,
                          hidden_list=mlp_dim)
        self.imnet2 = MLP_P(imnet_in_dim, out_dim=NIR_dim,
                            hidden_list=mlp_dim)

    def query_freq_a(self, feat, coord, hr_guide, mlp):
        b, c, h, w = feat.shape
        _, _, H, W = hr_guide.shape
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
        feat_coord = make_coord(
            (h, w), flatten=False).to(feat.device
        ).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(
            hr_guide, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        rx, ry = 1 / h, 1 / w
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w
                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                pred = mlp(inp.view(B * N, -1)).view(B, N, -1)
                preds.append(pred)
        preds = torch.stack(preds, dim=-1)
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)
               ).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret

    def query_freq_p(self, feat, coord, hr_guide, mlp):
        b, c, h, w = feat.shape
        _, _, H, W = hr_guide.shape
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
        feat_coord = make_coord(
            (h, w), flatten=False).to(feat.device
        ).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(
            hr_guide, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        rx, ry = 1 / h, 1 / w
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w
                inp = torch.cat(
                    [q_feat, q_guide_hr, rel_coord], dim=-1
                ).view(B, -1, H, W)
                pred = mlp(inp).view(B, N, -1)
                preds.append(pred)
        preds = torch.stack(preds, dim=-1)
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)
               ).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret

    def forward(self, feat, coord, hr_guide):
        feat_ffted = torch.fft.fftn(feat, dim=(-2, -1))
        guide_ffted = torch.fft.fftn(hr_guide, dim=(-2, -1))
        feat_mag = torch.abs(feat_ffted)
        feat_pha = torch.angle(feat_ffted)
        guide_mag = torch.abs(guide_ffted)
        guide_pha = torch.angle(guide_ffted)
        ffted_mag = self.query_freq_a(
            feat_mag, coord, guide_mag, self.imnet1)
        ffted_pha = self.query_freq_p(
            feat_pha, coord, guide_pha, self.imnet2)
        real = ffted_mag * torch.cos(ffted_pha)
        imag = ffted_mag * torch.sin(ffted_pha)
        ffted = torch.complex(real, imag)
        output = torch.fft.ifftn(ffted, dim=(-2, -1))
        output = torch.abs(output)
        return output


class FeINFNet(nn.Module):
    def __init__(self, hsi_dim=31, msi_dim=3, feat_dim=128,
                 guide_dim=128, spa_edsr_num=4, spe_edsr_num=4,
                 mlp_dim=None, NIR_dim=33, d_model=2, scale=4):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = [256, 128]
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.NIR_dim = NIR_dim
        self.d_model = d_model
        self.scale = scale

        self.spatial_encoder = make_edsr_baseline(
            n_resblocks=spa_edsr_num, n_feats=guide_dim,
            n_colors=hsi_dim + msi_dim)
        self.spectral_encoder = make_edsr_baseline(
            n_resblocks=spe_edsr_num, n_feats=feat_dim,
            n_colors=hsi_dim)

        imnet_in_dim = feat_dim + guide_dim + feat_dim + 2
        self.imnet = MLP(imnet_in_dim, out_dim=NIR_dim,
                         hidden_list=mlp_dim)
        self.hp = hightfre(in_channels=feat_dim, groups=1)
        self.decoder = ImplicitDecoder(
            in_channels=NIR_dim - 1, freq_dim=NIR_dim - 1,
            hidden_dims=[128, 128, 128], omega=30, scale=10.0)
        self.pe = PositionalEmbedding(d_model, max_len=4096)
        self.freq_query = FourierUnit(
            feat_dim, guide_dim, mlp_dim, NIR_dim)

    def query(self, feat, coord, hr_guide):
        b, c, h, w = feat.shape
        _, _, H, W = hr_guide.shape
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
        feat_coord = make_coord(
            (h, w), flatten=False).to(feat.device
        ).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)
        q_guide_hr = F.grid_sample(
            hr_guide, coord.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        rx, ry = 1 / h, 1 / w
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                hp = self.hp(feat)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                hp_feat = F.grid_sample(
                    hp, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w
                rel_coord = self.pe(rel_coord)
                inp = torch.cat(
                    [q_feat, q_guide_hr, hp_feat, rel_coord], dim=-1)
                pred = self.imnet(
                    inp.view(B * N, -1)).view(B, N, -1)
                preds.append(pred)
        preds = torch.stack(preds, dim=-1)
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)
               ).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)
        return ret

    def forward(self, HR_MSI, lms, LR_HSI):
        _, _, H, W = HR_MSI.shape
        coord = make_coord([H, W]).to(HR_MSI.device)
        feat = torch.cat([HR_MSI, lms], dim=1)
        hr_spa = self.spatial_encoder(feat)
        lr_spe = self.spectral_encoder(LR_HSI)
        freq_feature = self.freq_query(lr_spe, coord, hr_spa)
        NIR_feature = self.query(lr_spe, coord, hr_spa)
        output = self.decoder(NIR_feature, freq_feature)
        output = lms + output
        return output


###############################################################################
# EMA
###############################################################################

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1 - self.decay) * param.data)

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


###############################################################################
# SSIM Loss
###############################################################################

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channels=31):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * 1.5 ** 2))
            for x in range(window_size)])
        gauss = gauss / gauss.sum()
        _2d = gauss.unsqueeze(1).mm(gauss.unsqueeze(0))
        window = _2d.unsqueeze(0).unsqueeze(0).expand(
            channels, 1, window_size, window_size).contiguous()
        self.register_buffer('window', window)

    def forward(self, pred, target):
        pad = self.window_size // 2
        mu1 = F.conv2d(pred, self.window, padding=pad,
                       groups=self.channels)
        mu2 = F.conv2d(target, self.window, padding=pad,
                       groups=self.channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        s1 = F.conv2d(pred * pred, self.window, padding=pad,
                       groups=self.channels) - mu1_sq
        s2 = F.conv2d(target * target, self.window, padding=pad,
                       groups=self.channels) - mu2_sq
        s12 = F.conv2d(pred * target, self.window, padding=pad,
                        groups=self.channels) - mu1_mu2
        cs = (2 * s12 + self.C2) / (s1 + s2 + self.C2)
        ssim_map = ((2 * mu1_mu2 + self.C1) /
                    (mu1_sq + mu2_sq + self.C1)) * cs
        return 1 - ssim_map.mean()


###############################################################################
# Evaluation Metrics
###############################################################################

def calc_psnr(pred, gt):
    return -10.0 * torch.log10(
        torch.mean((pred - gt) ** 2) + 1e-8).item()


def calc_ssim(pred, gt, channels=31, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ax = torch.arange(window_size, device=pred.device,
                      dtype=pred.dtype) - window_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
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
    s2 = F.conv2d(gt * gt, window, padding=pad,
                   groups=channels) - mu2_sq
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
        if mg > 0:
            ergas += err[:, c].mean() / (mg ** 2)
    return math.sqrt(ergas / C) * 100.0 * scale


###############################################################################
# Training
###############################################################################

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    print("\nBuilding FeINFNet...")
    model = FeINFNet(
        hsi_dim=args.bands, msi_dim=3, feat_dim=128, guide_dim=128,
        spa_edsr_num=4, spe_edsr_num=4, mlp_dim=[256, 128],
        NIR_dim=33, d_model=2, scale=args.scale).to(device)
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    l1_loss = nn.L1Loss().to(device)
    ssim_loss = SSIMLoss(window_size=11, channels=args.bands).to(device)

    best_psnr = 0.0
    best_epoch = 0
    save_dir = os.path.join(args.output_dir, "feinfn_checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch: {args.batch_size}, Patch: {args.patch_size}, "
          f"LR: {args.lr}, EMA: {args.ema_decay}")
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
            lms = F.interpolate(lr_hsi, scale_factor=args.scale,
                                mode='bilinear', align_corners=False)

            output = model(hr_msi, lms, lr_hsi)
            output = output.clip(0, 1)

            loss_l1 = l1_loss(output, gt)
            loss_ssim = ssim_loss(output, gt)
            loss = (args.l1_weight * loss_l1
                    + args.ssim_weight * loss_ssim)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()
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
            ema.apply_shadow()
            model.eval()
            psnr_l, ssim_l, sam_l, ergas_l = [], [], [], []

            with torch.no_grad():
                for i in range(len(test_ds.scenes)):
                    item = test_ds[i]
                    gt_e = item["gt"].unsqueeze(0).to(device)
                    lr_e = item["lr_hsi"].unsqueeze(0).to(device)
                    hr_e = item["hr_msi"].unsqueeze(0).to(device)
                    lms_e = F.interpolate(
                        lr_e, scale_factor=args.scale,
                        mode='bilinear', align_corners=False)
                    pred = model(hr_e, lms_e, lr_e).clip(0, 1)
                    psnr_l.append(calc_psnr(pred, gt_e))
                    ssim_l.append(calc_ssim(pred, gt_e))
                    sam_l.append(calc_sam(pred, gt_e))
                    ergas_l.append(calc_ergas(pred, gt_e, args.scale))

            ema.restore()
            model.train()

            avg_psnr = np.mean(psnr_l)
            avg_ssim = np.mean(ssim_l)
            avg_sam = np.mean(sam_l)
            avg_ergas = np.mean(ergas_l)
            print(f"  [Eval] PSNR: {avg_psnr:.4f} dB | "
                  f"SSIM: {avg_ssim:.4f} | "
                  f"SAM: {avg_sam:.4f} deg | "
                  f"ERGAS: {avg_ergas:.4f}")

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                ckpt = os.path.join(save_dir, "feinfn_best.pth")
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


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FeINFN Training for CAVE x4")
    parser.add_argument(
        "--root", type=str,
        default="/kaggle/input/datasets/liptee/"
                "hyperspectral-image-restoration-based-on-cave",
        help="Dataset root directory")
    parser.add_argument("--bands", type=int, default=31)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--output_dir", type=str,
        default="/kaggle/working",
        help="Directory to save checkpoints")
    args = parser.parse_args()
    train(args)
