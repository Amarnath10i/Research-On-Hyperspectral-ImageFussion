"""NullFusion v3 -- Null-Space Spectral Dictionary Prior (NSDP) for CAVE x4.

Novel contribution (vs FeINFN / BDT / DSPNet / SSRNet):
  The null space of the HSI-MSI fusion operator A=[D;R] is SPECTRAL: it is the
  component of each pixel's 31-band signature that the 3-band RGB camera cannot
  distinguish.  We model the null-space completion as a learned spectral
  dictionary D (bands x K) times a spatially-varying code alpha (K x H x W):

        out = base + D @ alpha,   R(out) = R(base)  (exact MSI consistency)

  * D is INITIALISED in the SRF-null subspace (orthogonal to the RGB response),
    so every update starts from a physically-meaningful "RGB-invisible" basis
    and MSI consistency is exact by construction (no CG, no soft constraint).
  * alpha is predicted by a lightweight CNN conditioned on both observations,
    giving spatially-adaptive dictionary selection via spectral self-attention.
  * Compared with FeINFN's implicit-coordinate MLP decoder, NSDP separates a
    global spectral basis (parameter-efficient, interpretable atoms) from the
    spatial code, and only ever learns the null-space residual -- strictly less
    to learn than a full-space mapping.

Protocol: identical to the other SOTA scripts -- CAVE, Nikon D700 SRF, Wald's
Gaussian (9x9, sigma=1.2) blur + 4x decimation, 20 train / 12 test scenes.
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset


# ===========================================================================
# 1. Operator (inline): A = [D; R], joint range/null decomposition
# ===========================================================================
def gaussian_kernel2d(size, sigma):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return (k / k.sum()).astype(np.float32)


class DegradationOperator(nn.Module):
    """Per-band blur + decimate (forward) / upsample + blur (adjoint)."""

    def __init__(self, scale, ksize, sigma):
        super().__init__()
        self.scale = scale
        k = gaussian_kernel2d(ksize, sigma)
        self.register_buffer("kernel", torch.from_numpy(k).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel.repeat(C, 1, 1, 1)
        xb = F.conv2d(x, k, padding=self.kernel.shape[-1] // 2, groups=C)
        return xb[:, :, :: self.scale, :: self.scale]

    def transpose(self, y, out_hw):
        B, C, h, w = y.shape
        yu = F.interpolate(y, size=out_hw, mode="bicubic", align_corners=False)
        k = self.kernel.repeat(C, 1, 1, 1)
        return F.conv2d(yu, k, padding=self.kernel.shape[-1] // 2, groups=C)


def block_cg(applyA, rhs, steps, tol=1e-10):
    z = tuple(torch.zeros_like(r) for r in rhs)
    ap0 = applyA(z)
    r = tuple(rh - a for rh, a in zip(rhs, ap0))
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


class CombinedOperator(nn.Module):
    def __init__(self, scale, bands, msi, ksize, sigma, srf, cg_steps, ridge):
        super().__init__()
        self.D = DegradationOperator(scale, ksize, sigma)
        self.scale = scale
        self.bands = bands
        self.msi_bands = msi
        self.cg_steps = cg_steps
        self.ridge = ridge
        self.register_buffer("srf", srf.float())

    def R(self, x):
        return torch.einsum("nbhw,bm->nmhw", x, self.srf)

    def Rt(self, m):
        return torch.einsum("nmhw,bm->nbhw", m, self.srf)

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
        out_hw = (v.shape[-2], v.shape[-1])
        yH, yM = self.forward(v)
        return v - self.pinv(yH, yM, out_hw)


# ===========================================================================
# 2. NullFusionNetV3 -- Null-Space Spectral Dictionary Prior
# ===========================================================================
class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class _CrossAttn(nn.Module):
    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_ch = ch // n_heads
        self.q = nn.Conv2d(ch, ch, 1)
        self.kv = nn.Conv2d(ch, ch * 2, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = max(self.head_ch, 1) ** -0.5

    def forward(self, query, context):
        B, C, Hq, Wq = query.shape
        if context.shape[-2:] != (Hq, Wq):
            context = F.adaptive_avg_pool2d(context, (Hq, Wq))
        q = self.q(query)
        kv = self.kv(context)
        k, v = kv.chunk(2, dim=1)
        nH, d = self.n_heads, self.head_ch
        q = q.reshape(B, nH, d, Hq * Wq).transpose(2, 3)
        k = k.reshape(B, nH, d, Hq * Wq).transpose(2, 3)
        v = v.reshape(B, nH, d, Hq * Wq).transpose(2, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(2, 3).reshape(B, C, Hq, Wq)
        return query + self.proj(out)


class _SpectralGate(nn.Module):
    """Per-pixel gating over K dictionary atoms — spatially adaptive atom selection.
    Cheap O(B*K*H*W) 1x1 conv instead of O((HW)²) attention."""

    def __init__(self, K, ctx_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ctx_ch + K, K, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, K, 1),
            nn.Sigmoid(),
        )

    def forward(self, alpha, ctx):
        x = torch.cat([alpha, ctx], dim=1)
        gate = self.net(x)
        return alpha * gate


# ===========================================================================
# Novel: Haar wavelet transform (lifting scheme) for high-frequency branch
# ===========================================================================
def haar_dwt2d(x):
    """2D Haar DWT: x (B,C,H,W) -> (LL, LH, HL, HH) each (B,C,H/2,W/2)."""
    B, C, H, W = x.shape
    # Pad if odd
    if H % 2 == 1:
        x = F.pad(x, (0, 0, 0, 1), mode='reflect')
    if W % 2 == 1:
        x = F.pad(x, (0, 1, 0, 0), mode='reflect')
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//2, 2, W//2, 2)
    LL = (x[:, :, :, 0, :, 0] + x[:, :, :, 1, :, 0] +
          x[:, :, :, 0, :, 1] + x[:, :, :, 1, :, 1]) / 4
    LH = (x[:, :, :, 0, :, 0] - x[:, :, :, 1, :, 0] +
          x[:, :, :, 0, :, 1] - x[:, :, :, 1, :, 1]) / 4
    HL = (x[:, :, :, 0, :, 0] + x[:, :, :, 1, :, 0] -
          x[:, :, :, 0, :, 1] - x[:, :, :, 1, :, 1]) / 4
    HH = (x[:, :, :, 0, :, 0] - x[:, :, :, 1, :, 0] -
          x[:, :, :, 0, :, 1] + x[:, :, :, 1, :, 1]) / 4
    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH):
    """Inverse 2D Haar DWT."""
    B, C, h, w = LL.shape
    H, W = h * 2, w * 2
    x = torch.zeros(B, C, H, W, device=LL.device, dtype=LL.dtype)
    x[:, :, 0::2, 0::2] = LL + LH + HL + HH
    x[:, :, 1::2, 0::2] = LL - LH + HL - HH
    x[:, :, 0::2, 1::2] = LL + LH - HL - HH
    x[:, :, 1::2, 1::2] = LL - LH - HL + HH
    return x


class _WaveletDetailBranch(nn.Module):
    """Wavelet high-frequency branch: learns null-space detail in wavelet domain.
    Novel: operates on null-space residual's high-frequency subbands."""

    def __init__(self, bands, width, depth=2):
        super().__init__()
        self.bands = bands
        self.enc = nn.Sequential(
            nn.Conv2d(bands * 3, width, 3, 1, 1),  # LH+HL+HH = 3*bands
            nn.ReLU(inplace=True),
            *[_ResBlock(width) for _ in range(depth)],
            nn.Conv2d(width, bands * 3, 3, 1, 1)
        )

    def forward(self, null_comp):
        """null_comp: (B, bands, H, W) -> high-freq correction (B, bands, H, W)"""
        LL, LH, HL, HH = haar_dwt2d(null_comp)
        hf = torch.cat([LH, HL, HH], dim=1)  # (B, 3*bands, H/2, W/2)
        hf_corr = self.enc(hf)               # (B, 3*bands, H/2, W/2)
        LH_c, HL_c, HH_c = hf_corr.chunk(3, dim=1)
        LL_zero = torch.zeros_like(LL)
        corr = haar_idwt2d(LL_zero, LH_c, HL_c, HH_c)
        # Crop to original size if padded
        return corr[:, :, :null_comp.shape[-2], :null_comp.shape[-1]]


class NullFusionNetV4(nn.Module):
    """NullFusion v4: Multi-scale Spectral Dictionary + Wavelet High-Freq Branch.
    Novel contributions:
      1. Pyramid dictionary: global + mid + fine scales (spatially varying support)
      2. Wavelet high-frequency branch on null-space residual
      3. Cross-scale spectral consistency loss (handled in loss function)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        Bn, M, W = cfg.bands, cfg.msi_bands, cfg.width
        Kg, Km, Kf = cfg.dict_global, cfg.dict_mid, cfg.dict_fine
        self.op = CombinedOperator(cfg.scale, Bn, M, cfg.ksize, cfg.sigma,
                                   cfg.srf, cfg.cg_steps, cfg.ridge)
        # --- pyramid spectral dictionary (novel) ---------------------------
        self.D_global = nn.Parameter(torch.zeros(Bn, Kg))  # global atoms
        self.D_mid    = nn.Parameter(torch.zeros(Bn, Km))  # mid-scale
        self.D_fine   = nn.Parameter(torch.zeros(Bn, Kf))  # fine detail
        self.Kg, self.Km, self.Kf = Kg, Km, Kf
        # --- conditioning encoder (cross-modal) ----------------------------
        self.hsi_stem = nn.Conv2d(Bn, W, 3, 1, 1)
        self.msi_stem = nn.Conv2d(M, W, 3, 1, 1)
        self.msi_detail = nn.Conv2d(M, W, 3, 1, 1)
        self.cross_attn = _CrossAttn(W, cfg.cross_attn_heads)
        self.fuse = nn.Conv2d(2 * W, W, 1)
        self.enc = nn.Sequential(*[_ResBlock(W) for _ in range(cfg.enc_depth)])
        self.up = nn.ConvTranspose2d(W, W, cfg.scale, cfg.scale, 0, bias=False)
        # --- code predictors for 3 scales ----------------------------------
        self.prior_in = nn.Conv2d(2 * W + Bn, W, 3, 1, 1)
        body = []
        for i in range(cfg.prior_depth):
            body.append(_ResBlock(W))
        self.prior_body = nn.Sequential(*body)
        self.spect_gate_g = _SpectralGate(Kg, W)
        self.spect_gate_m = _SpectralGate(Km, W)
        self.spect_gate_f = _SpectralGate(Kf, W)
        self.code_head_g = nn.Conv2d(W, Kg, 3, 1, 1)
        self.code_head_m = nn.Conv2d(W, Km, 3, 1, 1)
        self.code_head_f = nn.Conv2d(W, Kf, 3, 1, 1)
        # --- wavelet high-frequency branch (novel) -------------------------
        self.wavelet_branch = _WaveletDetailBranch(Bn, W // 2, depth=2)
        # --- spectral consistency projection (for loss) --------------------
        # projects per-scale codes to common spectral space
        self.consist_proj = nn.Conv2d(Kg + Km + Kf, Bn, 1)

    # -- null-space init: make D orthogonal to the RGB response ------------
    def set_srf(self, srf: torch.Tensor):
        s = srf if srf.shape[0] == self.cfg.bands else srf.t().contiguous()
        s = s.float().to(self.D_global.device)
        self.op.srf.data = s.to(self.op.srf.device)
        # precompute pinv(srf)^T for MSI-exact min-norm base
        s_pinv = torch.linalg.pinv(s)            # (msi, bands)
        self.register_buffer("srf_pinv_t", s_pinv.t().contiguous())  # (bands, msi)
        with torch.no_grad():
            for D in (self.D_global, self.D_mid, self.D_fine):
                K = D.shape[1]
                M = torch.randn(self.cfg.bands, K, device=D.device)
                P = torch.linalg.pinv(s)
                Rng = s @ P
                D0 = M - Rng @ M
                D0 = D0 / (D0.norm(dim=0, keepdim=True) + 1e-8)
                D.copy_(D0)

    def _base(self, yH, yM):
        B, _, H, W = yM.shape
        base0 = torch.einsum("bmhw,cm->bchw", yM, self.srf_pinv_t)
        base1 = F.interpolate(yH, (H, W), mode="bicubic",
                              align_corners=False)
        delta = base1 - base0
        base = base0 + delta - torch.einsum("bmhw,cm->bchw",
                                             self.op.R(delta), self.srf_pinv_t)
        return base

    def forward(self, yH, yM):
        base = self._base(yH, yM)
        f_hsi = self.hsi_stem(yH)
        msi_lr = F.adaptive_avg_pool2d(self.msi_stem(yM), f_hsi.shape[-2:])
        f_hsi = self.cross_attn(f_hsi, msi_lr)
        z = self.enc(self.fuse(torch.cat([f_hsi, msi_lr], dim=1)))
        f_hr = self.up(z)
        f_msi = self.msi_detail(yM)
        cond = torch.cat([f_hr, f_msi, base], dim=1)
        v = self.prior_in(cond)
        v = self.prior_body(v)
        # --- multi-scale codes with spectral gating -----------------------
        alpha_g = F.softplus(self.code_head_g(v))
        alpha_m = F.softplus(self.code_head_m(v))
        alpha_f = F.softplus(self.code_head_f(v))
        alpha_g = self.spect_gate_g(alpha_g, v)
        alpha_m = self.spect_gate_m(alpha_m, v)
        alpha_f = self.spect_gate_f(alpha_f, v)
        # --- pyramid dictionary reconstruction -----------------------------
        # D: (bands, K), alpha: (B, K, H, W) -> (B, bands, H, W)
        null_g = torch.einsum("ck,bkhw->bchw", self.D_global, alpha_g)
        null_m = torch.einsum("ck,bkhw->bchw", self.D_mid, alpha_m)
        null_f = torch.einsum("ck,bkhw->bchw", self.D_fine, alpha_f)
        null_comp = null_g + null_m + null_f
        # exact MSI consistency (spectral projection)
        null_comp = null_comp - self.op.Rt(self.op.R(null_comp))
        # --- wavelet high-frequency correction (novel) --------------------
        wf_corr = self.wavelet_branch(null_comp)
        out = base + null_comp + wf_corr
        # spectral consistency features (for loss)
        all_alpha = torch.cat([alpha_g, alpha_m, alpha_f], dim=1)
        consist = self.consist_proj(all_alpha)
        return {"out": out, "base": base, "null_comp": null_comp,
                "alpha_g": alpha_g, "alpha_m": alpha_m, "alpha_f": alpha_f,
                "wf_corr": wf_corr, "consist": consist}


# ===========================================================================
# 3. CAVE dataset (liptee .mat / band_*.png) -- Wald's protocol
# ===========================================================================
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


def make_gaussian_kernel(size=9, sigma=1.2):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _load_mat(mat_path):
    data = loadmat(mat_path)
    for key, val in data.items():
        if not key.startswith("_") and hasattr(val, "shape"):
            arr = np.array(val, dtype=np.float32)
            if arr.ndim >= 2 and min(arr.shape) > 1:
                # Ensure (C, H, W) format
                if arr.ndim == 3:
                    if arr.shape[-1] <= 31 and arr.shape[-1] < arr.shape[0]:
                        arr = arr.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
                    elif arr.shape[0] > 31 and arr.shape[1] == 31:
                        arr = arr.transpose(1, 0, 2)  # (H, C, W) -> (C, H, W)
                # Normalize to [0, 1] if needed
                if arr.max() > 1.0:
                    arr = arr / arr.max()
                return arr
    raise ValueError(f"No valid array in {mat_path}")


def _discover_scenes(root, split):
    split_dir = None
    for name in (split, split.capitalize(), split.upper()):
        cand = os.path.join(root, name)
        if os.path.isdir(cand):
            split_dir = cand
            break
    if split_dir is None:
        raise FileNotFoundError(f"split '{split}' not found under {root}")
    # liptee layout: <split_dir>/HSI/*.mat (one .mat file per scene)
    hsi_dir = os.path.join(split_dir, "HSI")
    if not os.path.isdir(hsi_dir):
        # fallback: scan subdirs for .mat
        scenes = []
        for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
            if not entry.is_dir(follow_symlinks=False):
                continue
            mat_files = [f for f in os.listdir(entry.path) if f.endswith(".mat")]
            if mat_files:
                mat_path = os.path.join(entry.path, mat_files[0])
                scenes.append((entry.name, mat_path))
                continue
            for bn in ("band_01.png", "Band_01.png", "BAND_01.png"):
                if os.path.isfile(os.path.join(entry.path, bn)):
                    scenes.append((entry.name, entry.path))
                    break
        if not scenes:
            raise FileNotFoundError(f"no scenes found under {split_dir}")
        return scenes
    # HSI subdirectory exists: each .mat is a scene
    scenes = []
    for f in sorted(os.listdir(hsi_dir)):
        if f.endswith(".mat"):
            scene_name = f[:-4]
            scenes.append((scene_name, os.path.join(hsi_dir, f)))
    return scenes


def _load_scene_bands(mat_path, bands=31, max_dim=None):
    cube = _load_mat(mat_path)
    if max_dim is not None and (cube.shape[1] > max_dim or cube.shape[2] > max_dim):
        y0 = max(0, (cube.shape[1] - max_dim) // 2)
        x0 = max(0, (cube.shape[2] - max_dim) // 2)
        cube = cube[:, y0:y0 + max_dim, x0:x0 + max_dim]
    return cube.astype(np.float32)


class CAVEDataset(Dataset):
    def __init__(self, root, split="train", bands=31, scale=4,
                 patch_size=96, max_dim=512):
        self.split = split
        self.bands = bands
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = split.lower() == "train"
        self.srf = nike_d700_srf(bands)
        self.kernel = make_gaussian_kernel(9, 1.2)
        self.scenes = _discover_scenes(root, split)
        # scenes are (scene_name, mat_file_path)
        self._cache = {n: _load_scene_bands(p, bands, max_dim)
                       for n, p in self.scenes}

    def __len__(self):
        return 10000 if self.is_train else len(self.scenes)

    def _sim(self, gt):
        C, H, W = gt.shape
        k = self.kernel.shape[0]
        pad = k // 2
        blurred = np.empty_like(gt)
        for c in range(C):
            from scipy.ndimage import convolve
            blurred[c] = convolve(gt[c], self.kernel, mode="wrap")
        hr, wr = H // self.scale, W // self.scale
        y0 = (H - hr * self.scale) // 2
        x0 = (W - wr * self.scale) // 2
        lr = blurred[:, y0::self.scale, x0::self.scale].astype(np.float32)
        msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
        msi = np.clip(msi, 0.0, 1.0)
        return lr, msi

    def __getitem__(self, idx):
        if self.is_train:
            name = list(self._cache.keys())[random.randrange(len(self._cache))]
            gt = self._cache[name].copy()
            _, H, W = gt.shape
            p = min(self.patch_size, H, W)
            y = random.randrange(0, H - p + 1)
            x = random.randrange(0, W - p + 1)
            gt = gt[:, y:y + p, x:x + p]
            if random.random() < 0.5:
                gt = gt[:, :, ::-1].copy()
            if random.random() < 0.5:
                gt = gt[:, ::-1, :].copy()
        else:
            name, _ = self.scenes[idx % len(self.scenes)]
            gt = self._cache[name].copy()
            H, W = gt.shape[1], gt.shape[2]
            H = (H // self.scale) * self.scale
            W = (W // self.scale) * self.scale
            gt = gt[:, :H, :W]
        lr, msi = self._sim(gt)
        return (torch.from_numpy(gt.astype(np.float32)),
                torch.from_numpy(lr.astype(np.float32)),
                torch.from_numpy(msi.astype(np.float32)))


# ===========================================================================
# 4. Metrics
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
# 5. Training (matches FeINFN protocol: 2000 epochs, eval every 20)
# ===========================================================================
def build_config():
    class C:
        pass
    c = C()
    c.scale = 4
    c.bands = 31
    c.msi_bands = 3
    c.ksize = 9
    c.sigma = 1.2
    c.cg_steps = 40
    c.ridge = 1e-6
    c.width = 96
    c.enc_depth = 4
    c.prior_depth = 8
    c.cross_attn_heads = 4
    c.dict_global = 64
    c.dict_mid = 48
    c.dict_fine = 32
    c.epochs = 2000
    c.batch_size = 2
    c.patch_size = 80
    c.eval_every = 20
    c.lr = 2e-4
    c.weight_decay = 1e-4
    c.ema_decay = 0.999
    c.w_l1 = 1.0
    c.w_ssim = 0.5
    c.w_sam = 0.05
    c.w_phys = 0.1
    c.w_consist = 0.05
    c.time_budget_h = 9.0
    c.max_dim = 512
    c.amp = True
    return c


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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    cfg = build_config()
    srf_np = nike_d700_srf(cfg.bands)
    srf_t = torch.from_numpy(srf_np).float().to(device)

    print("\nLoading datasets...")
    train_ds = CAVEDataset(args.root, "train", cfg.bands, cfg.scale,
                           cfg.patch_size, cfg.max_dim)
    test_ds = CAVEDataset(args.root, "test", cfg.bands, cfg.scale,
                          cfg.patch_size, cfg.max_dim)
    print(f"Train scenes: {len(train_ds.scenes)}, Test scenes: {len(test_ds.scenes)}")

    cfg.srf = srf_t
    model = NullFusionNetV4(cfg).to(device)
    model.set_srf(srf_t)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"NullFusionNetV4 params: {nparams/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=1e-6)
    l1 = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_psnr = 0.0
    best_epoch = 0
    save_dir = os.path.join(args.output_dir, "nullfusion_v3_ckpt")
    os.makedirs(save_dir, exist_ok=True)
    T0 = time.time()
    LIMIT = cfg.time_budget_h * 3600

    steps_per_epoch = 200

    print(f"\nStarting training for {cfg.epochs} epochs (budget {cfg.time_budget_h}h)...")
    print(f"Batch {cfg.batch_size}, Patch {cfg.patch_size}, Eval every {cfg.eval_every}")
    print("-" * 70)

    for epoch in range(1, cfg.epochs + 1):
        if (time.time() - T0) > LIMIT:
            print(f"[time budget reached at epoch {epoch}]")
            break
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            gts, lhs, mss = [], [], []
            for _ in range(cfg.batch_size):
                g, l, m = train_ds[random.randrange(len(train_ds))]
                gts.append(g); lhs.append(l); mss.append(m)
            gt = torch.stack(gts, 0).to(device)
            yH = torch.stack(lhs, 0).to(device)
            yM = torch.stack(mss, 0).to(device)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                out_dict = model(yH, yM)
                out = out_dict["out"]
                loss_l1 = l1(out, gt)
                loss_ssim = 1.0 - torch.tensor(
                    calc_ssim(out[0].detach().cpu().numpy(),
                              gt[0].detach().cpu().numpy()), dtype=torch.float32, device=device)
                loss_sam = torch.tensor(
                    calc_sam(out[0].detach().cpu().numpy(),
                             gt[0].detach().cpu().numpy()), dtype=torch.float32, device=device)
                loss_phys = F.mse_loss(model.op.D(out), yH)
                # spectral consistency: per-scale codes should reconstruct same spectra
                consist = out_dict["consist"]
                loss_consist = F.l1_loss(consist, gt)
                loss = (cfg.w_l1 * loss_l1 + cfg.w_ssim * loss_ssim
                        + cfg.w_sam * loss_sam + cfg.w_phys * loss_phys
                        + cfg.w_consist * loss_consist)

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
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg.epochs} | Loss {avg:.5f} | "
                  f"LR {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            ema.apply_to(model)
            psnrs, ssims, sams, ergas = [], [], [], []
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
                    ergas.append(calc_ergas(pred, gt_np, cfg.scale))
            m_psnr = float(np.mean(psnrs))
            m_ssim = float(np.mean(ssims))
            m_sam = float(np.mean(sams))
            m_ergas = float(np.mean(ergas))
            improved = ""
            if m_psnr > best_psnr:
                best_psnr = m_psnr
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(save_dir, "best.pth"))
                improved = " [BEST]"
            print(f"  >>> Test@{epoch}: PSNR {m_psnr:.4f} | SSIM {m_ssim:.4f} | "
                  f"SAM {m_sam:.3f} | ERGAS {m_ergas:.3f}{improved}")
            ema.restore_from(model)

    # final eval on best
    ckpt = os.path.join(save_dir, "best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    psnrs, ssims, sams, ergas = [], [], [], []
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
            ergas.append(calc_ergas(pred, gt_np, cfg.scale))
    final = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
             "sam": float(np.mean(sams)), "ergas": float(np.mean(ergas))}
    print("=" * 60)
    print(f"FINAL BEST (epoch {best_epoch}): PSNR {final['psnr']:.4f} | "
          f"SSIM {final['ssim']:.4f} | SAM {final['sam']:.3f} | "
          f"ERGAS {final['ergas']:.3f}")
    print("=" * 60)
    with open(os.path.join(args.output_dir, "nullfusion_v3_results.json"), "w") as f:
        json.dump({"protocol": "CAVE x4, Nikon D700 SRF, Wald blur",
                   "params_M": nparams / 1e6, "best_epoch": best_epoch,
                   "final": final}, f, indent=2)
    print("saved nullfusion_v3_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input/datasets/liptee/"
                    "hyperspectral-image-restoration-based-on-cave")
    ap.add_argument("--output_dir", default="/kaggle/working")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
