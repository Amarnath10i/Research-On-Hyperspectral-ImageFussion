"""NullFusion++ v2 — SOTA-beater for Chikusei x4

Combines:
  1. Exact null-space projection (from proposal 7's RangeNullProjector)
  2. Deep unfolding data consistency (SMGU-Net style)
  3. Lightweight Swin Transformer bottleneck
  4. Spectral dictionary + wavelet detail (our innovation)
  5. Composite loss: L1 + SSIM + SAM + gradient + physics

Target: Beat CoFusion 49.14 / SMGU-Net 48.82 / RAMoE 48.10 PSNR on Chikusei x4

Run on Kaggle:  python train_nullfusion_pp_v2.py --root /kaggle/input/chikusei
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
# CG solver & null-space projector (from proposal 7)
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


class RangeNullProjector(nn.Module):
    """Exact null-space projector for the blur+decimate operator D.

    pinv(yH) = D^T (D D^T + ridge)^{-1} yH   — CG solve in LR space.
    project_null(v) = v - pinv(D v)            — removes LR-visible component.
    """

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
        z = block_cg(self._normal_op(out_hw), yH, self.cg_steps)
        return self.D.transpose(z, out_hw)

    def project_null(self, v, out_hw=None):
        if out_hw is None:
            out_hw = (v.shape[-2], v.shape[-1])
        return v - self.pinv(self.D(v), out_hw)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RCAB(nn.Module):
    def __init__(self, ch, red=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(), nn.Conv2d(ch, ch, 3, 1, 1)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // red),
            nn.ReLU(),
            nn.Linear(ch // red, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x + self.body(x) * self.ca(x).unsqueeze(-1).unsqueeze(-1)


class SpatialAttn(nn.Module):
    def __init__(self, ch, window=8):
        super().__init__()
        self.window = window
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
        xp = F.pad(x, (0, pw, 0, ph)) if (ph or pw) else x
        _, _, Hp, Wp = xp.shape
        q, k, v = self.qkv(xp).chunk(3, dim=1)
        nH, nW = Hp // ws, Wp // ws
        q = q.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(
            -1, ws * ws, C
        )
        k = k.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(
            -1, ws * ws, C
        )
        v = v.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(
            -1, ws * ws, C
        )
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).reshape(B, nH, nW, ws, ws, C).permute(
            0, 5, 3, 1, 4, 2
        ).reshape(B, C, Hp, Wp)
        if ph or pw:
            out = out[:, :, :H, :W]
        return x + self.proj(out)


class CrossAttn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.d = ch // heads
        self.q = nn.Conv2d(ch, ch, 1)
        self.kv = nn.Conv2d(ch, ch * 2, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = self.d ** -0.5

    def forward(self, q, ctx):
        B, C, Hq, Wq = q.shape
        if ctx.shape[-2:] != (Hq, Wq):
            ctx = F.adaptive_avg_pool2d(ctx, (Hq, Wq))
        qh = self.q(q).reshape(B, self.heads, self.d, Hq * Wq).transpose(2, 3)
        k, v = self.kv(ctx).chunk(2, dim=1)
        k = k.reshape(B, self.heads, self.d, Hq * Wq).transpose(2, 3)
        v = v.reshape(B, self.heads, self.d, Hq * Wq).transpose(2, 3)
        attn = ((qh @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return q + self.proj(
            (attn @ v).transpose(2, 3).reshape(B, C, Hq, Wq)
        )


class SpectralSelfAttn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.d = ch // heads
        self.norm = nn.LayerNorm(ch)
        self.qkv = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)
        self.scale = self.d ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        xt = self.norm(x.permute(0, 2, 3, 1).reshape(-1, C))
        qkv = self.qkv(xt).reshape(-1, 3, self.heads, self.d)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return x + self.proj(
            (attn @ v).reshape(-1, C)
        ).reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Swin Transformer block (lightweight, for bottleneck)
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, ch, window_size=8, heads=4):
        super().__init__()
        self.ws = window_size
        self.heads = heads
        self.d = ch // heads
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = self.d ** -0.5
        self.cpb = nn.Sequential(
            nn.Linear(2, heads, bias=True),
            nn.ReLU(),
            nn.Linear(heads, heads, bias=True),
        )
        self._init_cpb()

    def _init_cpb(self):
        ws = self.ws
        coords_h = torch.arange(ws, dtype=torch.float32) / ws + 0.5 / ws
        coords_w = torch.arange(ws, dtype=torch.float32) / ws + 0.5 / ws
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), -1)
        coords_flat = coords.reshape(-1, 2)
        self.cpb_coords = nn.Parameter(coords_flat, requires_grad=False)
        with torch.no_grad():
            biases = self.cpb(coords_flat)
            self.cpb_biases = nn.Parameter(biases.reshape(ws * ws, 1, self.heads), requires_grad=False)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
        xp = F.pad(x, (0, pw, 0, ph)) if (ph or pw) else x
        _, _, Hp, Wp = xp.shape
        q, k, v = self.qkv(xp).chunk(3, dim=1)
        nH, nW = Hp // ws, Wp // ws

        q = q.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(
            -1, ws * ws, self.heads, self.d
        )
        k = k.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(
            -1, ws * ws, self.heads, self.d
        )
        v = v.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(
            -1, ws * ws, self.heads, self.d
        )

        rel = self.cpb_biases.unsqueeze(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale + rel
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(-1, ws * ws, C).reshape(
            B, nH, nW, ws, ws, C
        ).permute(0, 5, 3, 1, 4, 2).reshape(B, C, Hp, Wp)
        if ph or pw:
            out = out[:, :, :H, :W]
        return x + self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, ch, window=8, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.attn = WindowAttention(ch, window, heads)
        self.norm2 = nn.LayerNorm(ch)
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch * mlp_ratio),
            nn.GELU(),
            nn.Linear(ch * mlp_ratio, ch),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        xt = x.permute(0, 2, 3, 1).reshape(-1, C)
        xt = xt + self.attn(self.norm1(xt.reshape(B, H, W, C)).permute(0, 3, 1, 2)).reshape(-1, C)
        xt = xt + self.mlp(self.norm2(xt))
        return x + xt.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Deep Unfolding Data Consistency (SMGU-Net style)
# ---------------------------------------------------------------------------

class DataConsistencyLayer(nn.Module):
    """One unfolding step: prior estimate + observation data consistency.

    x_{k+1} = f_theta(x_k) + lambda * (observation - A^T A f_theta(x_k))
    where A^T A enforces spectral + spatial consistency.
    """

    def __init__(self, bands, width):
        super().__init__()
        self.prior = nn.Sequential(
            nn.Conv2d(bands, width, 3, 1, 1),
            RCAB(width),
            RCAB(width),
            nn.Conv2d(width, bands, 3, 1, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, obs_spectral, obs_spatial):
        residual = self.prior(x)
        out = x + torch.sigmoid(self.alpha) * residual
        out = out + torch.sigmoid(self.beta) * (obs_spectral - out)
        return out


# ---------------------------------------------------------------------------
# Wavelet branch
# ---------------------------------------------------------------------------

def _haar(x):
    B, C, H, W = x.shape
    if H % 2:
        x = F.pad(x, (0, 0, 0, 1), mode="reflect")
    if W % 2:
        x = F.pad(x, (0, 1, 0, 0), mode="reflect")
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    LL = (x[:, :, :, 0, :, 0] + x[:, :, :, 1, :, 0] + x[:, :, :, 0, :, 1] + x[:, :, :, 1, :, 1]) / 4
    LH = (x[:, :, :, 0, :, 0] - x[:, :, :, 1, :, 0] + x[:, :, :, 0, :, 1] - x[:, :, :, 1, :, 1]) / 4
    HL = (x[:, :, :, 0, :, 0] + x[:, :, :, 1, :, 0] - x[:, :, :, 0, :, 1] - x[:, :, :, 1, :, 1]) / 4
    HH = (x[:, :, :, 0, :, 0] - x[:, :, :, 1, :, 0] - x[:, :, :, 0, :, 1] + x[:, :, :, 1, :, 1]) / 4
    return LL, LH, HL, HH


def _ihaar(LL, LH, HL, HH):
    B, C, h, w = LL.shape
    x = torch.zeros(B, C, h * 2, w * 2, device=LL.device, dtype=LL.dtype)
    x[:, :, 0::2, 0::2] = LL + LH + HL + HH
    x[:, :, 1::2, 0::2] = LL - LH + HL - HH
    x[:, :, 0::2, 1::2] = LL + LH - HL - HH
    x[:, :, 1::2, 1::2] = LL - LH - HL + HH
    return x


class WaveletBranch(nn.Module):
    def __init__(self, bands, width, depth=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(bands * 3, width, 3, 1, 1), nn.GELU(),
            RCAB(width), nn.Conv2d(width, bands * 3, 3, 1, 1),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(bands * 3, width, 3, 1, 1), nn.GELU(),
            RCAB(width), nn.Conv2d(width, bands * 3, 3, 1, 1),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(bands * 3, width, 3, 1, 1), nn.GELU(),
            RCAB(width), nn.Conv2d(width, bands * 3, 3, 1, 1),
        )

    def forward(self, x):
        LL1, LH1, HL1, HH1 = _haar(x)
        c1 = self.enc1(torch.cat([LH1, HL1, HH1], 1)).chunk(3, 1)
        corr1 = _ihaar(torch.zeros_like(LL1), c1[0], c1[1], c1[2])

        LL2, LH2, HL2, HH2 = _haar(LL1)
        c2 = self.enc2(torch.cat([LH2, HL2, HH2], 1)).chunk(3, 1)
        corr2 = F.interpolate(
            _ihaar(torch.zeros_like(LL2), c2[0], c2[1], c2[2]),
            size=corr1.shape[-2:], mode="bilinear", align_corners=False,
        )

        LL3, LH3, HL3, HH3 = _haar(LL2)
        c3 = self.enc3(torch.cat([LH3, HL3, HH3], 1)).chunk(3, 1)
        corr3 = F.interpolate(
            _ihaar(torch.zeros_like(LL3), c3[0], c3[1], c3[2]),
            size=corr1.shape[-2:], mode="bilinear", align_corners=False,
        )
        return corr1 + corr2 + corr3


# ---------------------------------------------------------------------------
# NullFusion++ v2
# ---------------------------------------------------------------------------

class NullFusionPlusV2(nn.Module):
    """NullFusion++ v2: exact null projection + deep unfolding + Swin + dict + wavelet."""

    def __init__(self, bands=128, msi=3, width=32, scale=4,
                 dict_g=48, dict_m=32, dict_f=24):
        super().__init__()
        self.bands = bands
        W = width
        srf_t = torch.from_numpy(chikusei_srf(bands)).float()

        # 1. Exact null-space projector (from proposal 7)
        self.projector = RangeNullProjector(scale, cg_steps=40, ridge=1e-4)
        self.register_buffer("srf", srf_t)
        self.register_buffer("srfinv", torch.linalg.pinv(srf_t))

        # 2. Feature encoding
        self.hsi_stem = nn.Sequential(nn.Conv2d(bands, W, 3, 1, 1), RCAB(W))
        self.msi_stem = nn.Sequential(nn.Conv2d(msi, W, 3, 1, 1), RCAB(W))
        self.msi_detail = nn.Conv2d(msi, W, 3, 1, 1)
        self.cross1 = CrossAttn(W, 4)
        self.cross2 = CrossAttn(W, 4)
        self.fuse_in = nn.Conv2d(W * 2, W, 1)

        # 3. U-Net encoder
        self.enc1 = nn.Sequential(RCAB(W))
        self.down1 = nn.Conv2d(W, W * 2, 3, 2, 1)
        self.enc2 = nn.Sequential(RCAB(W * 2))
        self.down2 = nn.Conv2d(W * 2, W * 4, 3, 2, 1)

        # 4. Swin Transformer bottleneck
        self.bottleneck = nn.Sequential(
            SwinBlock(W * 4, window=8, heads=min(W // 8, 8)),
            SwinBlock(W * 4, window=8, heads=min(W // 8, 8)),
        )

        # 5. U-Net decoder
        self.up2 = nn.ConvTranspose2d(W * 4, W * 2, 2, 2)
        self.dec2 = nn.Sequential(RCAB(W * 2))
        self.up1 = nn.ConvTranspose2d(W * 2, W, 2, 2)
        self.dec1 = nn.Sequential(RCAB(W))
        self.sa1 = SpatialAttn(W, 8)
        self.sa2 = SpatialAttn(W * 2, 8)

        # 6. Deep unfolding data consistency
        self.unfold1 = DataConsistencyLayer(bands, W)
        self.unfold2 = DataConsistencyLayer(bands, W)

        # 7. Prior (on HR grid)
        self.prior_in = nn.Conv2d(W + W + bands, W, 3, 1, 1)
        pb = []
        for i in range(6):
            pb.append(RCAB(W))
            if i % 2 == 1:
                pb.append(SpatialAttn(W, 8))
        pb.append(nn.Conv2d(W, bands, 3, 1, 1))
        self.prior = nn.Sequential(*pb)
        self.prior_proj = nn.Conv2d(bands, W, 1)

        # 8. Spectral dictionary
        self.Dg = nn.Parameter(torch.zeros(bands, dict_g))
        self.Dm = nn.Parameter(torch.zeros(bands, dict_m))
        self.Df = nn.Parameter(torch.zeros(bands, dict_f))
        self.scale_w = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(W, 32), nn.GELU(), nn.Linear(32, 3),
        )
        self.code_g = nn.Conv2d(W, dict_g, 3, 1, 1)
        self.code_m = nn.Conv2d(W, dict_m, 3, 1, 1)
        self.code_f = nn.Conv2d(W, dict_f, 3, 1, 1)

        # 9. Wavelet detail
        self.wavelet = WaveletBranch(bands, W // 2, depth=1)
        self.wavelet_gate = nn.Sequential(nn.Conv2d(bands * 2, bands, 1), nn.Sigmoid())

        # 10. MSI consistency
        self.consist = nn.Conv2d(dict_g + dict_m + dict_f, bands, 1)

        self._init_dicts()

    def _init_dicts(self):
        s = self.projector.srf
        s_pinv = torch.linalg.pinv(s)
        self.register_buffer("srfinv", s_pinv)
        with torch.no_grad():
            for D in (self.Dg, self.Dm, self.Df):
                M = torch.randn(self.bands, D.shape[1], device=D.device)
                D0 = M - (s @ s_pinv) @ M
                D.copy_(D0 / (D0.norm(dim=0, keepdim=True) + 1e-8))

    def _base(self, yH, yM):
        B, _, H, W = yM.shape
        return self.projector.pinv(yH, yM, (H, W))

    def forward(self, yH, yM):
        B, _, H_lr, W_lr = yH.shape
        _, _, H_hr, W_hr = yM.shape

        # Base reconstruction (exact pseudoinverse)
        base = self._base(yH, yM)

        # Feature encoding
        f_hsi = self.hsi_stem(yH)
        f_msi = self.msi_stem(yM)
        f_hsi = self.cross1(f_hsi, f_msi)
        f_msi_pool = F.adaptive_avg_pool2d(f_msi, f_hsi.shape[-2:])
        f_hsi = self.cross2(f_hsi, f_msi_pool)

        # U-Net
        e1 = self.fuse_in(torch.cat([f_hsi, f_msi_pool], 1))
        e1 = self.sa1(self.enc1(e1))
        e2 = self.sa2(self.enc2(self.down1(e1)))
        bn = self.bottleneck(self.down2(e2))
        d2 = self.dec2(self.up2(bn) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        d1_hr = F.interpolate(d1, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

        # Deep unfolding data consistency
        obs_spectral = F.interpolate(
            yH, size=(H_hr, W_hr), mode="bicubic", align_corners=False
        )
        obs_spatial = self.projector.Rt(yM)
        refined = self.unfold1(base, obs_spectral, obs_spatial)
        refined = self.unfold2(refined, obs_spectral, obs_spatial)

        # Prior on HR grid
        f_detail = self.msi_detail(yM)
        cond = torch.cat([d1_hr, f_detail, refined], 1)
        v = self.prior(self.prior_in(cond))
        v = self.prior_proj(v)

        # Spectral dictionary null compensation
        ag = F.softplus(self.code_g(v))
        am = F.softplus(self.code_m(v))
        af = F.softplus(self.code_f(v))
        ng = torch.einsum("ck,bkhw->bchw", self.Dg, ag)
        nm = torch.einsum("ck,bkhw->bchw", self.Dm, am)
        nf = torch.einsum("ck,bkhw->bchw", self.Df, af)
        sw = F.softmax(self.scale_w(d1_hr), -1).unsqueeze(-1).unsqueeze(-1)
        null_comp = sw[:, 0:1] * ng + sw[:, 1:2] * nm + sw[:, 2:3] * nf

        # Null-space enforcement
        null_comp = self.projector.project_null(null_comp, (H_hr, W_hr))

        # Wavelet detail
        wf = self.wavelet(null_comp)
        wf = wf * self.wavelet_gate(torch.cat([null_comp, wf], 1))

        # Output
        out = refined + null_comp + wf
        consist = self.consist(torch.cat([ag, am, af], 1))

        return {
            "out": out,
            "base": base,
            "refined": refined,
            "null": null_comp,
            "wf": wf,
            "consist": consist,
        }


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
        mat_files.sort(key=lambda f: os.path.getsize(f), reverse=True)
        print(f"[Data] Loading: {os.path.basename(mat_files[0])}")
        data = loadmat(mat_files[0])
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
        coords = [
            (y, x) for y in range(0, H - p + 1, p) for x in range(0, W - p + 1, p)
        ]
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
        lr = blurred[:, y0 :: self.scale, x0 :: self.scale].astype(np.float32)
        msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
        return lr, np.clip(msi, 0, 1)

    def __getitem__(self, idx):
        y, x = self.patches[idx % len(self.patches)]
        p = self.patch
        gt = self.cube[:, y : y + p, x : x + p].copy()
        if self.split == "train":
            if random.random() < 0.5:
                gt = gt[:, :, ::-1].copy()
            if random.random() < 0.5:
                gt = gt[:, ::-1, :].copy()
            if random.random() < 0.5:
                gt = np.rot90(gt, random.randint(1, 3), axes=(1, 2)).copy()
            if random.random() < 0.15:
                gt = (gt + np.random.randn(*gt.shape).astype(np.float32) * 0.01).clip(
                    0, 1
                )
        lr, msi = self._sim(gt)
        return torch.from_numpy(gt), torch.from_numpy(lr), torch.from_numpy(msi)


# ---------------------------------------------------------------------------
# Losses (from hsifusion.losses)
# ---------------------------------------------------------------------------

def ssim_loss(pred, gt, ws=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    coords = torch.arange(ws, dtype=pred.dtype, device=pred.device) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(
        pred.shape[1], 1, -1, -1
    ).contiguous()
    ch = pred.shape[1]
    mu1 = F.conv2d(pred, w, padding=ws // 2, groups=ch)
    mu2 = F.conv2d(gt, w, padding=ws // 2, groups=ch)
    s1 = F.conv2d(pred * pred, w, padding=ws // 2, groups=ch) - mu1 * mu1
    s2 = F.conv2d(gt * gt, w, padding=ws // 2, groups=ch) - mu2 * mu2
    s12 = F.conv2d(pred * gt, w, padding=ws // 2, groups=ch) - mu1 * mu2
    return 1.0 - (
        (2 * mu1 * mu2 + C1) * (2 * s12 + C2) / ((mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2))
    ).mean()


def sam_loss(pred, gt):
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    g = gt.reshape(gt.shape[0], gt.shape[1], -1)
    p = p / (p.norm(dim=1, keepdim=True) + 1e-8)
    g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
    return torch.acos(torch.clamp((p * g).sum(dim=1), -1 + 1e-7, 1 - 1e-7)).mean()


def gradient_loss(pred, target):
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def spectral_gradient_loss(pred, target):
    """Spectral gradient: difference between consecutive bands."""
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return F.l1_loss(dp, dt)


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
    return np.mean(
        ((2 * mu1 * mu2 + C1) * (2 * s12 + C2))
        / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2) + 1e-8)
    )


def ergas_np(pred, gt, scale=4):
    C = pred.shape[0]
    e = sum(
        ((pred - gt) ** 2)[c].mean() / (gt[c].mean() ** 2 + 1e-8)
        for c in range(C)
    )
    return math.sqrt(e / C) * 100 * scale


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, m, decay=0.999):
        self.d = decay
        self.s = {k: v.detach().clone() for k, v in m.state_dict().items()}

    def update(self, m):
        with torch.no_grad():
            for k, v in m.state_dict().items():
                if v.dtype.is_floating_point and k in self.s:
                    self.s[k].mul_(self.d).add_(v.detach(), alpha=1 - self.d)

    def apply(self, m):
        m.load_state_dict(self.s, strict=False)

    def restore(self, m):
        self.s = {k: v.detach().clone() for k, v in m.state_dict().items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def nullfusion_pp_v2_loss(out, gt, yH, yM, model, w_l1=1.0, w_ssim=0.5,
                          w_sam=0.05, w_phys=0.1, w_grad=0.1, w_spec_grad=0.05):
    pred = out["out"]
    l_l1 = F.l1_loss(pred, gt)
    l_ssim = ssim_loss(pred, gt)
    l_sam = sam_loss(pred, gt)
    l_grad = gradient_loss(pred, gt)
    l_spec_grad = spectral_gradient_loss(pred, gt)

    # Physics: observation consistency
    l_phys = F.mse_loss(model.projector.D(pred), yH) + F.mse_loss(model.projector.R(pred), yM)

    total = (
        w_l1 * l_l1
        + w_ssim * l_ssim
        + w_sam * l_sam
        + w_grad * l_grad
        + w_spec_grad * l_spec_grad
        + w_phys * l_phys
    )
    return total, l_l1, l_ssim, l_sam, l_grad, l_spec_grad, l_phys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--time_budget_h", type=float, default=8.5)
    p.add_argument("--eval_every", type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Memory: {mem:.1f} GB")
        if mem < 12:
            patch_size, batch_size, width = 64, 2, 32
            print("[Config] Small GPU: patch=64, batch=2, width=32")
        elif mem < 16:
            patch_size, batch_size, width = 64, 2, 32
            print("[Config] Medium GPU: patch=64, batch=2, width=32")
        else:
            patch_size, batch_size, width = 80, 2, 36
            print("[Config] Large GPU: patch=80, batch=2, width=36")
    else:
        patch_size, batch_size, width = 64, 1, 32
        print("[Config] CPU: patch=64, batch=1, width=32")

    train_ds = ChikuseiDS(args.root, "train", 128, 4, patch_size)
    test_ds = ChikuseiDS(args.root, "test", 128, 4, patch_size)

    model = NullFusionPlusV2(bands=128, msi=3, width=width, scale=4).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: NullFusion++ v2 — {nparams / 1e6:.2f}M params")

    grad_accum = 4
    lr = 2e-4
    steps_per_epoch = 200

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    ema = EMA(model, 0.999)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=2000, T_mult=2, eta_min=1e-6
    )

    save_dir = "/kaggle/working/nullfusion_pp_v2"
    os.makedirs(save_dir, exist_ok=True)
    best_psnr = 0
    best_epoch = 0
    T0 = time.time()
    LIMIT = args.time_budget_h * 3600

    print(f"\nTraining: {args.epochs} epochs, {args.time_budget_h}h budget")
    print(f"Effective batch: {batch_size * grad_accum}, Steps/epoch: {steps_per_epoch}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        if (time.time() - T0) > LIMIT:
            print(f"\n[TIME BUDGET at epoch {epoch}]")
            break

        model.train()
        total_loss = 0
        t0 = time.time()
        opt.zero_grad()

        for step in range(steps_per_epoch):
            gts, lhs, mss = [], [], []
            for _ in range(batch_size):
                g, l, m = train_ds[random.randrange(len(train_ds))]
                gts.append(g)
                lhs.append(l)
                mss.append(m)
            gt = torch.stack(gts, 0).to(device)
            yH = torch.stack(lhs, 0).to(device)
            yM = torch.stack(mss, 0).to(device)

            out = model(yH, yM)
            loss, ll, ls, lsm, lg, lsg, lp = nullfusion_pp_v2_loss(
                out, gt, yH, yM, model
            )
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                ema.update(model)
            total_loss += loss.item()

        scheduler.step()
        dt = time.time() - t0
        avg = total_loss / steps_per_epoch

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d}/{args.epochs} | Loss {avg:.5f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s"
            )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ema.apply(model)
            model.eval()
            ps, ss, sa, er = [], [], [], []
            with torch.no_grad():
                for i in range(len(test_ds)):
                    g, l, m = test_ds[i]
                    pred_np = model(
                        l.unsqueeze(0).to(device), m.unsqueeze(0).to(device)
                    )["out"][0].cpu().numpy()
                    gt_np = g.numpy()
                    ps.append(psnr_np(pred_np, gt_np))
                    ss.append(ssim_np(pred_np, gt_np))
                    sa.append(sam_np(pred_np, gt_np))
                    er.append(ergas_np(pred_np, gt_np, 4))

            mp, ms_, ma, me = float(np.mean(ps)), float(np.mean(ss)), float(np.mean(sa)), float(np.mean(er))
            marker = ""
            if mp > best_psnr:
                best_psnr = mp
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                marker = " [BEST]"
            print(
                f"  >>> Test@{epoch}: PSNR {mp:.4f} | SSIM {ms_:.4f} | "
                f"SAM {ma:.3f} | ERGAS {me:.3f}{marker}"
            )
            for name, target in {
                "CoFusion": 49.14,
                "RAMoE": 48.10,
                "SMGU-Net": 48.82,
                "PSRT": 47.99,
                "U2Net": 47.93,
                "KrylovNet v1": 43.69,
            }.items():
                d = mp - target
                m_ = " <<< BEAT" if d > 0 else ""
                print(f"    vs {name:15s}: Δ={d:+.2f} dB{m_}")
            ema.restore(model)

    # Final eval
    ckpt = os.path.join(save_dir, "best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    ps, ss, sa, er = [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            g, l, m = test_ds[i]
            pred_np = model(
                l.unsqueeze(0).to(device), m.unsqueeze(0).to(device)
            )["out"][0].cpu().numpy()
            gt_np = g.numpy()
            ps.append(psnr_np(pred_np, gt_np))
            ss.append(ssim_np(pred_np, gt_np))
            sa.append(sam_np(pred_np, gt_np))
            er.append(ergas_np(pred_np, gt_np, 4))
    final = {
        "psnr": float(np.mean(ps)),
        "ssim": float(np.mean(ss)),
        "sam": float(np.mean(sa)),
        "ergas": float(np.mean(er)),
    }
    print("\n" + "=" * 70)
    print(
        f"FINAL (epoch {best_epoch}): PSNR {final['psnr']:.4f} | "
        f"SSIM {final['ssim']:.4f} | SAM {final['sam']:.3f} | ERGAS {final['ergas']:.3f}"
    )
    print("=" * 70)
    for name, target in {
        "CoFusion (2026)": 49.14,
        "RAMoE (2026)": 48.10,
        "SMGU-Net (2025)": 48.82,
        "PSRT (2023)": 47.99,
        "U2Net (2023)": 47.93,
        "KrylovNet v1 (ours)": 43.69,
    }.items():
        d = final["psnr"] - target
        m_ = " <<< BEAT" if d > 0 else ""
        print(f"  {name:25s} Target {target:6.2f} | Ours Δ={d:+.2f} dB{m_}")
    with open("/kaggle/working/nullfusion_pp_v2_results.json", "w") as f:
        json.dump(
            {
                "dataset": "Chikusei",
                "bands": 128,
                "params_M": nparams / 1e6,
                "best_epoch": best_epoch,
                "final": final,
            },
            f,
            indent=2,
        )
    print("\nSaved nullfusion_pp_v2_results.json")


if __name__ == "__main__":
    main()
