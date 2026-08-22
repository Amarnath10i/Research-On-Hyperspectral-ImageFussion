r"""NullFusion v2 — Null-Space Conditional Fusion Network (proposal7).

UPGRADES over v1:
  * Cross-modal attention (HSI <-> MSI) in conditioning encoder.
  * Per-pixel spectral mixing inside f_theta (models band correlations cheaply).
  * Deeper prior (8 blocks) + deeper encoder (4 blocks).
  * Windowed spatial self-attention (O(N*ws^2), safe at any resolution).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from proposal1.ambiguity.operator import CombinedOperator


@dataclass
class NullFusionConfig:
    scale: int = 4
    bands: int = 31
    msi_bands: int = 3
    ksize: int = 9
    sigma: float = 1.2
    cg_steps: int = 80
    ridge: float = 1e-6
    width: int = 96
    enc_depth: int = 4
    prior_depth: int = 8
    use_attn: bool = True
    cross_attn_heads: int = 4
    rank: int = 31
    base: str = "pinv"
    clamp: bool = False


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class _SpatialSelfAttn(nn.Module):
    """Windowed spatial self-attention — O(N * ws^2) memory, safe at any resolution."""

    def __init__(self, ch: int, window: int = 16):
        super().__init__()
        self.window = window
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = max(ch, 1) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_pad = x
        _, _, Hp, Wp = x_pad.shape
        q, k, v = self.qkv(x_pad).chunk(3, dim=1)
        nH, nW = Hp // ws, Wp // ws
        q = q.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        k = k.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        v = v.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(B, nH, nW, ws, ws, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return x + self.proj(out)


class _SpectralMix(nn.Module):
    """Per-pixel channel mixing to model band correlations. O(N*C) memory."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.fc1 = nn.Linear(ch, ch * 2)
        self.fc2 = nn.Linear(ch * 2, ch)

    def forward(self, x):
        B, C, H, W = x.shape
        xt = x.permute(0, 2, 3, 1).reshape(-1, C)
        xt = self.norm(xt)
        xt = F.gelu(self.fc1(xt))
        xt = self.fc2(xt)
        return x + xt.reshape(B, H, W, C).permute(0, 3, 1, 2)


class _CrossAttn(nn.Module):
    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_ch = ch // n_heads
        self.q = nn.Conv2d(ch, ch, 1)
        self.kv = nn.Conv2d(ch, ch * 2, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = max(self.head_ch, 1) ** -0.5

    def forward(self, query, context):
        B, C, Hq, Wq = query.shape
        Hk, Wk = context.shape[2], context.shape[3]
        if (Hk, Wk) != (Hq, Wq):
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


class NullFusionNet(nn.Module):
    def __init__(self, cfg: NullFusionConfig):
        super().__init__()
        self.cfg = cfg
        self.op = CombinedOperator(cfg.scale, cfg.bands, cfg.msi_bands,
                                   cfg.ksize, cfg.sigma, srf=None,
                                   cg_steps=cfg.cg_steps, ridge=cfg.ridge)
        W = cfg.width
        self.cross_attn = _CrossAttn(W, cfg.cross_attn_heads)
        self.hsi_stem = nn.Conv2d(cfg.bands, W, 3, 1, 1)
        self.msi_stem = nn.Conv2d(cfg.msi_bands, W, 3, 1, 1)
        self.msi_detail = nn.Conv2d(cfg.msi_bands, W, 3, 1, 1)
        self.fuse = nn.Conv2d(2 * W, W, 1)
        self.enc = nn.Sequential(*[_ResBlock(W) for _ in range(cfg.enc_depth)])
        self.up = nn.ConvTranspose2d(W, W, cfg.scale, cfg.scale, 0, bias=False)
        in_ch = W + W + cfg.bands
        self.prior_in = nn.Conv2d(in_ch, W, 3, 1, 1)
        blocks = []
        for i in range(cfg.prior_depth):
            blocks.append(_ResBlock(W))
            if cfg.use_attn and (i % 2 == 1):
                blocks.append(_SpatialSelfAttn(W))
        if cfg.use_attn:
            blocks.append(_SpectralMix(W))
        blocks.append(nn.Conv2d(W, cfg.bands, 3, 1, 1))
        self.prior_body = nn.Sequential(*blocks)
        self.rank = min(cfg.rank, cfg.bands)
        if self.rank < cfg.bands:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(cfg.bands, self.rank, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.rank, cfg.bands, 1))
        else:
            self.bottleneck = nn.Identity()

    def set_srf(self, srf: torch.Tensor) -> None:
        s = srf if srf.shape[0] == self.cfg.bands else srf.t().contiguous()
        self.op.srf.data = s.float().to(self.op.srf.device)

    @staticmethod
    def _make_base(yH, yM, op, out_hw, mode):
        if mode == "pinv":
            return op.pinv(yH, yM, out_hw)
        up = F.interpolate(yH, size=out_hw, mode="bicubic", align_corners=False)
        return up

    def forward(self, yH: torch.Tensor, yM: torch.Tensor) -> dict:
        B, _, H, W = yH.shape
        h, w = yH.shape[-2], yH.shape[-1]
        out_hw = (yM.shape[-2], yM.shape[-1])
        Xobs = self._make_base(yH, yM, self.op, out_hw, self.cfg.base)
        f_hsi = self.hsi_stem(yH)
        msi_lr = F.adaptive_avg_pool2d(self.msi_stem(yM), (h, w))
        f_hsi = self.cross_attn(f_hsi, msi_lr)
        z = self.enc(self.fuse(torch.cat([f_hsi, msi_lr], dim=1)))
        f_hr = self.up(z)
        f_msi = self.msi_detail(yM)
        cond = torch.cat([f_hr, f_msi, Xobs], dim=1)
        v = self.prior_in(cond)
        v = self.prior_body(v)
        v = self.bottleneck(v)
        v_null = self.op.project_null(v)
        out = Xobs + v_null
        if self.cfg.clamp and not self.training:
            out = out.clamp(0, 1)
        return {"out": out, "base": Xobs, "null": v_null, "prior": v}
