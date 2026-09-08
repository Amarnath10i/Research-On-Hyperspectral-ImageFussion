"""Conditional spectral score network (denoiser).

The network predicts the Gaussian noise eps that was added to a clean HR
hyperspectral patch,

    eps_theta(y_t, M, d, t) ~ eps,      y_t = sqrt(a_t) y_0 + sqrt(1-a_t) eps

conditioned on
    y_t   the noisy HR hyperspectral cube (C bands),
    M     the HR multispectral guide (m bands) - the spatial-detail channel,
    d     a degradation code from the blind operator encoder (tells the model
          which blur/noise/SRF produced the pair),
    t     the diffusion timestep (noise level).

Architecture is a compact U-Net with FiLM conditioning and 1x1 spectral
mixing, sized for full scenes on a single 16 GB GPU.  The MSI is concatenated
at the stem so the spatial detail enters at native resolution.

Training objective is plain denoising score matching (MSE on eps); the
null-space projection that guarantees observation consistency lives in the
sampler, not here.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding of a timestep."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device)
                      / max(half, 1)).to(x.dtype)
    args = x[:, None].to(x.dtype) * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation from a shared conditioning vector.

    The last linear layer is zero-initialised so conditioning starts at the
    identity and only becomes active when gradients prove it useful.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.net = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)      # [B, C]
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.film = FiLM(cond_dim, out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.drop(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SpectralScoreNet(nn.Module):
    def __init__(self, bands: int, msi_bands: int, ch: int = 64,
                 ch_mult: tuple = (1, 2, 2, 2), num_res: int = 2,
                 cond_dim: int = 256, code_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.ch, self.cond_dim = ch, cond_dim
        self.stem = nn.Conv2d(bands + msi_bands, ch, 3, 1, 1)
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(True),
            nn.Linear(cond_dim, cond_dim))
        self.code_mlp = nn.Sequential(
            nn.Linear(code_dim, cond_dim), nn.SiLU(True),
            nn.Linear(cond_dim, cond_dim))

        levels = [ch * m for m in ch_mult]
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = ch
        for i, lv in enumerate(levels):
            blk = nn.ModuleList()
            for _ in range(num_res):
                blk.append(ResBlock(prev, lv, cond_dim, dropout))
                prev = lv
            self.enc.append(blk)
            if i < len(levels) - 1:
                self.downs.append(nn.Conv2d(lv, lv, 3, 2, 1))

        self.mid1 = ResBlock(prev, prev, cond_dim, dropout)
        self.mid2 = ResBlock(prev, prev, cond_dim, dropout)

        self.dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(levels))):
            lv = levels[i]
            blk = nn.ModuleList()
            for j in range(num_res + 1):
                skip_ch = lv if j == 0 else lv          # concat with encoder skip
                in_ch = prev + (lv if j == 0 else 0)
                blk.append(ResBlock(in_ch, lv, cond_dim, dropout))
                prev = lv
            self.dec.append(blk)
            if i > 0:
                self.ups.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, bands, 3, 1, 1)

    def _cond(self, t: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """Single shared conditioning vector for all FiLM layers."""
        te = self.t_mlp(sinusoidal_embedding(t, self.cond_dim))
        if code is None:
            return te
        return te + self.code_mlp(code)

    def forward(self, y_t: torch.Tensor, msi: torch.Tensor,
                code: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict the noise eps for y_t.

        Args:
            y_t:  [B, C, H, W] noisy hyperspectral cube
            msi:  [B, m, H, W] HR multispectral guide
            code: [B, code_dim] degradation code (optional)
            t:    [B] integer timestep in [1, T]; defaults to a fixed level
        """
        b = y_t.shape[0]
        if t is None:
            t = torch.full((b,), 100, device=y_t.device, dtype=torch.long)
        cond = self._cond(t.long(), code)

        h = self.stem(torch.cat([y_t, msi], dim=1))
        skips = []
        for i, blk in enumerate(self.enc):
            for layer in blk:
                h = layer(h, cond)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        h = self.mid1(h, cond)
        h = self.mid2(h, cond)

        for i, blk in enumerate(self.dec):
            sk = skips[len(skips) - 1 - i]
            h = torch.cat([h, sk], dim=1)
            for layer in blk:
                h = layer(h, cond)
            if i < len(self.ups):
                h = self.ups[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())