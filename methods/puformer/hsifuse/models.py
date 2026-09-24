"""PUFormer: Physics-Unfolded transFormer for HSI-MSI fusion, plus the SSRNet anchor.

Each of K stages:
    Z_k   = X_{k-1} - eta_H * Dt(D X - Y_H) - eta_M * Rt(R X - Y_M)     exact data-fidelity step
    X_k   = Z_k + P_k([Z_k, Y_M, Dt-residual], memory)                    learned prior (U-Net)
The prior is a 3-level U-Net of Restormer blocks: transposed (channel) attention,
which is attention across spectral feature maps, plus a gated depth-wise FFN.
Features of each stage's decoder are passed to the next stage (cross-stage memory).
The image path stays in float32; only the prior runs under autocast.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import Degradation


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.w[:, None, None] + self.b[:, None, None]


class MDTA(nn.Module):
    """Multi-Dconv-head transposed attention (attention over channels)."""

    def __init__(self, c, heads):
        super().__init__()
        self.h = heads
        self.t = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.out = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        n, c, hh, ww = x.shape
        q, k, v = self.dw(self.qkv(x)).chunk(3, 1)
        q, k, v = (t.reshape(n, self.h, c // self.h, hh * ww) for t in (q, k, v))
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
        a = (q @ k.transpose(-2, -1) * self.t).softmax(-1).to(v.dtype)
        return self.out((a @ v).reshape(n, c, hh, ww))


class GDFN(nn.Module):
    def __init__(self, c, mult=2.66):
        super().__init__()
        h = int(c * mult)
        self.inp = nn.Conv2d(c, h * 2, 1, bias=False)
        self.dw = nn.Conv2d(h * 2, h * 2, 3, padding=1, groups=h * 2, bias=False)
        self.out = nn.Conv2d(h, c, 1, bias=False)

    def forward(self, x):
        a, b = self.dw(self.inp(x)).chunk(2, 1)
        return self.out(F.gelu(a) * b)


class Block(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.n1, self.att, self.n2, self.ffn = LayerNorm2d(c), MDTA(c, heads), LayerNorm2d(c), GDFN(c)

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ffn(self.n2(x))


def stack(c, heads, n):
    return nn.Sequential(*[Block(c, heads) for _ in range(n)])


class Prior(nn.Module):
    """3-level Restormer U-Net: in_ch -> residual on the HSI."""

    def __init__(self, in_ch, out_ch, w=48, depth=(2, 3, 4), heads=(1, 2, 4), mem=True):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, w, 3, padding=1)
        self.mem = nn.Conv2d(w * 2, w, 1) if mem else None
        self.e1, self.d12 = stack(w, heads[0], depth[0]), nn.Conv2d(w, w * 2, 4, 2, 1)
        self.e2, self.d23 = stack(w * 2, heads[1], depth[1]), nn.Conv2d(w * 2, w * 4, 4, 2, 1)
        self.mid = stack(w * 4, heads[2], depth[2])
        self.u32, self.r2 = nn.ConvTranspose2d(w * 4, w * 2, 2, 2), nn.Conv2d(w * 4, w * 2, 1)
        self.dec2 = stack(w * 2, heads[1], depth[1])
        self.u21, self.r1 = nn.ConvTranspose2d(w * 2, w, 2, 2), nn.Conv2d(w * 2, w, 1)
        self.dec1 = stack(w, heads[0], depth[0])
        self.refine = stack(w, heads[0], 2)
        self.tail = nn.Conv2d(w, out_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight), nn.init.zeros_(self.tail.bias)  # start at the physics step

    def forward(self, x, memory=None):
        f = self.embed(x)
        if self.mem is not None and memory is not None:
            f = self.mem(torch.cat([f, memory], 1))
        e1 = self.e1(f)
        e2 = self.e2(self.d12(e1))
        m = self.mid(self.d23(e2))
        d2 = self.dec2(self.r2(torch.cat([self.u32(m), e2], 1)))
        d1 = self.dec1(self.r1(torch.cat([self.u21(d2), e1], 1)))
        d1 = self.refine(d1)
        return self.tail(d1), d1


class PUFormer(nn.Module):
    def __init__(self, deg: Degradation, bands=128, msi=8, stages=3, width=48,
                 depth=(2, 3, 4), heads=(1, 2, 4)):
        super().__init__()
        self.deg, self.K, self.bands = deg, stages, bands
        self.amp = False                      # fp16 autocast for the priors only
        self.eta_h = nn.Parameter(torch.full((stages,), 1.0))
        self.eta_m = nn.Parameter(torch.full((stages,), 1.0))
        in_ch = bands + msi + bands          # Z, Y_M, Dt(D Z - Y_H)
        self.priors = nn.ModuleList(
            Prior(in_ch, bands, width, depth, heads, mem=k > 0) for k in range(stages))

    def fidelity_grad(self, x, yh, ym):
        rh = self.deg.D(x) - yh
        rm = self.deg.R(x) - ym
        return self.deg.Dt(rh), self.deg.Rt(rm)

    def forward(self, yh, ym, return_all=False):
        x = F.interpolate(yh, scale_factor=self.deg.scale, mode="bicubic", align_corners=False)
        mem, outs = None, []
        for k in range(self.K):
            gh, gm = self.fidelity_grad(x, yh, ym)
            # D has gain ~1/16 on smooth content, so the H step is scaled by 16
            z = x - F.softplus(self.eta_h[k]) * 16 * gh - F.softplus(self.eta_m[k]) * gm
            rh, _ = self.fidelity_grad(z, yh, ym)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and x.is_cuda):
                res, mem = self.priors[k](torch.cat([z, ym, rh * 16], 1), mem)
            x = z + res.float()
            outs.append(x)
        return outs if return_all else x


class SSRNet(nn.Module):
    """SSR-NET (Zhang et al., TGRS 2021) - reference anchor, same as baselines/ssrnet."""

    def __init__(self, scale=4, msi=8, bands=128):
        super().__init__()
        self.scale, self.msi, self.bands = scale, msi, bands
        conv = lambda: nn.Sequential(nn.Conv2d(bands, bands, 3, padding=1), nn.ReLU())
        self.conv_fus, self.conv_spat, self.conv_spec = conv(), conv(), conv()

    def forward(self, yh, ym, return_all=False):
        x = F.interpolate(yh, scale_factor=self.scale, mode="bilinear")
        gap = self.bands / (self.msi - 1.0)
        x = x.clone()
        for i in range(self.msi - 1):
            x[:, int(gap * i)] = ym[:, i]
        x[:, self.bands - 1] = ym[:, self.msi - 1]
        x = self.conv_fus(x)
        x = x + self.conv_spat(x)
        x = x + self.conv_spec(x)
        return [x] if return_all else x


def build(name: str, deg: Degradation, **kw):
    if name == "puformer":
        return PUFormer(deg, **kw)
    if name == "ssrnet":
        return SSRNet()
    raise ValueError(name)
