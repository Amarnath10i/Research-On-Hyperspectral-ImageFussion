"""Network building blocks.

Each block below implements the mechanism the design document claims, and the
claim is checked numerically in `selfcheck.py` rather than asserted in prose:

  P4ConvZ2 / P4ConvP4          -> genuine p4 equivariance   (checked to ~1e-6)
  HaarDWT                      -> exact orthonormal inverse (checked to ~1e-7)
  TensorSpectralSpatialEncoder -> the Tucker core carries gradient (checked)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- EFE
class P4ConvZ2(nn.Module):
    """Lifting convolution Z2 -> p4. The output carries an explicit orientation
    axis of size 4, produced by convolving with the four rotated copies of one
    shared kernel."""

    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3, bias: bool = True):
        super().__init__()
        self.out_ch, self.ksize = out_ch, ksize
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, ksize, ksize))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,Cin,H,W]->[B,Cout,4,H,W]
        w = torch.cat([torch.rot90(self.weight, r, dims=(2, 3)) for r in range(4)], dim=0)
        b = None if self.bias is None else self.bias.repeat(4)
        y = F.conv2d(x, w, b, padding=self.ksize // 2)
        bsz, _, h, wd = y.shape
        return y.view(bsz, 4, self.out_ch, h, wd).transpose(1, 2)


class P4ConvP4(nn.Module):
    """Group convolution p4 -> p4.

    For output orientation r the filter is rotated in space *and* cyclically
    shifted along the orientation axis; doing only one of the two is the usual
    way a 'rotation-equivariant' layer silently fails to be equivariant.
    """

    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3, bias: bool = True):
        super().__init__()
        self.in_ch, self.out_ch, self.ksize = in_ch, out_ch, ksize
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, 4, ksize, ksize))
        nn.init.kaiming_normal_(self.weight.view(out_ch, -1, ksize, ksize),
                                mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,Cin,4,H,W]->[B,Cout,4,H,W]
        bsz, cin, _, h, wd = x.shape
        xf = x.reshape(bsz, cin * 4, h, wd)
        ws = []
        for r in range(4):
            wr = torch.rot90(self.weight, r, dims=(3, 4))   # rotate the spatial support
            wr = torch.roll(wr, shifts=r, dims=2)           # act on the orientation axis
            ws.append(wr.reshape(self.out_ch, cin * 4, self.ksize, self.ksize))
        w = torch.cat(ws, dim=0)
        b = None if self.bias is None else self.bias.repeat(4)
        y = F.conv2d(xf, w, b, padding=self.ksize // 2)
        return y.view(bsz, 4, self.out_ch, h, wd).transpose(1, 2)


class EquivariantFeatureExtractor(nn.Module):
    """EFE. BatchNorm3d shares statistics across orientations, so normalisation
    does not break equivariance; the closing max over the orientation axis makes
    the output a plain feature map that rotates with the input."""

    def __init__(self, in_ch: int, width: int, out_ch: int, depth: int = 2):
        super().__init__()
        self.lift = P4ConvZ2(in_ch, width)
        self.bn0 = nn.BatchNorm3d(width)
        self.blocks = nn.ModuleList([
            nn.ModuleList([P4ConvP4(width, width), nn.BatchNorm3d(width)])
            for _ in range(depth)
        ])
        self.proj = nn.Conv2d(width, out_ch, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn0(self.lift(x)))
        for conv, bn in self.blocks:
            h = self.act(bn(conv(h))) + h
        h = h.max(dim=2).values          # group pooling over the four orientations
        return self.proj(h)


class PlainFeatureExtractor(nn.Module):
    """Non-equivariant control arm for the EFE ablation: matched depth and
    parameter budget, ordinary convolutions."""

    def __init__(self, in_ch: int, width: int, out_ch: int, depth: int = 2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width * 4, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(depth):
            layers += [nn.Conv2d(width * 4, width * 4, 3, 1, 1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width * 4, out_ch, 1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# --------------------------------------------------------------- degradation code
class DegradationEncoder(nn.Module):
    """Estimates a degradation code from the observed (LR-HSI, MSI) pair.

    An auxiliary head regresses the true degradation parameters, which are known
    during training because we synthesise them. Without that supervision the
    code tends to collapse to a constant and the conditioning does nothing.
    """

    def __init__(self, hsi_ch: int, msi_ch: int, code: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(hsi_ch + msi_ch, 64, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 96, 3, 2, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(96, 128, 3, 1, 1), nn.LeakyReLU(0.1, True),
        )
        self.head = nn.Sequential(nn.Linear(256, code), nn.LeakyReLU(0.1, True),
                                  nn.Linear(code, code))
        self.deg_head = nn.Linear(code, 5)   # sx, sy, sin2t, cos2t, noise

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        msi_lr = F.adaptive_avg_pool2d(msi, lr_hsi.shape[-2:])
        f = self.body(torch.cat([lr_hsi, msi_lr], dim=1))
        stats = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        code = self.head(stats)
        return code, self.deg_head(code)


class FiLM(nn.Module):
    """Feature-wise linear modulation. Zero-initialised so the conditioned model
    starts exactly at the unconditioned one."""

    def __init__(self, code: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(code, channels * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(code).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


# -------------------------------------------------------------------------- TSSE
class TensorSpectralSpatialEncoder(nn.Module):
    """z[b,k,h,w] = sum_{i,j} G[i,j,k] * a[b,i,h,w] * b[b,j,h,w]

    A Tucker-style contraction of the outer product of the two projected feature
    maps against a learned core tensor G, realised as a 1x1 convolution whose
    weights *are* G - so the core genuinely participates and receives gradient.
    """

    def __init__(self, hsi_ch: int, msi_ch: int, out_ch: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        self.proj_hsi = nn.Conv2d(hsi_ch, rank, 1)
        self.proj_msi = nn.Conv2d(msi_ch, rank, 1)
        self.core = nn.Parameter(torch.randn(rank, rank, rank) * (rank ** -0.75))
        self.out = nn.Sequential(nn.Conv2d(rank, out_ch, 1), nn.LeakyReLU(0.1, True),
                                 nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip = nn.Conv2d(hsi_ch + msi_ch, out_ch, 1)
        self.norm = nn.GroupNorm(8, out_ch)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pa = self.proj_hsi(a)                        # [B,R1,H,W]
        pb = self.proj_msi(b)                        # [B,R2,H,W]
        outer = (pa.unsqueeze(2) * pb.unsqueeze(1)).flatten(1, 2)   # [B,R1*R2,H,W]
        core = self.core.permute(2, 0, 1).reshape(self.rank, self.rank * self.rank, 1, 1)
        z = F.conv2d(outer, core.to(outer.dtype))    # [B,R3,H,W]
        return self.norm(self.out(z) + self.skip(torch.cat([a, b], dim=1)))

    def rank_penalty(self) -> torch.Tensor:
        """Nuclear norm of the mode-3 unfolding: a convex surrogate for rank."""
        return torch.linalg.svdvals(self.core.reshape(self.rank, -1).float()).sum()


# ------------------------------------------------------------------------- AF-MoE
class RegionAwareMoE(nn.Module):
    """Per-pixel top-k expert routing.

    A globally pooled gate must commit to one fusion strategy for a whole image.
    Routing per pixel lets shadowed, textured and flat regions of the same scene
    take different experts, which is the property the region-aware MoE
    literature reports as the win.
    """

    def __init__(self, channels: int, experts: int = 4, topk: int = 2):
        super().__init__()
        self.experts_n, self.topk = experts, min(topk, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, True),
                nn.Conv2d(channels, channels, 3, 1, 1),
            ) for _ in range(experts)
        ])
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels // 2, experts, 1),
        )
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)                              # [B,E,H,W]
        if self.topk < self.experts_n:
            thresh = logits.topk(self.topk, dim=1).values[:, -1:, :, :]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        g = logits.softmax(dim=1)
        self.last_gate = g
        out = x
        for i, expert in enumerate(self.experts):
            out = out + g[:, i: i + 1] * expert(x)
        return out

    def balance_loss(self) -> torch.Tensor:
        """Squared coefficient of variation of expert usage; 0 when uniform.
        Without it, top-k routing collapses onto a single expert."""
        if self.last_gate is None:
            return torch.zeros((), device=next(self.parameters()).device)
        imp = self.last_gate.mean(dim=(0, 2, 3))
        return self.experts_n * (imp ** 2).sum() - 1.0

    @torch.no_grad()
    def usage(self) -> Optional[torch.Tensor]:
        """Per-expert mean gate weight - used for the interpretability figure."""
        return None if self.last_gate is None else self.last_gate.mean(dim=(0, 2, 3))


# --------------------------------------------------------------------------- FDRM
class HaarDWT(nn.Module):
    """Orthonormal Haar transform as a fixed grouped convolution, exactly
    inverted by the transposed convolution with the same filters."""

    def __init__(self):
        super().__init__()
        h = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        g1 = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        g2 = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        g3 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.register_buffer("filt", torch.stack([h, g1, g2, g3]).unsqueeze(1) * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,C,H,W]->[B,4C,H/2,W/2]
        b, c, h, w = x.shape
        y = F.conv2d(x.reshape(b * c, 1, h, w), self.filt.to(x.dtype), stride=2)
        return y.reshape(b, c * 4, h // 2, w // 2)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:      # [B,4C,H,W]->[B,C,2H,2W]
        b, c4, h, w = y.shape
        c = c4 // 4
        x = F.conv_transpose2d(y.reshape(b * c, 4, h, w), self.filt.to(y.dtype), stride=2)
        return x.reshape(b, c, h * 2, w * 2) * 0.25


class FrequencyDomainRefinement(nn.Module):
    """Wavelet-domain refinement: per-subband processing with learnable
    soft-thresholding (classical wavelet shrinkage, made learnable), plus
    cross-subband mixing, then an exact inverse transform.

    The v1 module was three convolutions of different kernel size and touched no
    frequency representation at all.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dwt = HaarDWT()
        self.sub = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, True),
                          nn.Conv2d(channels, channels, 3, 1, 1))
            for _ in range(4)
        ])
        self.thresh = nn.Parameter(torch.zeros(3, channels))   # detail subbands
        self.mix = nn.Conv2d(channels * 4, channels * 4, 1)
        self.fuse = nn.Conv2d(channels, channels, 3, 1, 1)

    @staticmethod
    def _shrink(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = F.softplus(t)[None, :, None, None].to(x.dtype)
        return torch.sign(x) * torch.clamp(x.abs() - t, min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        bands = self.dwt(x).chunk(4, dim=1)
        out = []
        for i, band in enumerate(bands):
            y = self.sub[i](band)
            if i > 0:                                  # shrink detail subbands only
                y = self._shrink(y, self.thresh[i - 1])
            out.append(y + band)
        y = self.dwt.inverse(self.mix(torch.cat(out, dim=1)))
        if pad_h or pad_w:
            y = y[..., :h, :w]
        return self.fuse(y) + x[..., :h, :w]


# ---------------------------------------------------------------------- upsamplers
class BackProjectionUpsampler(nn.Module):
    """Learned upsampling with iterative observation-model correction:

        y <- Up(x);   then repeatedly   y <- y + Up_res( x - Down(y) )

    The residual is measured in LR space, where the actual observation is
    available, so each step pulls the estimate back onto the observation
    manifold. Bicubic interpolation cannot do this because it never looks at
    how well its own output re-explains the input.
    """

    def __init__(self, bands: int, scale: int, width: int = 64, iters: int = 2):
        super().__init__()
        self.scale, self.iters = scale, iters
        stages, s, ch = [], scale, bands
        while s > 1:
            step = 2 if s % 2 == 0 else s
            stages += [nn.Conv2d(ch, width * step * step, 3, 1, 1),
                       nn.PixelShuffle(step), nn.LeakyReLU(0.1, True)]
            ch = width
            s //= step
        stages += [nn.Conv2d(ch, bands, 3, 1, 1)]
        self.up = nn.Sequential(*stages)
        self.down = nn.Sequential(
            nn.Conv2d(bands, width, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(width, width, 2 * scale + 1, scale, scale), nn.LeakyReLU(0.1, True),
            nn.Conv2d(width, bands, 3, 1, 1),
        )
        self.up_res = nn.Sequential(
            nn.Conv2d(bands, width, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(width, bands * scale * scale, 3, 1, 1), nn.PixelShuffle(scale),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(lr, scale_factor=self.scale, mode="bicubic",
                             align_corners=False)
        y = self.up(lr) + base                     # learned residual over a cheap prior
        for _ in range(self.iters):
            err = lr - self.down(y)
            if err.shape[-2:] != lr.shape[-2:]:
                err = F.interpolate(err, size=lr.shape[-2:], mode="bilinear",
                                    align_corners=False)
            y = y + self.up_res(err)
        return y


class BicubicUpsampler(nn.Module):
    """Control arm for the back-projection ablation."""

    def __init__(self, bands: int, scale: int, width: int = 64):
        super().__init__()
        self.scale = scale
        self.refine = nn.Sequential(
            nn.Conv2d(bands, width, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(width, bands, 3, 1, 1),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(lr, scale_factor=self.scale, mode="bicubic",
                          align_corners=False)
        return y + self.refine(y)
