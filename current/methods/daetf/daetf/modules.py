"""Network building blocks — DAETF-Net v3.

Architecture identity: Adaptive Spectral-Causal Routing (ASCR).

Every block is numerically verified in `selfcheck.py`:
  P4ConvZ2 / P4ConvP4          -> genuine p4 equivariance   (checked to ~1e-6)
  HaarDWT                      -> exact orthonormal inverse (checked to ~1e-7)
  TensorSpectralSpatialEncoder -> the Tucker core carries gradient (checked)

New in v3:
  SpectralDisagreementField    -> D(p) = [|Δ|, ∇Δ, ∇²Δ]  cross-modal mismatch
  DegradationConditionedMoE   -> gate(F_H, F_M, d, D) -> semantic experts
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


# ============================================================= NEW v3 MODULES =

class SpectralDisagreementField(nn.Module):
    """Cross-modal disagreement at the feature level.

    After projecting HSI features into MSI space, computes:
        Δ(p)  = F_M(p) − P(F_H)(p)               spectral mismatch
        D(p)  = [|Δ|, ∇Δ, ∇²Δ]                   mismatch + its spatial gradients

    This tells the routing network WHERE the two modalities disagree, and HOW
    rapidly that disagreement varies spatially (edge vs smooth mismatch).

    The output D(p) is a 3-channel map at HSI/MSI feature resolution.  It is
    concatenated to the gate input in DegradationConditionedMoE so the routing
    policy is explicitly informed by the local reliability of each modality.

    Note: this is intentionally lightweight (one 1×1 conv) so it does not add
    significant memory or runtime on Kaggle GPUs.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Project HSI features into MSI space (same channel width C)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.eye_(self.proj.weight.view(channels, channels))  # start as identity

        # Compress disagreement representation to 3 channels
        # [|Δ| + ∇Δ + ∇²Δ] individually, then fuse
        self.compress = nn.Conv2d(channels * 3, 3, 1, bias=True)
        nn.init.zeros_(self.compress.bias)

        # Laplacian kernel (fixed, not learned)
        lap = torch.tensor([[0., 1., 0.],
                             [1., -4., 1.],
                             [0., 1., 0.]], dtype=torch.float32)
        self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3))

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Laplacian channel-wise via grouped conv."""
        b, c, h, w = x.shape
        k = self.lap_kernel.expand(c, 1, 3, 3).to(x.dtype)
        return F.conv2d(x, k, padding=1, groups=c)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel magnitude |∇x| per channel."""
        b, c, h, w = x.shape
        dx = x[..., :, 1:] - x[..., :, :-1]   # [B,C,H,W-1]
        dy = x[..., 1:, :] - x[..., :-1, :]   # [B,C,H-1,W]
        dx = F.pad(dx, (0, 1))                  # pad to [B,C,H,W]
        dy = F.pad(dy, (0, 0, 0, 1))
        return (dx ** 2 + dy ** 2 + 1e-6).sqrt()

    def forward(self, f_hsi: torch.Tensor, f_msi: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_hsi: [B, C, H, W] — HSI features (at HR resolution after upsampling)
            f_msi: [B, C, H, W] — MSI features (same spatial size)
        Returns:
            delta: [B, C, H, W]  raw disagreement (for correction branch)
            D:     [B, 3, H, W]  compressed disagreement field for gating
        """
        f_hsi_proj = self.proj(f_hsi)           # project to same space as f_msi
        delta = f_msi - f_hsi_proj              # [B, C, H, W]

        abs_delta = delta.abs()                         # |Δ|
        grad_delta = self._gradient_magnitude(delta)    # |∇Δ|
        lap_delta = self._laplacian(delta).abs()        # |∇²Δ|

        D_raw = torch.cat([abs_delta, grad_delta, lap_delta], dim=1)  # [B, 3C, H, W]
        D = self.compress(D_raw)                       # [B, 3, H, W]
        return delta, D


class _SpectralExpert(nn.Module):
    """Spectral-preservation expert: depthwise-separable convolutions that operate
    independently per band to avoid spectral mixing. Ideal for regions where
    HSI spectral curves are reliable and should be preserved."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            # Depthwise: per-channel spatial smoothing
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.LeakyReLU(0.1, True),
            # Pointwise: cross-band spectral mixing (careful, small kernel)
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _EdgeExpert(nn.Module):
    """Edge-reconstruction expert: uses Laplacian-guided residual to sharpen
    spatial edges. Primarily useful in regions where the MSI provides reliable
    high-frequency spatial structure."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 2, dilation=2)  # dilated
        self.conv3 = nn.Conv2d(channels * 2, channels, 1)
        self.act = nn.LeakyReLU(0.1, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.act(self.conv1(x))
        f2 = self.act(self.conv2(x))
        return self.conv3(torch.cat([f1, f2], dim=1))


class _TextureSmoothExpert(nn.Module):
    """Texture and smooth region expert: operates at two scales simultaneously.
    A wide kernel (5x5) captures smooth region context while a 3x3 handles
    mid-frequency textures. Combines both for adaptive reconstruction."""

    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.branch_smooth = nn.Sequential(
            nn.Conv2d(channels, mid, 5, 1, 2), nn.LeakyReLU(0.1, True),
        )
        self.branch_texture = nn.Sequential(
            nn.Conv2d(channels, mid, 3, 1, 1), nn.LeakyReLU(0.1, True),
        )
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.branch_smooth(x)
        t = self.branch_texture(x)
        return self.fuse(torch.cat([s, t], dim=1))


class _CrossModalCorrectionExpert(nn.Module):
    """Cross-modal correction expert: explicitly corrects the fused features
    using the disagreement signal. This expert is most active in high-disagreement
    regions where one modality's information dominates."""

    def __init__(self, channels: int, disagree_ch: int = 3):
        super().__init__()
        # Takes fused features + disagreement map
        self.conv_in = nn.Conv2d(channels + disagree_ch, channels, 1)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.alpha = nn.Parameter(torch.zeros(1))   # zero-init: start as no-op

    def forward(self, x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(torch.cat([x, D], dim=1))
        return x + torch.tanh(self.alpha) * self.body(h)


class DegradationConditionedMoE(nn.Module):
    """Degradation-conditioned semantic expert routing.

    This is the central new mechanism of DAETF-Net v3. Unlike RegionAwareMoE
    which uses a generic gate with no degradation awareness, this module:

    1. Uses 4 semantically specialised experts (spectral, edge, texture/smooth,
       cross-modal correction) — each designed for a different reconstruction need.
    2. Routes using: gate(F_fused, d, D) where d is the degradation code and
       D is the cross-modal disagreement field.
    3. The correction expert directly uses the disagreement field, so the routing
       policy adapts to local modality reliability.

    Physical motivation:
      - Under strong blur  → edge expert dominates
      - Under spectral noise → spectral expert dominates
      - In high-disagreement regions → correction expert activates
      - In flat/homogeneous regions → texture/smooth expert activates
    """

    def __init__(self, channels: int, code_dim: int, disagree_ch: int = 3,
                 topk: int = 2):
        super().__init__()
        self.channels = channels
        self.topk = min(topk, 4)

        # 4 semantic experts
        self.e_spectral = _SpectralExpert(channels)
        self.e_edge = _EdgeExpert(channels)
        self.e_texture = _TextureSmoothExpert(channels)
        self.e_correction = _CrossModalCorrectionExpert(channels, disagree_ch)

        # Gate input: fused features (C) + disagreement (3) + deg code (code_dim)
        # The degradation code is broadcast spatially before concatenation
        self.deg_proj = nn.Linear(code_dim, channels // 4)   # project d to spatial dim
        gate_in = channels + disagree_ch + channels // 4
        self.gate = nn.Sequential(
            nn.Conv2d(gate_in, channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels // 2, 4, 1),               # 4 expert logits
        )

        self.last_gate: Optional[torch.Tensor] = None
        self._gate: Optional[torch.Tensor] = None
        self.gate_noise = 1.0      # logit noise during training (exploration)
        self.gate_floor = 0.01     # uniform floor so no expert is ever starved

    def forward(self, x: torch.Tensor, d: torch.Tensor,
                D: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] fused features
            d: [B, code_dim]  degradation code
            D: [B, 3, H, W]  disagreement field
        Returns:
            out: [B, C, H, W] routed expert output
        """
        b, c, h, w = x.shape

        # Broadcast degradation code to spatial domain
        d_proj = self.deg_proj(d)                           # [B, C/4]
        d_spatial = d_proj[:, :, None, None].expand(b, -1, h, w)  # [B, C/4, H, W]

        # Gate: sees fused features + disagreement + degradation
        gate_in = torch.cat([x, D, d_spatial], dim=1)      # [B, C+3+C/4, H, W]
        logits = self.gate(gate_in)                         # [B, 4, H, W]

        # --- Top-k sparse routing -----------------------------------------
        # Noisy top-k during training (Shazeer et al.): without exploration the
        # ranking at initialisation is self-fulfilling - an expert outside the
        # top-k has its output multiplied by exactly zero, receives no gradient,
        # never improves, and so is never selected again. Measured across three
        # seeds, 1-2 of the 4 semantic experts were dead from initialisation,
        # which would have made the 'experts specialise' claim unsupportable.
        if self.training and self.gate_noise > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise
        if self.topk < 4:
            thresh = logits.topk(self.topk, dim=1).values[:, -1:, :, :]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        g = logits.softmax(dim=1)                           # [B, 4, H, W]

        # Uniform floor so every expert keeps a gradient path even when not
        # selected. eps is tiny, so routing stays effectively sparse; at
        # evaluation it is dropped entirely and the routing is exactly top-k.
        if self.training and self.gate_floor > 0:
            g = (1.0 - self.gate_floor) * g + self.gate_floor / 4.0

        # Keep BOTH: the differentiable gate for the balance loss, and a
        # detached copy for diagnostics. Previously only the detached copy was
        # stored, so balance_loss() had no path to the gating network and the
        # load-balancing term contributed exactly zero gradient - the mechanism
        # meant to prevent collapse was itself inert.
        self._gate = g
        self.last_gate = g.detach()

        # Expert outputs
        e0 = self.e_spectral(x)
        e1 = self.e_edge(x)
        e2 = self.e_texture(x)
        e3 = self.e_correction(x, D)

        out = (g[:, 0:1] * e0 +
               g[:, 1:2] * e1 +
               g[:, 2:3] * e2 +
               g[:, 3:4] * e3)
        return out + x    # residual connection

    def balance_loss(self) -> torch.Tensor:
        """Squared coefficient of variation of expert usage; 0 when uniform.

        Computed from the DIFFERENTIABLE gate. Using the detached diagnostic
        copy - as this did previously - makes the term a constant and the
        anti-collapse mechanism a no-op.
        """
        if self._gate is None:
            return torch.zeros((), device=next(self.parameters()).device)
        imp = self._gate.mean(dim=(0, 2, 3))        # [4]
        return 4.0 * (imp ** 2).sum() - 1.0

    @torch.no_grad()
    def expert_usage(self) -> Optional[torch.Tensor]:
        """Per-expert mean gate weight [4] — used for the conflict matrix figure."""
        if self.last_gate is None:
            return None
        return self.last_gate.mean(dim=(0, 2, 3))

    @torch.no_grad()
    def expert_usage_map(self) -> Optional[torch.Tensor]:
        """Return last gate map [4, H, W] for spatial visualisation."""
        if self.last_gate is None:
            return None
        return self.last_gate[0]     # first batch element


# ============================================================= END NEW MODULES =

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
    cross-subband mixing, then an exact inverse transform."""

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


# ----------------------------------------------------------------- utility modules
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Adaptively re-weights spectral/feature channels based on global context.
    Applied after Tucker fusion to let the model emphasise informative bands
    and suppress noisy or redundant ones.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.body(x)[:, :, None, None]
        return x * w


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB): dense connections with local residual learning.

    Three dense layers where each layer receives the concatenation of all
    preceding features, followed by a 1x1 bottleneck and a local skip.

    Ref: Zhang et al., "Residual Dense Network for Image Super-Resolution", CVPR 2018.
    """

    def __init__(self, channels: int, growth: int = 32, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = channels
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_ch, growth, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
            ))
            in_ch += growth
        self.bottleneck = nn.Conv2d(in_ch, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.layers:
            out = layer(torch.cat(feats, dim=1))
            feats.append(out)
        return x + self.bottleneck(torch.cat(feats, dim=1)) * 0.2


class GeometricSelfEnsemble(nn.Module):
    """Eight-fold geometric self-ensemble at test time.

    Averages predictions over all combinations of 4 rotations x 2 flips.
    This is a free ~0.1-0.3 dB PSNR boost with zero retraining cost, commonly
    used in image restoration competitions. The p4-equivariant stem makes
    DAETF-Net especially well-suited to benefit from this.
    """

    @staticmethod
    @torch.no_grad()
    def forward_ensemble(model, lr: torch.Tensor, msi: torch.Tensor,
                         inference_fn) -> torch.Tensor:
        """Apply model with 8 augmentations, average the de-augmented outputs."""
        preds = []
        for flip in [False, True]:
            for rot in range(4):
                lr_aug = lr
                msi_aug = msi
                if flip:
                    lr_aug = torch.flip(lr_aug, [-1])
                    msi_aug = torch.flip(msi_aug, [-1])
                if rot > 0:
                    lr_aug = torch.rot90(lr_aug, rot, [-2, -1])
                    msi_aug = torch.rot90(msi_aug, rot, [-2, -1])

                pred = inference_fn(model, lr_aug, msi_aug)

                # de-augment
                if rot > 0:
                    pred = torch.rot90(pred, -rot, [-2, -1])
                if flip:
                    pred = torch.flip(pred, [-1])
                preds.append(pred)

        return torch.stack(preds).mean(dim=0).clamp(0, 1)
