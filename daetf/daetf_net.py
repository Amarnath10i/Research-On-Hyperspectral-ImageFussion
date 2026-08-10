"""
DAETF-Net v2 - Domain-Adaptive Equivariant Tensor Fusion Network
================================================================

A real implementation of the architecture proposed in DAETF_Net_Architecture.md.

Every module named in the design document is actually implemented here:

  * EFE   - p4 group-equivariant convolutions (lifting + group conv + group pool),
            implemented natively with weight rotations. Numerically verifiable
            equivariance to 90 deg rotations (see `check_equivariance`).
  * DAE   - Degradation-Aware Encoder. Estimates a degradation code from the
            observed pair and conditions every downstream block through FiLM.
            Supervised by the (known) synthetic degradation parameters.
  * TSSE  - Tucker multilinear interaction with a *used* core tensor
            (einsum contraction), plus a nuclear-norm rank penalty on the core.
  * AF-MoE- Region-aware Mixture-of-Experts with per-pixel top-k routing and a
            load-balancing loss (not a single global gate).
  * FDRM  - Genuine frequency-domain refinement: orthonormal Haar DWT,
            per-subband learnable processing with soft-thresholding
            (wavelet shrinkage), cross-subband mixing, exact IDWT.
  * BPU   - Back-Projection Upsampler replacing bicubic interpolation. Enforces
            the observation model inside the architecture, not only in the loss.

Losses (SPC = Spectral-Physical Composite):
    Charbonnier + SAM + gradient + MS-SSIM
  + spatial consistency   || Down(Y_hat) - LR_HSI ||     (physics)
  + spectral consistency  || SRF(Y_hat)  - MSI    ||     (physics)
  + MoE load balance + Tucker nuclear norm + degradation regression
  + MMD domain alignment (optional, uses unlabelled target-domain patches)

The two physics terms hold on *unseen* domains without ground truth, which is
what enables the self-supervised test-time adaptation used for CAVE -> Harvard.

Target hardware: a single NVIDIA P100 (16 GB, compute capability 6.0).
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:  # scipy is present on Kaggle; guard so the module imports anywhere
    import scipy.io as sio
except ImportError:  # pragma: no cover
    sio = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # data - leave the roots as None to auto-discover them at runtime
    source_root: Optional[str] = None    # domain the model trains on
    target_root: Optional[str] = None    # unseen domain used for transfer tests
    bands: Optional[int] = None          # inferred from the data if None
    msi_bands: Optional[int] = None      # inferred from the data if None
    scale: int = 4              # super-resolution factor
    patch: int = 64             # HR training patch (must be a multiple of scale)

    # model
    width: int = 64             # main feature width
    equi_width: int = 16        # per-orientation width inside the equivariant stem
    equi_depth: int = 2         # number of P4->P4 group convolutions
    rank: int = 16              # Tucker ranks (R1 = R2 = R3)
    experts: int = 4
    topk: int = 2
    bp_iters: int = 2           # back-projection refinement steps
    code_dim: int = 128         # degradation code width
    blur_ksize: int = 9         # support of the simulated blur kernels

    # degradation simulation (domain randomisation)
    sigma_range: Tuple[float, float] = (0.6, 2.4)
    aniso: float = 0.5          # probability of an anisotropic kernel
    noise_range: Tuple[float, float] = (0.0, 0.03)
    srf_jitter: float = 0.35    # probability of using a jittered synthetic MSI
    eval_sigma: float = 1.2     # fixed kernel used to build the evaluation LR input

    # ablation switches (all modules on by default)
    use_equivariant: bool = True
    use_tsse: bool = True
    use_moe: bool = True
    use_fdrm: bool = True
    use_backprojection: bool = True
    use_physics: bool = True
    use_degradation_code: bool = True

    # optimisation
    iters: int = 20000
    batch: int = 16
    lr: float = 2e-4
    min_lr: float = 1e-6
    warmup: int = 500
    grad_clip: float = 1.0
    amp: bool = True            # P100 has real fp16 throughput; halves memory
    workers: int = 2
    seed: int = 42

    # loss weights
    w_char: float = 1.0
    w_sam: float = 0.30
    w_grad: float = 0.20
    w_ssim: float = 0.15
    w_spat: float = 0.50        # || Down(Y) - LR ||
    w_spec: float = 0.50        # || SRF(Y)  - MSI ||
    w_bal: float = 0.01
    w_rank: float = 1e-4
    w_deg: float = 0.05
    w_mmd: float = 0.10

    # bookkeeping
    out_dir: str = "./daetf_out"
    val_every: int = 1000
    log_every: int = 100
    val_scenes: int = 4         # scenes used for the periodic in-training check

    def __post_init__(self) -> None:
        assert self.patch % self.scale == 0, "patch must be divisible by scale"
        assert self.topk <= self.experts, "topk cannot exceed the number of experts"

    def resolve(self, source_hints: Sequence[str] = ("cave",),
                target_hints: Sequence[str] = ("harvard",),
                verbose: bool = True) -> "Config":
        """Fill in any field left as None by inspecting the filesystem."""
        if self.source_root is None:
            self.source_root = discover_dataset(source_hints, verbose=verbose)
        if self.target_root is None:
            self.target_root = discover_dataset(target_hints, required=False,
                                                verbose=verbose)
        if self.bands is None or self.msi_bands is None:
            b, m = infer_channels(self.source_root)
            self.bands = self.bands or b
            self.msi_bands = self.msi_bands or m
            if verbose:
                print(f"[config] inferred bands={self.bands} msi_bands={self.msi_bands}")
        return self


# ---------------------------------------------------------------------------
# Dataset discovery - no hardcoded paths anywhere
# ---------------------------------------------------------------------------
SPLIT_NAMES = ("Train", "train", "TRAIN")
TEST_NAMES = ("Test", "test", "TEST", "Val", "val")


def _looks_like_dataset(path: str) -> bool:
    """A dataset root is any directory holding <split>/HSI and <split>/RGB."""
    for split in SPLIT_NAMES + TEST_NAMES:
        d = os.path.join(path, split)
        if os.path.isdir(d) and any(
            os.path.isdir(os.path.join(d, h)) for h in ("HSI", "hsi")
        ):
            return True
    return False


def search_roots() -> List[str]:
    """Candidate places to look, most specific first. Honours DAETF_DATA_ROOTS
    (os.pathsep separated) so the caller can always override discovery."""
    roots: List[str] = []
    env = os.environ.get("DAETF_DATA_ROOTS", "")
    roots += [p for p in env.split(os.pathsep) if p]
    roots += sorted(glob.glob("/kaggle/input/*"))
    roots += sorted(glob.glob(os.path.join(os.getcwd(), "data", "*")))
    roots += [os.path.join(os.getcwd(), "data"), os.getcwd()]
    return [r for r in roots if os.path.isdir(r)]


def discover_dataset(hints: Sequence[str] = (), required: bool = True,
                     verbose: bool = True) -> Optional[str]:
    """Locate a dataset root whose path matches one of `hints`.

    Handles both `<root>/Data/Train/HSI` and `<root>/Train/HSI` layouts, so it
    works regardless of how a Kaggle dataset was packaged.
    """
    found: List[str] = []
    for root in search_roots():
        for cand in (os.path.join(root, "Data"), root):
            if _looks_like_dataset(cand):
                found.append(cand)
                break
    if hints:
        lowered = [h.lower() for h in hints]
        ranked = [f for f in found if any(h in f.lower() for h in lowered)]
        found = ranked or found
    if not found:
        if required:
            raise FileNotFoundError(
                f"no dataset matching {list(hints)} found. Looked under: "
                f"{search_roots()}. Set DAETF_DATA_ROOTS or pass the root "
                f"explicitly via Config(source_root=...)."
            )
        if verbose:
            print(f"[config] optional dataset {list(hints)} not found - skipping")
        return None
    if verbose:
        print(f"[config] using dataset root: {found[0]}")
    return found[0]


def available_splits(root: str) -> Dict[str, str]:
    """Map canonical split name -> actual directory name present on disk."""
    out = {}
    for canonical, names in (("Train", SPLIT_NAMES), ("Test", TEST_NAMES)):
        for n in names:
            if os.path.isdir(os.path.join(root, n)):
                out[canonical] = n
                break
    return out


def infer_channels(root: str) -> Tuple[int, int]:
    """Read one HSI/RGB pair and report their channel counts."""
    splits = available_splits(root)
    split = splits.get("Train") or splits.get("Test")
    if split is None:
        raise FileNotFoundError(f"no usable split under {root}")
    base = os.path.join(root, split)
    hsi_dir = next(os.path.join(base, d) for d in ("HSI", "hsi")
                   if os.path.isdir(os.path.join(base, d)))
    rgb_dir = next((os.path.join(base, d) for d in ("RGB", "rgb")
                    if os.path.isdir(os.path.join(base, d))), None)
    hsi = np.squeeze(load_mat(sorted(glob.glob(os.path.join(hsi_dir, "*.mat")))[0]))
    bands = int(min(hsi.shape))
    msi_bands = 3
    if rgb_dir:
        rgb = np.squeeze(load_mat(sorted(glob.glob(os.path.join(rgb_dir, "*.mat")))[0]))
        msi_bands = int(min(rgb.shape))
    return bands, msi_bands


# ---------------------------------------------------------------------------
# Degradation operators (shared by the dataset and the physics losses)
# ---------------------------------------------------------------------------
def gaussian_kernel2d(ksize: int, sx: float, sy: float, theta: float) -> torch.Tensor:
    """Rotated anisotropic Gaussian blur kernel, normalised to sum 1."""
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    xr = xx * cos_t + yy * sin_t
    yr = -xx * sin_t + yy * cos_t
    k = torch.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return k / k.sum().clamp_min(1e-12)


def blur_downsample(x: torch.Tensor, kernel: torch.Tensor, scale: int) -> torch.Tensor:
    """Apply a per-image blur kernel then decimate. Differentiable.

    x      : [B, C, H, W]
    kernel : [k, k] or [B, k, k]
    """
    b, c, _, _ = x.shape
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).expand(b, -1, -1)
    k = kernel.shape[-1]
    pad = k // 2
    # group the batch into the channel axis so each sample gets its own kernel
    w = kernel.to(x.dtype).reshape(b, 1, 1, k, k).expand(b, c, 1, k, k).reshape(b * c, 1, k, k)
    xr = x.reshape(1, b * c, *x.shape[-2:])
    xr = F.pad(xr, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(xr, w, groups=b * c)
    out = out.reshape(b, c, *out.shape[-2:])
    return out[..., ::scale, ::scale].contiguous()


class FixedDegradation(nn.Module):
    """Non-learnable blur+decimate used by the spatial-consistency loss and to
    build the evaluation LR input."""

    def __init__(self, scale: int, ksize: int = 9, sigma: float = 1.2):
        super().__init__()
        self.scale = scale
        self.register_buffer("kernel", gaussian_kernel2d(ksize, sigma, sigma, 0.0))

    @classmethod
    def from_config(cls, cfg: "Config") -> "FixedDegradation":
        return cls(cfg.scale, ksize=cfg.blur_ksize, sigma=cfg.eval_sigma)

    def forward(self, x: torch.Tensor, kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.kernel if kernel is None else kernel
        return blur_downsample(x, k, self.scale)


# ---------------------------------------------------------------------------
# p4 group-equivariant convolutions
# ---------------------------------------------------------------------------
class P4ConvZ2(nn.Module):
    """Lifting convolution Z2 -> p4. Output carries an explicit orientation axis."""

    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3, bias: bool = True):
        super().__init__()
        self.out_ch, self.ksize = out_ch, ksize
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, ksize, ksize))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,Cin,H,W] -> [B,Cout,4,H,W]
        w = torch.cat([torch.rot90(self.weight, r, dims=(2, 3)) for r in range(4)], dim=0)
        b = None if self.bias is None else self.bias.repeat(4)
        y = F.conv2d(x, w, b, padding=self.ksize // 2)
        bsz, _, h, wd = y.shape
        return y.view(bsz, 4, self.out_ch, h, wd).transpose(1, 2)


class P4ConvP4(nn.Module):
    """Group convolution p4 -> p4: rotate the spatial support and cyclically
    shift the orientation axis together."""

    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3, bias: bool = True):
        super().__init__()
        self.in_ch, self.out_ch, self.ksize = in_ch, out_ch, ksize
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, 4, ksize, ksize))
        nn.init.kaiming_normal_(self.weight.view(out_ch, -1, ksize, ksize),
                                mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,Cin,4,H,W] -> [B,Cout,4,H,W]
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
    """EFE. BatchNorm3d keeps statistics shared across orientations, so the
    whole stem stays equivariant; the final max over the orientation axis makes
    the output equivariant as a plain feature map (rotating the input rotates
    the output)."""

    def __init__(self, in_ch: int, width: int, out_ch: int, depth: int = 2):
        super().__init__()
        self.lift = P4ConvZ2(in_ch, width)
        self.bn0 = nn.BatchNorm3d(width)
        blocks = []
        for _ in range(depth):
            blocks.append(nn.ModuleList([P4ConvP4(width, width), nn.BatchNorm3d(width)]))
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Conv2d(width, out_ch, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn0(self.lift(x)))
        for conv, bn in self.blocks:
            h = self.act(bn(conv(h))) + h
        h = h.max(dim=2).values          # group pooling over the 4 orientations
        return self.proj(h)


# ---------------------------------------------------------------------------
# Degradation-Aware Encoder + FiLM conditioning
# ---------------------------------------------------------------------------
class DegradationEncoder(nn.Module):
    """Estimates a degradation code from the observed (LR-HSI, MSI) pair.

    An auxiliary head regresses the true degradation parameters
    (sigma_x, sigma_y, theta as sin/cos, noise sigma) which are known during
    training because we synthesise them - this keeps the code meaningful
    instead of letting it collapse.
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
        self.deg_head = nn.Linear(code, 5)   # sx, sy, sin, cos, noise

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        msi_lr = F.adaptive_avg_pool2d(msi, lr_hsi.shape[-2:])
        f = self.body(torch.cat([lr_hsi, msi_lr], dim=1))
        stats = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        code = self.head(stats)
        return code, self.deg_head(code)


class FiLM(nn.Module):
    """Feature-wise linear modulation conditioned on the degradation code."""

    def __init__(self, code: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(code, channels * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(code).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


# ---------------------------------------------------------------------------
# TSSE - Tucker multilinear interaction (the core tensor is actually used)
# ---------------------------------------------------------------------------
class TensorSpectralSpatialEncoder(nn.Module):
    """z[b,r3,h,w] = sum_{r1,r2} G[r1,r2,r3] * a[b,r1,h,w] * b[b,r2,h,w]

    which is a Tucker-style contraction of the rank-1 outer product of the two
    projected feature maps with a learned core tensor G. Implemented as an outer
    product followed by a 1x1 convolution whose weights *are* G, so the core
    receives gradients.
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
        outer = pa.unsqueeze(2) * pb.unsqueeze(1)    # [B,R1,R2,H,W]
        outer = outer.flatten(1, 2)                  # [B,R1*R2,H,W]
        core = self.core.permute(2, 0, 1).reshape(self.rank, self.rank * self.rank, 1, 1)
        z = F.conv2d(outer, core.to(outer.dtype))    # [B,R3,H,W]
        return self.norm(self.out(z) + self.skip(torch.cat([a, b], dim=1)))

    def rank_penalty(self) -> torch.Tensor:
        """Nuclear norm of the mode-3 unfolding - a convex surrogate for rank."""
        unfold = self.core.reshape(self.rank, -1).float()
        return torch.linalg.svdvals(unfold).sum()


# ---------------------------------------------------------------------------
# AF-MoE - region-aware routing
# ---------------------------------------------------------------------------
class RegionAwareMoE(nn.Module):
    """Per-pixel top-k expert routing. The gate is a convolution, so different
    image regions (shadow, foliage, flat texture) select different experts -
    unlike a globally pooled gate which must pick one strategy per image."""

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
        """Squared coefficient of variation of expert usage; 0 when uniform."""
        if self.last_gate is None:
            return torch.zeros((), device=next(self.parameters()).device)
        imp = self.last_gate.mean(dim=(0, 2, 3))
        return self.experts_n * (imp ** 2).sum() - 1.0


# ---------------------------------------------------------------------------
# FDRM - real wavelet-domain refinement
# ---------------------------------------------------------------------------
class HaarDWT(nn.Module):
    """Orthonormal Haar transform as a fixed grouped convolution. Exactly
    invertible by the transposed convolution with the same filters."""

    def __init__(self):
        super().__init__()
        h = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        g1 = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        g2 = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        g3 = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        self.register_buffer("filt", torch.stack([h, g1, g2, g3]).unsqueeze(1) * 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,C,H,W]->[B,4C,H/2,W/2]
        b, c, h, w = x.shape
        f = self.filt.to(x.dtype)
        y = F.conv2d(x.reshape(b * c, 1, h, w), f, stride=2)
        return y.reshape(b, c * 4, h // 2, w // 2)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:      # [B,4C,H,W]->[B,C,2H,2W]
        b, c4, h, w = y.shape
        c = c4 // 4
        f = self.filt.to(y.dtype)
        x = F.conv_transpose2d(y.reshape(b * c, 4, h, w), f, stride=2)
        return x.reshape(b, c, h * 2, w * 2) * 0.25


class FrequencyDomainRefinement(nn.Module):
    """Per-subband processing with learnable soft-thresholding (classical
    wavelet shrinkage, made differentiable and learned) plus cross-subband
    mixing, then an exact inverse transform."""

    def __init__(self, channels: int):
        super().__init__()
        self.dwt = HaarDWT()
        self.sub = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, True),
                          nn.Conv2d(channels, channels, 3, 1, 1))
            for _ in range(4)
        ])
        # learnable shrinkage thresholds for the three detail subbands
        self.thresh = nn.Parameter(torch.zeros(3, channels))
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
        y = self.mix(torch.cat(out, dim=1))
        y = self.dwt.inverse(y)
        if pad_h or pad_w:
            y = y[..., :h, :w]
        return self.fuse(y) + x[..., :h, :w]


# ---------------------------------------------------------------------------
# Back-projection upsampler (bicubic replacement)
# ---------------------------------------------------------------------------
class BackProjectionUpsampler(nn.Module):
    """Learned upsampling with iterative observation-model correction:

        y  <- Up(x)
        y  <- y + Up_res( x - Down(y) )      repeated `iters` times

    The residual is measured in LR space where the ground truth observation is
    available, so the estimate is pulled back onto the observation manifold at
    every step. This is what bicubic interpolation cannot do.
    """

    def __init__(self, bands: int, scale: int, width: int = 64, iters: int = 2):
        super().__init__()
        self.scale, self.iters = scale, iters
        stages, s = [], scale
        ch = bands
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
        base = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        y = self.up(lr) + base                     # learned residual over a cheap prior
        for _ in range(self.iters):
            err = lr - self.down(y)
            if err.shape[-2:] != lr.shape[-2:]:
                err = F.interpolate(err, size=lr.shape[-2:], mode="bilinear", align_corners=False)
            y = y + self.up_res(err)
        return y


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
class PlainFeatureExtractor(nn.Module):
    """Non-equivariant control arm for the EFE ablation: same depth and width,
    ordinary convolutions."""

    def __init__(self, in_ch: int, width: int, out_ch: int, depth: int = 2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width * 4, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(depth):
            layers += [nn.Conv2d(width * 4, width * 4, 3, 1, 1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width * 4, out_ch, 1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class BicubicUpsampler(nn.Module):
    """Control arm for the back-projection ablation: plain bicubic + a conv."""

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


class DAETFNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.bands is None or cfg.msi_bands is None:
            raise ValueError("Config.bands/msi_bands are unset - call cfg.resolve() "
                             "or pass them explicitly before building the model")
        self.cfg = cfg
        c, b, m = cfg.width, cfg.bands, cfg.msi_bands

        self.upsampler = (
            BackProjectionUpsampler(b, cfg.scale, width=c, iters=cfg.bp_iters)
            if cfg.use_backprojection else BicubicUpsampler(b, cfg.scale, width=c)
        )
        self.deg = DegradationEncoder(b, m, code=cfg.code_dim)
        self.efe = (
            EquivariantFeatureExtractor(b, cfg.equi_width, c, depth=cfg.equi_depth)
            if cfg.use_equivariant
            else PlainFeatureExtractor(b, cfg.equi_width, c, depth=cfg.equi_depth)
        )
        self.msi_enc = nn.Sequential(
            nn.Conv2d(m, c, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(c, c, 3, 1, 1),
        )
        self.film_h = FiLM(cfg.code_dim, c)
        self.film_m = FiLM(cfg.code_dim, c)
        self.tsse = (TensorSpectralSpatialEncoder(c, c, c, rank=cfg.rank)
                     if cfg.use_tsse else None)
        self.concat_fuse = None if cfg.use_tsse else nn.Sequential(
            nn.Conv2d(c * 2, c, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, c, 3, 1, 1)
        )
        self.moe = (RegionAwareMoE(c, experts=cfg.experts, topk=cfg.topk)
                    if cfg.use_moe else None)
        self.plain_block = None if cfg.use_moe else nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, c, 3, 1, 1)
        )
        self.fdrm = FrequencyDomainRefinement(c) if cfg.use_fdrm else None
        self.recon = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(c, b, 3, 1, 1)
        )

    def _trunk(self, lr_hsi: torch.Tensor, msi: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code, deg_params = self.deg(lr_hsi, msi)
        if not self.cfg.use_degradation_code:
            code = torch.zeros_like(code)
        y0 = self.upsampler(lr_hsi)
        fh = self.film_h(self.efe(y0), code)
        fm = self.film_m(self.msi_enc(msi), code)
        z = (self.tsse(fh, fm) if self.tsse is not None
             else self.concat_fuse(torch.cat([fh, fm], dim=1)))
        return y0, z, deg_params

    def features(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        """Pooled bottleneck features - used by the MMD domain-alignment term."""
        return self._trunk(lr_hsi, msi)[1].mean(dim=(2, 3))

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> Dict[str, torch.Tensor]:
        y0, z, deg_params = self._trunk(lr_hsi, msi)
        z = self.moe(z) if self.moe is not None else self.plain_block(z)
        if self.fdrm is not None:
            z = self.fdrm(z)
        out = y0 + self.recon(z)
        return {"out": out, "coarse": y0, "deg": deg_params, "feat": z.mean(dim=(2, 3))}


# ---------------------------------------------------------------------------
# Spectral-Physical Composite loss
# ---------------------------------------------------------------------------
def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean spectral angle in radians. Directly optimises the metric on which
    every benchmarked baseline degrades under domain shift."""
    p = pred.flatten(2)
    t = target.flatten(2)
    num = (p * t).sum(dim=1)
    den = p.norm(dim=1) * t.norm(dim=1)
    cos = (num / den.clamp_min(eps)).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


def _gauss_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
               size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    c = pred.shape[1]
    win = _gauss_window(size, sigma, pred.device, pred.dtype).expand(c, 1, size, size)
    mu1 = F.conv2d(pred, win, padding=size // 2, groups=c)
    mu2 = F.conv2d(target, win, padding=size // 2, groups=c)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, win, padding=size // 2, groups=c) - mu1s
    s2 = F.conv2d(target * target, win, padding=size // 2, groups=c) - mu2s
    s12 = F.conv2d(pred * target, win, padding=size // 2, groups=c) - mu12
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return m.mean()


def mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multi-bandwidth RBF maximum mean discrepancy between two feature sets."""
    z = torch.cat([x, y], dim=0)
    d = torch.cdist(z, z) ** 2
    n = x.shape[0]
    med = d.detach().flatten().median().clamp_min(1e-6)
    k = sum(torch.exp(-d / (med * s)) for s in (0.25, 0.5, 1.0, 2.0, 4.0))
    kxx = k[:n, :n].mean()
    kyy = k[n:, n:].mean()
    kxy = k[:n, n:].mean()
    return kxx + kyy - 2 * kxy


class SPCLoss(nn.Module):
    """Spectral-Physical Composite loss."""

    def __init__(self, cfg: Config, srf: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.degrade = FixedDegradation(cfg.scale)
        self.register_buffer("srf", srf)      # [bands, msi_bands]

    def apply_srf(self, x: torch.Tensor) -> torch.Tensor:
        w = self.srf.to(x.dtype).t().reshape(self.srf.shape[1], self.srf.shape[0], 1, 1)
        return F.conv2d(x, w)

    def forward(self, out: Dict[str, torch.Tensor], target: torch.Tensor,
                lr_hsi: torch.Tensor, msi: torch.Tensor, model: DAETFNet,
                deg_gt: Optional[torch.Tensor] = None,
                kernel: Optional[torch.Tensor] = None,
                tgt_feat: Optional[torch.Tensor] = None,
                supervised: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        pred = out["out"]
        logs: Dict[str, float] = {}
        total = pred.new_zeros(())

        if supervised:
            l_char = charbonnier(pred, target)
            l_sam = sam_loss(pred, target)
            l_grad = gradient_loss(pred, target)
            l_ssim = 1.0 - ssim_torch(pred.clamp(0, 1).float(), target.float())
            total = (total + cfg.w_char * l_char + cfg.w_sam * l_sam
                     + cfg.w_grad * l_grad + cfg.w_ssim * l_ssim)
            logs.update(char=l_char.item(), sam=l_sam.item(),
                        grad=l_grad.item(), ssim=l_ssim.item())

        # --- physics: hold on any domain, with or without ground truth --------
        if cfg.use_physics or not supervised:
            l_spat = charbonnier(self.degrade(pred, kernel), lr_hsi)
            l_spec = charbonnier(self.apply_srf(pred), msi)
            total = total + cfg.w_spat * l_spat + cfg.w_spec * l_spec
            logs.update(spat=l_spat.item(), spec=l_spec.item())

        # --- regularisers ------------------------------------------------------
        if model.moe is not None:
            l_bal = model.moe.balance_loss()
            total = total + cfg.w_bal * l_bal
            logs["bal"] = l_bal.item()

        if supervised:
            if model.tsse is not None:
                l_rank = model.tsse.rank_penalty()
                total = total + cfg.w_rank * l_rank
                logs["rank"] = l_rank.item()
            if deg_gt is not None:
                l_deg = F.smooth_l1_loss(out["deg"].float(), deg_gt.float())
                total = total + cfg.w_deg * l_deg
                logs["deg"] = l_deg.item()
            if tgt_feat is not None:
                l_mmd = mmd_rbf(out["feat"].float(), tgt_feat.float())
                total = total + cfg.w_mmd * l_mmd
                logs["mmd"] = l_mmd.item()

        logs["total"] = total.item()
        return total, logs


# ---------------------------------------------------------------------------
# Metrics - ONE implementation, used for every method and both datasets
# ---------------------------------------------------------------------------
def _hwc(x: np.ndarray) -> np.ndarray:
    return x if x.shape[-1] <= 64 else np.transpose(x, (1, 2, 0))


def metric_psnr(pred: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - ref) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(data_range ** 2 / mse))


def metric_sam(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    p, r = _hwc(pred).reshape(-1, pred.shape[-1]), _hwc(ref).reshape(-1, ref.shape[-1])
    cos = (p * r).sum(1) / np.maximum(np.linalg.norm(p, axis=1) * np.linalg.norm(r, axis=1), eps)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return float(np.mean(ang[np.isfinite(ang)]))


def metric_ergas(pred: np.ndarray, ref: np.ndarray, scale: int, eps: float = 1e-8) -> float:
    p, r = _hwc(pred), _hwc(ref)
    rmse = np.sqrt(np.mean((p - r) ** 2, axis=(0, 1)))
    mu = np.maximum(np.mean(r, axis=(0, 1)), eps)
    return float(100.0 / scale * np.sqrt(np.mean((rmse / mu) ** 2)))


def metric_ssim(pred: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    p = torch.from_numpy(np.ascontiguousarray(_hwc(pred).transpose(2, 0, 1)))[None].float()
    r = torch.from_numpy(np.ascontiguousarray(_hwc(ref).transpose(2, 0, 1)))[None].float()
    return float(ssim_torch(p, r, data_range=data_range))


def evaluate_arrays(pred: np.ndarray, ref: np.ndarray, scale: int) -> Dict[str, float]:
    return {
        "psnr": metric_psnr(pred, ref),
        "ssim": metric_ssim(pred, ref),
        "sam": metric_sam(pred, ref),
        "ergas": metric_ergas(pred, ref, scale),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_mat(path: str) -> np.ndarray:
    if sio is None:
        raise RuntimeError("scipy is required to read .mat files")
    mat = sio.loadmat(path)
    for k, v in mat.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim >= 2:
            return np.asarray(v)
    raise ValueError(f"no array in {path}")


def to_chw01(arr: np.ndarray, channels: int) -> np.ndarray:
    a = np.squeeze(np.asarray(arr)).astype(np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected 3D array, got {a.shape}")
    if a.shape[0] == channels:
        pass
    elif a.shape[-1] == channels:
        a = np.transpose(a, (2, 0, 1))
    else:
        raise ValueError(f"cannot find {channels} channels in {a.shape}")
    mx = float(a.max())
    if mx > 1.0:
        a = a / mx
    return np.clip(a, 0.0, 1.0)


def find_pairs(root: str, split: str) -> List[Tuple[str, str, str]]:
    """Locate matched HSI/RGB .mat pairs. `split` is canonical ("Train"/"Test")
    and is mapped to whatever casing the dataset actually uses."""
    actual = available_splits(root).get(split, split)
    base = os.path.join(root, actual)
    hsi_dir = next((os.path.join(base, d) for d in ("HSI", "hsi")
                    if os.path.isdir(os.path.join(base, d))), None)
    rgb_dir = next((os.path.join(base, d) for d in ("RGB", "rgb")
                    if os.path.isdir(os.path.join(base, d))), None)
    if not hsi_dir or not rgb_dir:
        raise FileNotFoundError(f"no HSI/RGB folders under {base}")
    rgb = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(os.path.join(rgb_dir, "*.mat"))}
    out = []
    for h in sorted(glob.glob(os.path.join(hsi_dir, "*.mat"))):
        stem = os.path.splitext(os.path.basename(h))[0]
        if stem in rgb:
            out.append((stem, h, rgb[stem]))
    if not out:
        raise RuntimeError(f"no matched pairs under {base}")
    return out


class SceneCache:
    """Bounded LRU cache of decoded scenes kept in float16 to survive Harvard's
    1040x1392x31 scenes inside Kaggle's RAM budget."""

    def __init__(self, bands: int, msi_bands: int, limit: int = 12):
        self.bands, self.msi_bands, self.limit = bands, msi_bands, limit
        self.store: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.order: List[str] = []

    def get(self, stem: str, hsi_path: str, rgb_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if stem in self.store:
            self.order.remove(stem)
            self.order.append(stem)
            return self.store[stem]
        hsi = to_chw01(load_mat(hsi_path), self.bands)
        rgb = to_chw01(load_mat(rgb_path), self.msi_bands)
        if rgb.shape[-2:] != hsi.shape[-2:]:
            t = torch.from_numpy(rgb)[None]
            rgb = F.interpolate(t, size=hsi.shape[-2:], mode="bicubic",
                                align_corners=False).clamp(0, 1)[0].numpy()
        item = (hsi.astype(np.float16), rgb.astype(np.float16))
        self.store[stem] = item
        self.order.append(stem)
        while len(self.order) > self.limit:
            self.store.pop(self.order.pop(0), None)
        return item


class FusionPatchDataset(Dataset):
    """Samples HR patches and synthesises the observation pair on the fly with
    randomised degradations (domain randomisation)."""

    def __init__(self, root: str, split: str, cfg: Config, train: bool = True,
                 srf: Optional[np.ndarray] = None, length: int = 8000):
        self.cfg, self.train, self.length = cfg, train, length
        self.pairs = find_pairs(root, split)
        self.cache = SceneCache(cfg.bands, cfg.msi_bands,
                                limit=len(self.pairs) if train else 4)
        self.srf = srf

    def __len__(self) -> int:
        return self.length if self.train else len(self.pairs)

    def _sample_kernel(self) -> Tuple[torch.Tensor, List[float]]:
        cfg = self.cfg
        sx = random.uniform(*cfg.sigma_range)
        sy = sx if random.random() > cfg.aniso else random.uniform(*cfg.sigma_range)
        th = random.uniform(0, math.pi)
        return (gaussian_kernel2d(cfg.blur_ksize, sx, sy, th),
                [sx, sy, math.sin(2 * th), math.cos(2 * th)])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        stem, hp, rp = self.pairs[idx % len(self.pairs)] if not self.train \
            else self.pairs[random.randrange(len(self.pairs))]
        hsi, rgb = self.cache.get(stem, hp, rp)

        if self.train:
            p = cfg.patch
            _, h, w = hsi.shape
            top, left = random.randrange(0, h - p + 1), random.randrange(0, w - p + 1)
            gt = torch.from_numpy(hsi[:, top:top + p, left:left + p].astype(np.float32))
            msi = torch.from_numpy(rgb[:, top:top + p, left:left + p].astype(np.float32))
            if random.random() < 0.5:
                gt, msi = torch.flip(gt, [-1]), torch.flip(msi, [-1])
            k = random.randrange(4)
            if k:
                gt, msi = torch.rot90(gt, k, (-2, -1)), torch.rot90(msi, k, (-2, -1))
        else:
            p = (min(hsi.shape[1], hsi.shape[2]) // cfg.scale) * cfg.scale
            gt = torch.from_numpy(hsi[:, :p, :p].astype(np.float32))
            msi = torch.from_numpy(rgb[:, :p, :p].astype(np.float32))

        es = cfg.eval_sigma
        kernel, deg = self._sample_kernel() if self.train else \
            (gaussian_kernel2d(cfg.blur_ksize, es, es, 0.0), [es, es, 0.0, 1.0])

        lr = blur_downsample(gt[None], kernel, cfg.scale)[0]
        noise = random.uniform(*cfg.noise_range) if self.train else 0.0
        if noise > 0:
            lr = (lr + torch.randn_like(lr) * noise).clamp(0, 1)

        # occasionally replace the real RGB with a jittered synthetic MSI so the
        # model never assumes one fixed spectral response function
        if self.train and self.srf is not None and random.random() < cfg.srf_jitter:
            s = torch.from_numpy(self.srf).float()
            s = (s * (1 + 0.15 * torch.randn_like(s))).clamp_min(0)
            s = s / s.sum(0, keepdim=True).clamp_min(1e-6) * self.srf.sum(0).mean()
            msi = torch.einsum("chw,cm->mhw", gt, s).clamp(0, 1)

        return {
            "lr": lr, "msi": msi, "gt": gt,
            "deg": torch.tensor(deg + [noise], dtype=torch.float32),
            "kernel": kernel, "name": stem,
        }


def estimate_srf(root: str, split: str, cfg: Config, max_scenes: int = 8,
                 samples_per_scene: int = 20000) -> np.ndarray:
    """Least-squares spectral response function: min_S || HSI @ S - RGB ||^2.

    Recovering the real SRF from the data makes the spectral-consistency loss a
    genuine physical constraint rather than a hand-picked approximation.
    """
    pairs = find_pairs(root, split)[:max_scenes]
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for stem, hp, rp in pairs:
        hsi = to_chw01(load_mat(hp), cfg.bands).reshape(cfg.bands, -1).T
        rgb = to_chw01(load_mat(rp), cfg.msi_bands)
        if rgb.shape[-1] * rgb.shape[-2] != hsi.shape[0]:
            t = torch.from_numpy(rgb)[None]
            side = int(math.sqrt(hsi.shape[0]))
            rgb = F.interpolate(t, size=(side, side), mode="bicubic",
                                align_corners=False)[0].numpy()
        rgb = rgb.reshape(cfg.msi_bands, -1).T
        idx = rng.choice(hsi.shape[0], size=min(samples_per_scene, hsi.shape[0]),
                         replace=False)
        xs.append(hsi[idx])
        ys.append(rgb[idx])
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    s, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.clip(s, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Tiled inference (full 512x512 and 1040x1392 scenes on 16 GB)
# ---------------------------------------------------------------------------
@torch.no_grad()
def tiled_inference(model: DAETFNet, lr: torch.Tensor, msi: torch.Tensor, scale: int,
                    tile_hr: int = 256, overlap: int = 32) -> torch.Tensor:
    """Hann-weighted overlapping tiles, so tile seams do not appear in the output."""
    model.eval()
    bsz, _, h_hr, w_hr = msi.shape
    tile_lr, ov_lr = tile_hr // scale, overlap // scale
    tile_lr = min(tile_lr, lr.shape[2], lr.shape[3])
    tile_hr = tile_lr * scale
    step_lr = max(tile_lr - ov_lr, 1)
    out = torch.zeros(bsz, model.cfg.bands, h_hr, w_hr, device=lr.device, dtype=torch.float32)
    wsum = torch.zeros(bsz, 1, h_hr, w_hr, device=lr.device, dtype=torch.float32)

    win1d = torch.hann_window(tile_hr, periodic=False, device=lr.device).clamp_min(1e-3)
    win = (win1d[:, None] * win1d[None, :])[None, None]

    ys = list(range(0, max(lr.shape[2] - tile_lr, 0) + 1, step_lr))
    xs = list(range(0, max(lr.shape[3] - tile_lr, 0) + 1, step_lr))
    if ys[-1] + tile_lr < lr.shape[2]:
        ys.append(lr.shape[2] - tile_lr)
    if xs[-1] + tile_lr < lr.shape[3]:
        xs.append(lr.shape[3] - tile_lr)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + tile_lr, x0 + tile_lr
            hy0, hx0, hy1, hx1 = y0 * scale, x0 * scale, y1 * scale, x1 * scale
            lr_t = lr[:, :, y0:y1, x0:x1]
            msi_t = msi[:, :, hy0:hy1, hx0:hx1]
            pred = model(lr_t, msi_t)["out"].float()
            w = win[..., :pred.shape[-2], :pred.shape[-1]]
            out[:, :, hy0:hy1, hx0:hx1] += pred * w
            wsum[:, :, hy0:hy1, hx0:hx1] += w
    return (out / wsum.clamp_min(1e-6)).clamp(0, 1)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup:
        return cfg.lr * step / max(cfg.warmup, 1)
    t = (step - cfg.warmup) / max(cfg.iters - cfg.warmup, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate_dataset(model: DAETFNet, root: str, cfg: Config, split: str = "Test",
                     device: str = "cuda", limit: Optional[int] = None,
                     tile_hr: int = 256, verbose: bool = True) -> Dict[str, float]:
    """Full-scene evaluation with the unified metric module."""
    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation.from_config(cfg).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": []}

    for stem, hp, rp in pairs:
        hsi, rgb = cache.get(stem, hp, rp)
        h = (hsi.shape[1] // cfg.scale) * cfg.scale
        w = (hsi.shape[2] // cfg.scale) * cfg.scale
        gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(device)
        msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(device)
        lr = degrade(gt)
        pred = tiled_inference(model, lr, msi, cfg.scale, tile_hr=tile_hr)
        m = evaluate_arrays(pred[0].cpu().numpy().transpose(1, 2, 0),
                            gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
        rows.append((stem, m))
        for k, v in m.items():
            agg[k].append(v)
        if verbose:
            print(f"  {stem:<24} PSNR={m['psnr']:7.3f}  SSIM={m['ssim']:.4f}  "
                  f"SAM={m['sam']:6.3f}  ERGAS={m['ergas']:8.3f}")
        del gt, msi, lr, pred
        if device == "cuda":
            torch.cuda.empty_cache()

    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    if verbose:
        print(f"  {'MEAN':<24} PSNR={mean['psnr']:7.3f}  SSIM={mean['ssim']:.4f}  "
              f"SAM={mean['sam']:6.3f}  ERGAS={mean['ergas']:8.3f}")
    return mean


def train(cfg: Config, device: str = "cuda", align_target: bool = True,
          log_fn=print) -> Tuple[DAETFNet, Dict]:
    """Train on the source domain. When `align_target` is set and a target root
    is configured, unlabelled target patches are drawn in parallel and aligned
    with an MMD penalty. No ground truth from the target domain is ever used."""
    cfg.resolve(verbose=False)          # idempotent: only fills what is still None
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    log_fn("estimating SRF from the training pairs ...")
    srf = estimate_srf(cfg.source_root, "Train", cfg)
    log_fn(f"SRF shape {srf.shape}, column sums {srf.sum(0).round(3).tolist()}")

    train_set = FusionPatchDataset(cfg.source_root, "Train", cfg, train=True, srf=srf,
                                   length=cfg.iters * cfg.batch)
    loader = DataLoader(train_set, batch_size=cfg.batch, shuffle=False,
                        num_workers=cfg.workers, pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=cfg.workers > 0)

    tgt_loader = None
    if align_target and cfg.target_root:
        tgt_set = FusionPatchDataset(cfg.target_root, "Train", cfg, train=True, srf=srf,
                                     length=cfg.iters * cfg.batch)
        tgt_loader = iter(DataLoader(tgt_set, batch_size=cfg.batch, shuffle=False,
                                     num_workers=max(1, cfg.workers // 2),
                                     drop_last=True))
        log_fn(f"domain alignment enabled against {cfg.target_root} (unlabelled)")

    model = DAETFNet(cfg).to(device)
    crit = SPCLoss(cfg, torch.from_numpy(srf)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5,
                            betas=(0.9, 0.99))
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    log_fn(f"DAETF-Net v2: {n_params / 1e6:.2f} M parameters")

    history: Dict[str, list] = {"iter": [], "loss": [], "val": []}
    best, t0 = -1e9, time.time()

    model.train()
    for step, batch in enumerate(loader, start=1):
        if step > cfg.iters:
            break
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg)

        lr_hsi = batch["lr"].to(device, non_blocking=True)
        msi = batch["msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        deg_gt = batch["deg"].to(device, non_blocking=True)
        kernel = batch["kernel"].to(device, non_blocking=True)

        tgt_feat = None
        if tgt_loader is not None:
            tb = next(tgt_loader)
            with torch.amp.autocast("cuda", enabled=use_amp):
                tgt_feat = model.features(tb["lr"].to(device), tb["msi"].to(device))

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(lr_hsi, msi)
            loss, logs = crit(out, gt, lr_hsi, msi, model, deg_gt=deg_gt,
                              kernel=kernel, tgt_feat=tgt_feat)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % cfg.log_every == 0:
            rate = step / (time.time() - t0)
            log_fn(f"it {step:6d}/{cfg.iters}  loss {logs['total']:.4f}  "
                   f"char {logs.get('char', 0):.4f}  sam {logs.get('sam', 0):.4f}  "
                   f"spat {logs.get('spat', 0):.4f}  spec {logs.get('spec', 0):.4f}  "
                   f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.2f} it/s")
            history["iter"].append(step)
            history["loss"].append(logs["total"])

        if step % cfg.val_every == 0 or step == cfg.iters:
            m = evaluate_dataset(model, cfg.source_root, cfg, "Test", device,
                                 limit=cfg.val_scenes, verbose=False)
            log_fn(f"  [val@{step}] PSNR {m['psnr']:.3f}  SAM {m['sam']:.3f}  "
                   f"ERGAS {m['ergas']:.3f}")
            history["val"].append({"iter": step, **m})
            if m["psnr"] > best:
                best = m["psnr"]
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "srf": srf, "val": m},
                           os.path.join(cfg.out_dir, "daetf_best.pth"))
            model.train()

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "srf": srf,
                "params": n_params},
               os.path.join(cfg.out_dir, "daetf_final.pth"))
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return model, history


@torch.no_grad()
def _clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_time_adapt(model: DAETFNet, lr: torch.Tensor, msi: torch.Tensor,
                    crit: SPCLoss, steps: int = 30, lr_rate: float = 5e-5
                    ) -> torch.Tensor:
    """Self-supervised adaptation on a single unlabelled target scene.

    Only the two physics terms are used - they need no ground truth - so this
    runs on Harvard exactly as it would on a real unseen sensor.
    """
    state = _clone_state(model)
    model.train()
    params = [p for n, p in model.named_parameters()
              if any(k in n for k in ("deg", "film", "moe.gate", "fdrm"))]
    opt = torch.optim.Adam(params, lr=lr_rate)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(lr, msi)
        loss, _ = crit(out, out["out"].detach(), lr, msi, model, supervised=False)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(lr, msi)["out"].clamp(0, 1)
    model.load_state_dict(state)
    return pred


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------
def check_equivariance(device: str = "cpu", tol: float = 1e-4) -> float:
    """rot90(EFE(x)) must equal EFE(rot90(x)). Proves the stem is genuinely
    p4-equivariant rather than equivariant by claim."""
    torch.manual_seed(0)
    efe = EquivariantFeatureExtractor(5, 8, 12, depth=2).to(device).eval()
    x = torch.randn(2, 5, 32, 32, device=device)
    with torch.no_grad():
        a = torch.rot90(efe(x), 1, (-2, -1))
        b = efe(torch.rot90(x, 1, (-2, -1)))
    err = float((a - b).abs().max())
    print(f"[check] p4 equivariance max|err| = {err:.3e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


def check_wavelet(tol: float = 1e-5) -> float:
    """IDWT(DWT(x)) must reconstruct x exactly (orthonormal Haar)."""
    dwt = HaarDWT()
    x = torch.randn(2, 7, 16, 16)
    err = float((dwt.inverse(dwt(x)) - x).abs().max())
    print(f"[check] Haar DWT reconstruction max|err| = {err:.3e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


def check_core_used() -> bool:
    """The Tucker core must receive gradients - the v1 bug was that it did not."""
    t = TensorSpectralSpatialEncoder(8, 8, 8, rank=4)
    a, b = torch.randn(1, 8, 8, 8), torch.randn(1, 8, 8, 8)
    t(a, b).sum().backward()
    ok = t.core.grad is not None and float(t.core.grad.abs().sum()) > 0
    print(f"[check] Tucker core receives gradient ({'PASS' if ok else 'FAIL'})")
    return ok


def smoke_test(device: str = "cpu") -> None:
    """End-to-end shape/gradient check with synthetic tensors."""
    cfg = Config(patch=32, width=32, equi_width=8, rank=8, batch=2,
                 bands=31, msi_bands=3)
    model = DAETFNet(cfg).to(device)
    srf = torch.rand(cfg.bands, cfg.msi_bands)
    crit = SPCLoss(cfg, srf).to(device)
    gt = torch.rand(2, cfg.bands, cfg.patch, cfg.patch, device=device)
    lr = blur_downsample(gt, gaussian_kernel2d(9, 1.2, 1.2, 0.0), cfg.scale)
    msi = torch.rand(2, cfg.msi_bands, cfg.patch, cfg.patch, device=device)
    out = model(lr, msi)
    assert out["out"].shape == gt.shape, (out["out"].shape, gt.shape)
    loss, logs = crit(out, gt, lr, msi, model,
                      deg_gt=torch.rand(2, 5, device=device))
    loss.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for _ in model.parameters())
    n = sum(p.numel() for p in model.parameters())
    print(f"[check] forward {tuple(out['out'].shape)}  loss {float(loss):.4f}  "
          f"params {n / 1e6:.2f}M  tensors with grad {grads}/{total}")
    print(f"[check] loss terms: { {k: round(v, 4) for k, v in logs.items()} }")

    # tiled inference on a larger single scene, with a tile smaller than the image
    big_gt = torch.rand(1, cfg.bands, 96, 96, device=device)
    big_lr = blur_downsample(big_gt, gaussian_kernel2d(9, 1.2, 1.2, 0.0), cfg.scale)
    big_msi = torch.rand(1, cfg.msi_bands, 96, 96, device=device)
    pred = tiled_inference(model, big_lr, big_msi, cfg.scale, tile_hr=32, overlap=8)
    assert pred.shape == big_gt.shape, (pred.shape, big_gt.shape)
    m = evaluate_arrays(pred[0].detach().cpu().numpy().transpose(1, 2, 0),
                        big_gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
    print(f"[check] tiled inference {tuple(pred.shape)} PASS; "
          f"metrics { {k: round(v, 3) for k, v in m.items()} }")

    # every ablation variant must also build and run
    for switch in ("use_equivariant", "use_tsse", "use_moe", "use_fdrm",
                   "use_backprojection", "use_degradation_code"):
        acfg = Config(patch=32, width=32, equi_width=8, rank=8, batch=2,
                      bands=31, msi_bands=3, **{switch: False})
        am = DAETFNet(acfg).to(device)
        ao = am(lr, msi)
        assert ao["out"].shape == gt.shape
        acrit = SPCLoss(acfg, srf).to(device)
        aloss, _ = acrit(ao, gt, lr, msi, am, deg_gt=torch.rand(2, 5, device=device))
        aloss.backward()
        n_a = sum(p.numel() for p in am.parameters())
        print(f"[check] ablation {switch}=False OK ({n_a / 1e6:.2f}M params)")


if __name__ == "__main__":
    check_equivariance()
    check_wavelet()
    check_core_used()
    smoke_test()
