"""ASON model: VelocityNet + DegradationCode + ASONNet with null-space projection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Null-space projector (from SpectralFlow / proposal5)
# ---------------------------------------------------------------------------

class RangeNullProjector(nn.Module):
    """Projects signal onto null-space of downsampling + SRF degradation."""

    def __init__(self, scale: int = 4, bands: int = 128, kernel_size: int = 9,
                 sigma: float = 1.2):
        super().__init__()
        self.scale = scale
        self.bands = bands
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _make_gaussian_kernel(self, k: int, sx: float, sy: float) -> torch.Tensor:
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 / (2 * sx**2) + yy**2 / (2 * sy**2)))
        return kernel / kernel.sum()

    def D(self, x: torch.Tensor) -> torch.Tensor:
        """Degradation: blur + downsample."""
        B, C, H, W = x.shape
        k = self._make_gaussian_kernel(self.kernel_size, self.sigma, self.sigma).to(x.device, x.dtype)
        k = k.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
        pad = self.kernel_size // 2
        blurred = F.conv2d(x.reshape(B * C, 1, H, W), k.unsqueeze(1), padding=pad, groups=C)
        blurred = blurred.reshape(B, C, H, W)
        return blurred[:, :, ::self.scale, ::self.scale]

    def project_null(self, v: torch.Tensor, kernel: torch.Tensor | None = None) -> torch.Tensor:
        """Project velocity onto consistent set: D(projected) = D(original)."""
        B, C, H, W = v.shape
        dv = self.D(v)
        if kernel is not None:
            k = kernel
        else:
            k = self._make_gaussian_kernel(self.kernel_size, self.sigma, self.sigma).to(v.device, v.dtype)
        k = k.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
        pad = self.kernel_size // 2
        upsampled = F.interpolate(dv, size=(H, W), mode="nearest")
        correction = F.conv2d(upsampled.reshape(B * C, 1, H, W), k.unsqueeze(1),
                              padding=pad, groups=C).reshape(B, C, H, W)
        return v - correction


# ---------------------------------------------------------------------------
# Velocity network with FiLM conditioning
# ---------------------------------------------------------------------------

class VelocityNet(nn.Module):
    """Velocity field v(y, m, t, code) for rectified flow."""

    def __init__(self, bands: int, msi_bands: int, code_dim: int = 64,
                 hidden: int = 24, blocks: int = 3):
        super().__init__()
        self.proj_m = nn.Conv2d(msi_bands, bands, 1)
        self.code_proj = nn.Linear(code_dim, hidden * 2)
        ch = 2 * bands + 1
        body = [nn.Conv2d(ch, hidden, 3, padding=1)]
        for _ in range(blocks - 1):
            body += [nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1)]
        body += [nn.SiLU(), nn.Conv2d(hidden, bands, 3, padding=1)]
        self.net = nn.Sequential(*body)
        self.film_gamma = nn.Linear(hidden * 2, hidden)
        self.film_beta = nn.Linear(hidden * 2, hidden)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, y: torch.Tensor, m: torch.Tensor, t: float,
                code: torch.Tensor) -> torch.Tensor:
        B, C, H, W = y.shape
        if not torch.is_tensor(t):
            t = torch.full((B,), float(t), device=y.device, dtype=y.dtype)
        tch = t.reshape(B, 1, 1, 1).expand(B, 1, H, W)
        inp = torch.cat([y, self.proj_m(m), tch], dim=1)
        code_feat = self.code_proj(code)
        gamma = self.film_gamma(code_feat).unsqueeze(-1).unsqueeze(-1)
        beta = self.film_beta(code_feat).unsqueeze(-1).unsqueeze(-1)
        x = self.net[0](inp)
        x = self.net[1](x)
        x = x * (1 + gamma) + beta
        for layer in self.net[2:]:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Degradation encoder
# ---------------------------------------------------------------------------

class DegradationCode(nn.Module):
    """Encodes LR-HSI + MSI into a degradation embedding."""

    def __init__(self, bands: int, msi_bands: int, code_dim: int = 64):
        super().__init__()
        self.enc_lr = nn.Sequential(
            nn.Conv2d(bands, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.enc_ms = nn.Sequential(
            nn.Conv2d(msi_bands, 16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, code_dim))

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.enc_lr(lr_hsi), self.enc_ms(msi)], 1))


# ---------------------------------------------------------------------------
# ASON network
# ---------------------------------------------------------------------------

class ASONNet(nn.Module):
    """Adaptive Spectral Operator Network.

    Composes:
      - DegradationCode: encodes LR-HSI + MSI into context
      - VelocityNet: predicts rectified flow velocity
      - RangeNullProjector: enforces consistency D(y) = X
    """

    def __init__(self, bands: int, msi_bands: int, scale: int = 4,
                 hidden: int = 24, n_blocks: int = 3, sample_steps: int = 4,
                 cg_steps: int = 8, eval_sigma: float = 1.2):
        super().__init__()
        self.bands = bands
        self.msi_bands = msi_bands
        self.scale = scale
        self.sample_steps = sample_steps
        self.cg_steps = cg_steps

        self.deg_code = DegradationCode(bands, msi_bands)
        self.velocity = VelocityNet(bands, msi_bands, hidden=hidden, blocks=n_blocks)
        self.projector = RangeNullProjector(scale, bands, sigma=eval_sigma)

        self.srf = None

    def set_srf(self, srf: torch.Tensor):
        """Set the spectral response function [msi_bands, bands]."""
        self.srf = srf

    def sample(self, x_lr: torch.Tensor, m: torch.Tensor,
               kernel: torch.Tensor | None = None,
               steps: int | None = None,
               code: torch.Tensor | None = None) -> torch.Tensor:
        """Rectified flow sampling with null-space projection."""
        if steps is None:
            steps = self.sample_steps
        B = x_lr.shape[0]
        y = x_lr.clone()
        if code is None:
            code = self.deg_code(x_lr, m)
        for k in range(steps):
            t = (k + 0.5) / steps
            v = self.velocity(y, m, t, code)
            v = self.projector.project_null(v, kernel)
            y = y + v / steps
        return y

    def training_step(self, lr_hsi: torch.Tensor, msi: torch.Tensor,
                      gt: torch.Tensor, t: torch.Tensor,
                      kernel: torch.Tensor | None = None):
        """Training step returning velocity target and prediction."""
        code = self.deg_code(lr_hsi, msi)
        y = (1 - t.view(-1, 1, 1, 1)) * lr_hsi + t.view(-1, 1, 1, 1) * gt
        target = gt - lr_hsi
        pred_v = self.velocity(y, msi, t, code)
        return {"velocity": pred_v, "target": target, "code": code}

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor,
                kernel: torch.Tensor | None = None):
        hw = (msi.shape[-2], msi.shape[-1])
        code = self.deg_code(lr_hsi, msi)
        out = self.sample(lr_hsi, msi, kernel=kernel, code=code)
        return {"out": out}
