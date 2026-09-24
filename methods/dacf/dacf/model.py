"""DACF model: DegradationEncoder + ConditionalFlow + NullProjector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import DegradationEncoder
from .flow import ConditionalFlow


class NullProjector(nn.Module):
    """Projects signal to maintain D(y) = X consistency.

    Simplified: applies blur+downsample to velocity, computes correction.
    """

    def __init__(self, scale: int = 4, kernel_size: int = 9, sigma: float = 1.2):
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ax = torch.arange(self.kernel_size, device=device, dtype=dtype) - self.kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        k = torch.exp(-(xx**2 / (2 * self.sigma**2) + yy**2 / (2 * self.sigma**2)))
        return k / k.sum()

    def D(self, x: torch.Tensor) -> torch.Tensor:
        """Degradation: blur + downsample."""
        B, C, H, W = x.shape
        k = self._kernel(x.device, x.dtype).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
        pad = self.kernel_size // 2
        blurred = F.conv2d(x.reshape(B * C, 1, H, W), k.unsqueeze(1), padding=pad, groups=C)
        return blurred.reshape(B, C, H, W)[:, :, ::self.scale, ::self.scale]

    def project(self, v: torch.Tensor) -> torch.Tensor:
        """Project velocity for consistency."""
        B, C, H, W = v.shape
        dv = self.D(v)
        k = self._kernel(v.device, v.dtype).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
        pad = self.kernel_size // 2
        upsampled = F.interpolate(dv, size=(H, W), mode="nearest")
        correction = F.conv2d(
            upsampled.reshape(B * C, 1, H, W), k.unsqueeze(1), padding=pad, groups=C
        ).reshape(B, C, H, W)
        return v - correction


class DACFNet(nn.Module):
    """Degradation-Adaptive Conditional Flow network.

    Pipeline:
        1. DegradationEncoder: LR-HSI + MSI → code
        2. ConditionalFlow: LR-HSI + code → HR-HSI (via flow matching)
        3. NullProjector: enforce D(output) = LR-HSI

    Works for any band count (PaviaU=103, Chikusei=128, etc.)
    """

    def __init__(self, bands: int, msi_bands: int = 3, scale: int = 4,
                 enc_hidden: int = 32, code_dim: int = 64,
                 flow_hidden: int = 64, flow_layers: int = 8,
                 flow_steps: int = 4):
        super().__init__()
        self.bands = bands
        self.msi_bands = msi_bands
        self.scale = scale
        self.flow_steps = flow_steps

        self.encoder = DegradationEncoder(bands, msi_bands, code_dim, enc_hidden)
        self.flow = ConditionalFlow(bands, code_dim, flow_hidden, flow_layers)
        self.projector = NullProjector(scale)
        self.srf = None

    def set_srf(self, srf: torch.Tensor):
        self.srf = srf

    def encode(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        """Get degradation code from observations."""
        return self.encoder(lr_hsi, msi)

    def forward_flow(self, lr_hsi: torch.Tensor, code: torch.Tensor,
                     reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Run flow transform."""
        return self.flow(lr_hsi, code, reverse=reverse)

    def sample(self, lr_hsi: torch.Tensor, msi: torch.Tensor,
               steps: int | None = None) -> torch.Tensor:
        """Generate fused HR-HSI from observations.

        Args:
            lr_hsi: [B, bands, H_lr, W_lr]
            msi: [B, msi_bands, H, W]
            steps: ODE steps (default: self.flow_steps)

        Returns:
            hr_hsi: [B, bands, H, W]
        """
        if steps is None:
            steps = self.flow_steps
        code = self.encode(lr_hsi, msi)
        # Use LR-HSI as base, flow transforms to HR
        hr_hsi = self.flow.sample(lr_hsi, code, steps=steps)
        # Apply null-space projection for consistency
        hr_hsi = self.projector.project(hr_hsi - lr_hsi) + lr_hsi
        return hr_hsi

    def training_step(self, lr_hsi: torch.Tensor, msi: torch.Tensor,
                      gt: torch.Tensor, t: torch.Tensor):
        """Training step for flow matching.

        Args:
            lr_hsi: [B, bands, H_lr, W_lr]
            msi: [B, msi_bands, H, W]
            gt: [B, bands, H, W] (ground truth HR-HSI)
            t: [B] time steps for flow matching

        Returns:
            dict with pred_v, target_v, code
        """
        code = self.encode(lr_hsi, msi)
        # Interpolate between LR and GT
        t_expand = t.view(-1, 1, 1, 1)
        y_t = (1 - t_expand) * lr_hsi + t_expand * gt
        # Target velocity
        target_v = gt - lr_hsi
        # Predicted velocity
        pred_v, _ = self.flow.forward(y_t, code, reverse=False)
        return {"pred_v": pred_v, "target_v": target_v, "code": code}

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor):
        """Full forward pass."""
        out = self.sample(lr_hsi, msi)
        return {"out": out}
