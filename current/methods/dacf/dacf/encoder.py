"""Degradation encoder: estimates degradation characteristics from observations."""

from __future__ import annotations

import torch
import torch.nn as nn


class DegradationEncoder(nn.Module):
    """Encodes LR-HSI + MSI into a degradation code.

    The encoder learns to extract blur, noise, and SRF characteristics
    from the observed data pair. This code conditions the flow network
    to adapt its behavior to the specific degradation.

    Architecture:
        LR-HSI path: Conv → Pool → Conv → Pool → Flatten → Linear
        MSI path:    Conv → Pool → Flatten → Linear
        Fusion:      concat → MLP → code_dim
    """

    def __init__(self, bands: int, msi_bands: int, code_dim: int = 64,
                 hidden: int = 32):
        super().__init__()
        self.bands = bands
        self.msi_bands = msi_bands
        self.code_dim = code_dim

        # LR-HSI encoder: handles variable band count
        self.enc_lr = nn.Sequential(
            nn.Conv2d(bands, hidden, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # MSI encoder: lightweight
        self.enc_ms = nn.Sequential(
            nn.Conv2d(msi_bands, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Fusion MLP
        self.head = nn.Sequential(
            nn.Linear(hidden + 16, 32),
            nn.ReLU(),
            nn.Linear(32, code_dim),
        )

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        """Encode observations into degradation code.

        Args:
            lr_hsi: [B, bands, H_lr, W_lr] low-resolution HSI
            msi: [B, msi_bands, H, W] multispectral image

        Returns:
            code: [B, code_dim] degradation embedding
        """
        f_lr = self.enc_lr(lr_hsi)
        f_ms = self.enc_ms(msi)
        return self.head(torch.cat([f_lr, f_ms], dim=1))
