"""Continuous scene field F(x,y,lambda) for P3 (sensor-independent fusion).

The field is a parametric function of continuous spatial (x,y) and spectral
(lambda in [0,1]) coordinates:

    F(x,y,lam) = sum_{b,k,l} Z[b,k,l] * psi_b(lam) * cos(2 pi kx x) cos(2 pi ky y)

with psi_b = Gaussian spectral bumps and cosines as the spatial modes.  Any
sensor is just an operator O_s applied to this ONE field, so a field fitted on
sensors A,B can be rendered for an unseen sensor C without re-fitting: that is
the P3 zero-shot claim this package scaffolds.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SceneField(nn.Module):
    def __init__(self, bands: int = 6, modes: int = 4, bump_w: float = 0.03,
                 seed: int = 0):
        super().__init__()
        self.bands, self.modes = bands, modes
        centers = torch.linspace(0.1, 0.9, bands)
        self.register_buffer("lam_centers", centers)
        self.register_buffer("bump_w", torch.tensor(bump_w))
        k = torch.arange(modes, dtype=torch.float32)
        kx, ky = torch.meshgrid(k, k, indexing="ij")
        self.register_buffer("mode_freq", torch.stack([kx, ky], dim=0))
        g = torch.Generator().manual_seed(seed)
        self.Z = nn.Parameter(torch.randn(bands, modes, modes,
                                          generator=g) * 0.2)

    def spectral(self, lam: torch.Tensor) -> torch.Tensor:
        """Gaussian bumps over lambda: (B, N_lam)."""
        diff = lam[None, :] - self.lam_centers[:, None]
        w = torch.exp(-0.5 * (diff / self.bump_w) ** 2)
        return w

    def modes2d(self, hw: int) -> torch.Tensor:
        """Spatial mode grid: (K, K, H, W) in [0,1]^2."""
        g = (torch.arange(hw, dtype=torch.float32) + 0.5) / hw
        x, y = torch.meshgrid(g, g, indexing="ij")
        kx, ky = self.mode_freq[0, :, :, None, None], self.mode_freq[1, :, :, None, None]
        return torch.cos(2 * torch.pi * kx * x[None, None]) \
            * torch.cos(2 * torch.pi * ky * y[None, None])

    def render(self, lam: torch.Tensor, hw: int) -> torch.Tensor:
        """F on a dense lambda grid: (N_lam, H, W)."""
        spec = self.spectral(lam)                      # (B, N_lam)
        mode = self.modes2d(hw)                        # (K, K, H, W)
        return torch.einsum("bkl,bn,klhw->nhw",
                            self.Z, spec, mode.to(self.Z.device))