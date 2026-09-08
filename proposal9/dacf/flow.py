"""Conditional normalizing flow for HSI fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: output = gamma * input + beta."""

    def __init__(self, in_ch: int, code_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(code_dim, in_ch)
        self.beta_proj = nn.Linear(code_dim, in_ch)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(code).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_proj(code).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class CouplingLayer(nn.Module):
    """Affine coupling layer with FiLM conditioning."""

    def __init__(self, channels: int, code_dim: int, hidden: int = 64):
        super().__init__()
        self.split = channels // 2
        self.rest = channels - self.split

        self.net = nn.Sequential(
            nn.Conv2d(self.split, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, self.rest * 2, 3, padding=1),
        )
        self.film = FiLMBlock(hidden, code_dim)

    def forward(self, x: torch.Tensor, code: torch.Tensor,
                reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        x1, x2 = x[:, :self.split], x[:, self.split:]

        h = self.net[:2](x1)
        h = self.film(h, code)
        h = self.net[2:](h)
        log_scale, shift = h[:, :self.rest], h[:, self.rest:]

        if not reverse:
            x2 = x2 * torch.exp(log_scale.tanh()) + shift
            log_det = log_scale.tanh().flatten(1).sum(1)
        else:
            x2 = (x2 - shift) * torch.exp(-log_scale.tanh())
            log_det = torch.zeros(B, device=x.device)

        return torch.cat([x1, x2], dim=1), log_det


class ConditionalFlow(nn.Module):
    """Conditional normalizing flow with FiLM-modulated coupling layers."""

    def __init__(self, bands: int, code_dim: int = 64, hidden: int = 64,
                 n_layers: int = 8):
        super().__init__()
        self.bands = bands
        self.layers = nn.ModuleList(
            [CouplingLayer(bands, code_dim, hidden) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, code: torch.Tensor,
                reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        if not reverse:
            for layer in self.layers:
                x, ld = layer(x, code, reverse=False)
                log_det_total = log_det_total + ld
        else:
            for layer in reversed(self.layers):
                x, ld = layer(x, code, reverse=True)
                log_det_total = log_det_total + ld
        return x, log_det_total

    def sample(self, z: torch.Tensor, code: torch.Tensor,
               steps: int = 4) -> torch.Tensor:
        """Rectified-flow-style sampling from base to target."""
        y = z.clone()
        for k in range(steps):
            t = (k + 0.5) / steps
            y_t = (1 - t) * z + t * y
            y_next, _ = self.forward(y_t, code, reverse=False)
            y = (1 - t) * y + t * y_next
        return y
