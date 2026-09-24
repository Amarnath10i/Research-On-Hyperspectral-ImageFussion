"""ASON loss: flow matching + physics consistency + spectral penalty."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, y: torch.Tensor, alpha: float = 1e-3) -> torch.Tensor:
    """Charbonnier loss (smooth L1 variant)."""
    return torch.mean(torch.sqrt((x - y) ** 2 + alpha ** 2))


def sam_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral Angle Mapper loss."""
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    t = target.reshape(target.shape[0], target.shape[1], -1)
    cos = F.cosine_similarity(p, t, dim=1)
    return torch.mean(torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))


class ASONLoss:
    """Combined ASON loss.

    Components:
      1. Flow matching: MSE between predicted and target velocity
      2. L1 reconstruction: charbonnier between sampled output and GT
      3. Physics consistency: D(sampled) matches LR-HSI
      4. SRF consistency: S(sampled) matches MSI
      5. Spectral penalty: SAM between sampled and GT
    """

    def __init__(self, lambda_velocity: float = 1.0, lambda_l1: float = 0.05,
                 lambda_consistency: float = 0.5, lambda_spectral: float = 0.1):
        self.lambda_velocity = lambda_velocity
        self.lambda_l1 = lambda_l1
        self.lambda_consistency = lambda_consistency
        self.lambda_spectral = lambda_spectral

    def __call__(self, pred: dict, gt: torch.Tensor, lr: torch.Tensor,
                 msi: torch.Tensor, srf: torch.Tensor | None = None,
                 sampled: torch.Tensor | None = None) -> torch.Tensor:
        loss_v = F.mse_loss(pred["velocity"], pred["target"])
        total = self.lambda_velocity * loss_v

        if sampled is not None:
            total = total + self.lambda_l1 * charbonnier(sampled, gt)
            if srf is not None:
                msi_pred = torch.einsum("mb,bhw->mhw", srf.to(sampled.device), sampled)
                total = total + self.lambda_consistency * charbonnier(msi_pred, msi)
            total = total + self.lambda_spectral * sam_loss(sampled, gt)

        return total
