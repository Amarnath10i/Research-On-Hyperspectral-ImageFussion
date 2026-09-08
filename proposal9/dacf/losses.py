"""DACF loss: flow matching + cycle consistency + SRF + spectral fidelity."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(x: torch.Tensor, y: torch.Tensor, alpha: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((x - y) ** 2 + alpha ** 2))


def sam_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral Angle Mapper loss."""
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    t = target.reshape(target.shape[0], target.shape[1], -1)
    cos = F.cosine_similarity(p, t, dim=1)
    return torch.mean(torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))


class DACFLoss:
    """Combined loss for DACF training.

    Phase 1 (flow matching):
        L = λ_flow * MSE(pred_v, target_v)
          + λ_cycle * Charbonnier(D(fused), LR-HSI)
          + λ_srf * Charbonnier(S(fused), MSI)
          + λ_spectral * SAM(fused, GT)

    Phase 2 (self-supervised adaptation):
        L = λ_cycle * Charbonnier(D(fused), LR-HSI)
          + λ_srf * Charbonnier(S(fused), MSI)
    """

    def __init__(self, lambda_flow: float = 1.0, lambda_cycle: float = 0.5,
                 lambda_srf: float = 0.3, lambda_spectral: float = 0.1):
        self.lambda_flow = lambda_flow
        self.lambda_cycle = lambda_cycle
        self.lambda_srf = lambda_srf
        self.lambda_spectral = lambda_spectral

    def phase1(self, pred: dict, gt: torch.Tensor, lr: torch.Tensor,
               msi: torch.Tensor, fused: torch.Tensor, srf: torch.Tensor,
               projector) -> torch.Tensor:
        """Phase 1: flow matching + consistency + spectral."""
        # Flow matching loss
        loss_flow = F.mse_loss(pred["pred_v"], pred["target_v"])

        # Cycle consistency: D(fused) ≈ LR-HSI
        d_fused = projector.D(fused)
        loss_cycle = charbonnier(d_fused, lr)

        # SRF consistency: S(fused) ≈ MSI
        s_fused = torch.einsum("mb,bhw->mhw", srf.to(fused.device), fused)
        loss_srf = charbonnier(s_fused, msi)

        # Spectral fidelity
        loss_spectral = sam_loss(fused, gt)

        return (self.lambda_flow * loss_flow
                + self.lambda_cycle * loss_cycle
                + self.lambda_srf * loss_srf
                + self.lambda_spectral * loss_spectral)

    def phase2(self, fused: torch.Tensor, lr: torch.Tensor,
               msi: torch.Tensor, srf: torch.Tensor,
               projector) -> torch.Tensor:
        """Phase 2: self-supervised cycle consistency (no GT needed)."""
        d_fused = projector.D(fused)
        loss_cycle = charbonnier(d_fused, lr)

        s_fused = torch.einsum("mb,bhw->mhw", srf.to(fused.device), fused)
        loss_srf = charbonnier(s_fused, msi)

        return self.lambda_cycle * loss_cycle + self.lambda_srf * loss_srf
