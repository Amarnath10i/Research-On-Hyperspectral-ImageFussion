"""P1 Ambiguity Auditor: applies H/U decomposition to any named fusion method.

Given a trained (or training-free) HSI-MSI fusion model, the auditor:
1. Feeds LR-HSI + HR-MSI through the model →得到 O_r (observed component)
2. Decomposes O_r into observable (E_obs) and ambiguous (E_null) parts
3. Computes per-pixel uncertainty and hallucination maps
4. Reports the ambiguity score H

This is the "ambiguity auditing" contribution of P1: we don't build a new
fusion model — we show where existing models hallucinate.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AmbiguityAuditor(nn.Module):
    """Wraps any (lr_hsi, msi) → {"out": ...} model and audits its ambiguity.

    Usage:
        base_model = ...  # any fusion model (trained or training-free)
        auditor = AmbiguityAuditor(base_model, srf, obs_mask)

        with torch.no_grad():
            result = auditor(lr_hsi, msi)

        result["obs"]       # E_obs: observable component
        result["ambig"]     # E_null: ambiguous (hallucinated) component
        result["H"]         # scalar ambiguity score
        result["uncert"]    # per-pixel uncertainty map
        result["halluc"]    # per-pixel hallucination map
    """

    def __init__(self, model: nn.Module, srf: torch.Tensor,
                 obs_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        B = srf.shape[0]
        # obs_mask: (B,) binary mask, 1 = observed band, 0 = unobserved
        if obs_mask is None:
            obs_mask = torch.ones(B)
        self.register_buffer("obs_mask", obs_mask.float())
        self.register_buffer("srf", srf.float())

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        # Step 1: get the model's output
        with torch.no_grad():
            out = self.model(lr_hsi, msi)["out"]  # (B, H, W)

        B, H, W = out.shape

        # Step 2: observable subspace projection
        # E_obs = U_obs @ U_obs^T @ out
        # where U_obs spans the spectral directions the sensor can observe
        # Simplification: for rank-r approximation, E_obs is the rank-r
        # part; E_null = out - E_obs
        U_r = self._estimate_basis(out)  # (B, r)

        flat = out.reshape(B, -1)  # (B, N)
        E_obs_flat = U_r @ (U_r.t() @ flat)  # (B, N)
        E_null_flat = flat - E_obs_flat

        E_obs = E_obs_flat.reshape(B, H, W)
        E_null = E_null_flat.reshape(B, H, W)

        # Step 3: per-pixel ambiguity metrics
        obs_energy = E_obs_flat.norm(dim=0)  # (N,)
        null_energy = E_null_flat.norm(dim=0)  # (N,)
        total = obs_energy + null_energy + 1e-8

        # H: normalized ambiguity score per pixel
        H_map = (null_energy / total).reshape(H, W)

        # uncertainty: energy of the null component (high = uncertain)
        uncert = null_energy.reshape(H, W)

        # hallucination: cosine similarity between E_null and E_obs
        # high value means the model is inventing spectra correlated with
        # the observed part (hallucination, not just noise)
        cos_sim = torch.sum(E_obs_flat * E_null_flat, dim=0) / (
            obs_energy * null_energy + 1e-8)
        halluc = (cos_sim.abs()).reshape(H, W)

        # Step 4: scalar ambiguity score
        H_scalar = E_null_flat.norm() / (flat.norm() + 1e-8)

        return {
            "out": out,
            "obs": E_obs,
            "ambig": E_null,
            "H": H_scalar,
            "H_map": H_map,
            "uncert": uncert,
            "halluc": halluc,
            "null_energy": null_energy,
            "obs_energy": obs_energy,
        }

    def _estimate_basis(self, out: torch.Tensor) -> torch.Tensor:
        """Estimate spectral basis from the model output.

        In production, this uses the LR-HSI spectral basis.  For auditing
        without ground truth, we estimate from the output itself.
        """
        B, H, W = out.shape
        flat = out.reshape(B, -1)  # (B, N)
        # take the top-r singular vectors
        r = min(B, 10)  # safe default; in practice r̂_id is used
        U, S, _ = torch.linalg.svd(flat, full_matrices=False)
        return U[:, :r]  # (B, r)


def audit_model(model: nn.Module, lr_hsi: torch.Tensor, msi: torch.Tensor,
                srf: torch.Tensor, obs_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
    """Convenience function: wrap a model and audit one scene."""
    auditor = AmbiguityAuditor(model, srf, obs_mask)
    with torch.no_grad():
        return auditor(lr_hsi, msi)


def batch_audit(model: nn.Module, scenes: list, srf: torch.Tensor,
                obs_mask: Optional[torch.Tensor] = None,
                device: str = "cuda") -> Dict[str, list]:
    """Audit multiple scenes, collect statistics."""
    auditor = AmbiguityAuditor(model, srf, obs_mask).to(device)
    stats = {"H": [], "mean_uncert": [], "mean_halluc": []}

    auditor.eval()
    for lr_hsi, msi in scenes:
        lr_hsi = lr_hsi.to(device)
        msi = msi.to(device)
        result = auditor(lr_hsi, msi)
        stats["H"].append(result["H"].item())
        stats["mean_uncert"].append(result["uncert"].mean().item())
        stats["mean_halluc"].append(result["halluc"].mean().item())

    return stats