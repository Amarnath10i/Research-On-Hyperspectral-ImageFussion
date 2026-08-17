"""Null-space-projected reverse diffusion for HSI-MSI fusion.

CONTRIBUTION
------------
Fusion is solved as posterior sampling on the *null space of the observation
operator*:

    1. the range component D_pinv(X) is computed in closed form from the
       measured LR-HSI and never learned;
    2. a reverse diffusion process generates the null-space component, guided
       by the HR-MSI and a degradation code;
    3. at every reverse step the estimate is re-projected onto the consistent
       set  D_pinv(X) + P_perp(.)  so D(Y_hat) = X holds by construction for
       any score-network behaviour.

This is what distinguishes the method from a diffusion model that generates
freely and hopes the observation is satisfied, and from a discriminative
residual network that merely penalises the mismatch in its loss.  The identity
holds with respect to the operator D actually supplied, whether that is the
known evaluation kernel or an estimate produced by the blind operator encoder
(and optionally refined per scene).

The reverse sampler below is deterministic DDIM (eta=0) with the projection
folded in; a stochastic variant is available by setting ``eta > 0``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .nullspace import (RangeNullProjector, decode_degradation_params,
                        kernel_from_params)
from .score import SpectralScoreNet


class LinearNoiseSchedule:
    """Discrete DDPM-style noise schedule, a_t = cumprod(1 - beta_t)."""

    def __init__(self, num_timesteps: int = 200, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.T = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        a = torch.cumprod(alphas, dim=0)
        # a[0] corresponds to t=0 (clean). Build 0..T indexing.
        self.a = torch.cat([torch.ones(1, dtype=torch.float64), a])
        self.betas = torch.cat([torch.zeros(1), betas])

    def to(self, device, dtype=torch.float32):
        self.a = self.a.to(device=device, dtype=dtype)
        self.betas = self.betas.to(device=device, dtype=dtype)
        return self

    def sqrt_a(self, t: int) -> torch.Tensor:
        return self.a[t].sqrt()

    def sqrt_1ma(self, t: int) -> torch.Tensor:
        return (1 - self.a[t]).sqrt()

    def noise_std(self, t: int) -> float:
        return float((1 - self.a[t]).item())


class DiffusionSampler:
    """Reverse diffusion with per-step null-space projection.

    ``score_net`` maps (y_t, msi, code, t) -> predicted noise.  ``projector``
    implements the consistent set for a given kernel.  Sampling is deterministic
    DDIM by default (``eta = 0``).
    """

    def __init__(self, score_net: SpectralScoreNet,
                 projector: RangeNullProjector,
                 num_timesteps: int = 200, sample_steps: int = 10,
                 eta: float = 0.0, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.score_net = score_net
        self.projector = projector
        self.schedule = LinearNoiseSchedule(num_timesteps, beta_start, beta_end)
        self.sample_steps = sample_steps
        self.eta = eta

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _consistent(self, x0: torch.Tensor, range_part: torch.Tensor,
                    kernel: Optional[torch.Tensor]) -> torch.Tensor:
        """D_pinv(X) + P_perp(x0) - the algebraic data-consistency step."""
        hw = (x0.shape[-2], x0.shape[-1])
        proj = self.projector.project_null(x0, kernel)
        return range_part + proj

    def _steps(self, device) -> list:
        T = self.schedule.T
        steps = torch.linspace(T, 1, self.sample_steps).long().tolist()
        return steps

    @torch.no_grad()
    def sample(self, lr_hsi: torch.Tensor, msi: torch.Tensor,
               kernel: Optional[torch.Tensor] = None,
               code: Optional[torch.Tensor] = None,
               return_x0_history: bool = False):
        """Draw a fused HR-HSI cube.

        Args:
            lr_hsi: [B, C, h, w] the measured LR hyperspectral observation.
            msi:    [B, m, H, W] the HR multispectral guide.
            kernel: [B, k, k] or None for the projector default.
            code:   [B, code_dim] degradation code or None.
        Returns:
            [B, C, H, W] fused cube with D(out) == lr_hsi to solver tolerance,
            plus the x0 history if requested.
        """
        self.schedule = self.schedule.to(lr_hsi.device)
        self.score_net.eval()
        b, c, h, w = msi.shape[0], lr_hsi.shape[1], msi.shape[2], msi.shape[3]
        range_part = self.projector.pinv(lr_hsi, (h, w), kernel)

        y = torch.randn(b, c, h, w, device=lr_hsi.device, dtype=lr_hsi.dtype)
        steps = self._steps(lr_hsi.device)
        hist = []

        for k, t in enumerate(steps):
            a_t = self.schedule.sqrt_a(t)
            sq1_t = self.schedule.sqrt_1ma(t)
            tt = torch.full((b,), t, device=lr_hsi.device, dtype=torch.long)
            eps = self.score_net(y, msi, code, tt)
            x0 = (y - sq1_t * eps) / a_t
            x0 = self._consistent(x0, range_part, kernel)
            # Clamping is needed for a valid image, but the clamp itself would
            # break D(x0) = X, so project again after the clamp to restore the
            # identity exactly.
            x0 = self._consistent(x0.clamp(0, 1), range_part, kernel)
            if return_x0_history:
                hist.append(x0)

            if k == len(steps) - 1:
                y = x0
                break

            tp = steps[k + 1]
            a_p = self.schedule.sqrt_a(tp)
            eps_dir = (y - a_t * x0) / sq1_t.clamp_min(1e-8)
            if self.eta > 0:
                sigma = self.eta * math.sqrt(
                    float((1 - self.schedule.a[tp]) / (1 - self.schedule.a[t]))
                    * float((1 - self.schedule.a[t] / self.schedule.a[tp])))
                z = torch.randn_like(y)
                y = a_p * x0 + math.sqrt(float(1 - self.schedule.a[tp]) - sigma ** 2) \
                    * eps_dir + sigma * z
            else:
                y = a_p * x0 + self.schedule.sqrt_1ma(tp) * eps_dir

        return (y, hist) if return_x0_history else y


# ------------------------------------------------------------- refinement
def refine_operator_kernel(lr_hsi: torch.Tensor, msi: torch.Tensor,
                           y_sample: torch.Tensor, init_params: torch.Tensor,
                           srf: Optional[torch.Tensor], ksize: int = 9,
                           steps: int = 3, lr_rate: float = 1e-2,
                           w_spec: float = 1.0) -> torch.Tensor:
    """Refine a blind operator estimate against the current sample.

    The degradation encoder gives an initial kernel; this performs a few
    gradient steps on the *physics* of the current sample,

        min_p  || D_k(y) - X ||^2  +  w_spec || S(y) - M ||^2,

    so the operator used for the null-space projection and the sample it acts
    on are mutually consistent.  This is the blind-generalisation step: the
    target scene's true blur/response is never known, only its observations.
    """
    b = lr_hsi.shape[0]
    p = init_params.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr_rate)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        kernel = kernel_from_params(decode_degradation_params(p), ksize)
        from .nullspace import DegradationOperator
        D = DegradationOperator(int(msi.shape[-2] / lr_hsi.shape[-2]), ksize)
        loss = F.mse_loss(D(y_sample, kernel), lr_hsi)
        if srf is not None and w_spec > 0:
            sr = srf.to(y_sample.dtype).t().reshape(
                srf.shape[1], srf.shape[0], 1, 1)
            loss = loss + w_spec * F.mse_loss(F.conv2d(y_sample, sr), msi)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return kernel_from_params(decode_degradation_params(p), ksize)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = SpectralScoreNet(bands=5, msi_bands=3, ch=16, ch_mult=(1, 2))
    P = RangeNullProjector(scale=4, cg_steps=8)
    sam = DiffusionSampler(net, P, num_timesteps=50, sample_steps=4)
    lr = torch.rand(1, 5, 16, 16)
    msi = torch.rand(1, 3, 64, 64)
    code = torch.rand(1, 64)
    out = sam.sample(lr, msi, code=code)
    print("sampler shape", tuple(out.shape), "range", out.min().item(), out.max().item())