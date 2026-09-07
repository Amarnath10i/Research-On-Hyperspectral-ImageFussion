"""One-step (and few-step) null-space-consistent sampling.

CONTRIBUTION (Proposal 6)
-------------------------
SpectralFlow (P5) guarantees D(Y_hat) = X by projecting at every reverse step,
but pays for it with 10-50 score forward passes per tile.  ConsistentFlow
removes the sampling loop: a **consistency map** f_phi(y_t, t, M, d) is
distilled so that a single evaluation maps a noise sample straight to the
clean endpoint.  The null-space projection is then applied once:

    Y_hat = D_pinv(X) + P_perp( f_phi(y_T, T, M, d) )

so the algebraic guarantee survives at ~1x regressor cost.

The map is the same U-Net architecture as P5's score network, re-read as a
denoiser (x0 = (y_t - sqrt(1-a_t) eps)/sqrt(a_t)), which is why this proposal
explicitly reuses the P5 prior: the contribution is the distillation and the
one-step sampler, not a new generative architecture.

Multi-step sampling is the standard stochastic consistency sampling (re-noise
between steps) and exists only for the speed/quality trade-off study.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from spectralflow.sampler import LinearNoiseSchedule
from spectralflow.nullspace import RangeNullProjector


class ConsistencySampler:
    """One-step (or few-step) consistency sampling with null-space projection.

    ``consist_net`` maps (y_t, msi, code, t) -> predicted noise, from which the
    clean estimate is recovered by the schedule.  ``projector`` implements the
    consistent set D_pinv(X) + P_perp(.) for a supplied kernel.
    """

    def __init__(self, consist_net: torch.nn.Module,
                 projector: RangeNullProjector,
                 num_timesteps: int = 200, sample_steps: int = 1,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 use_projection: bool = True):
        self.consist_net = consist_net
        self.projector = projector
        self.schedule = LinearNoiseSchedule(num_timesteps, beta_start, beta_end)
        self.sample_steps = sample_steps
        self.use_projection = use_projection

    def to_x0(self, y_t: torch.Tensor, eps: torch.Tensor,
              t: torch.Tensor) -> torch.Tensor:
        """Clean-image estimate from the predicted noise at level t."""
        a = self.schedule.sqrt_a(int(t[0]))
        sq1 = self.schedule.sqrt_1ma(int(t[0]))
        return (y_t - sq1 * eps) / a

    def _steps(self, device) -> list:
        if self.sample_steps <= 1:
            return [self.schedule.T]
        return torch.linspace(self.schedule.T, 1, self.sample_steps).long().tolist()

    def _consistent(self, x0: torch.Tensor, range_part: torch.Tensor,
                    kernel: Optional[torch.Tensor]) -> torch.Tensor:
        x0 = x0.clamp(0, 1)
        return range_part + self.projector.project_null(x0, kernel)

    @torch.no_grad()
    def sample(self, lr_hsi: torch.Tensor, msi: torch.Tensor,
               kernel: Optional[torch.Tensor] = None,
               code: Optional[torch.Tensor] = None):
        """Draw a fused HR-HSI cube in ``sample_steps`` forward passes.

        Args:
            lr_hsi: [B, C, h, w] measured LR hyperspectral observation.
            msi:    [B, m, H, W] HR multispectral guide.
            kernel: [B, k, k] or None for the projector default.
            code:   [B, code_dim] degradation code or None.
        Returns:
            [B, C, H, W] fused cube with D(out) == lr_hsi to solver tolerance.
        """
        self.schedule = self.schedule.to(lr_hsi.device)
        self.consist_net.eval()
        b, c, h, w = (msi.shape[0], lr_hsi.shape[1], msi.shape[2], msi.shape[3])
        range_part = (self.projector.pinv(lr_hsi, (h, w), kernel)
                      if self.use_projection else None)

        y = torch.randn(b, c, h, w, device=lr_hsi.device, dtype=lr_hsi.dtype)
        steps = self._steps(lr_hsi.device)
        x0 = None
        for k, t in enumerate(steps):
            tt = torch.full((b,), t, device=lr_hsi.device, dtype=torch.long)
            eps = self.consist_net(y, msi, code, tt)
            x0 = self.to_x0(y, eps, tt)
            if self.use_projection:
                x0 = self._consistent(x0, range_part, kernel)
            else:
                x0 = x0.clamp(0, 1)
            if k == len(steps) - 1:
                break
            # re-noise to the next step (stochastic consistency sampling)
            tp = steps[k + 1]
            a = self.schedule.sqrt_a(tp)
            y = a * x0 + self.schedule.sqrt_1ma(tp) * torch.randn_like(x0)
        return x0
