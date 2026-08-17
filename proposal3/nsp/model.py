"""NSP model: the discretised diffusion-reaction PDE with a learned tensor.

The network unrolls explicit-Euler steps of

    du/dt = div(D(u) grad u) - lam1 D^T(D u - X) - lam2 S^T(S u - M)

The observation terms are soft penalties, so the steady state is the minimiser
of  E = int g|grad u|^2 + lam1/2 ||D u - X||^2 + lam2/2 ||S u - M||^2
(consistency in the penalty sense, not as hard constraints -- see docs).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from proposal1.daetf.degrade import gaussian_kernel2d

from proposal2.krylovnet.solver import FusionOperator

from .config import Config
from .pde import TensorNet, divergence


def _soft_positive(p: Optional[torch.nn.Parameter], fixed: float) -> torch.Tensor:
    return F.softplus(p) if p is not None else torch.tensor(
        fixed, dtype=torch.float32)


class NSPModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.op = FusionOperator(cfg.scale, rho=0.0)
        self.tensor_net = TensorNet(cfg.bands, cfg.msi_bands,
                                    cfg.tensor_hidden, cfg.tensor_layers) \
            if cfg.use_tensor_net else None
        if cfg.learn_dt:
            self.log_dt = nn.Parameter(torch.tensor(float(cfg.dt)).log())
        else:
            self.register_buffer("log_dt", torch.tensor(float(cfg.dt)).log())
        if cfg.learn_lam:
            self.log_lam1 = nn.Parameter(torch.tensor(float(cfg.lam1)).log())
            self.log_lam2 = nn.Parameter(torch.tensor(float(cfg.lam2)).log())
        else:
            self.register_buffer("log_lam1",
                                 torch.tensor(float(cfg.lam1)).log())
            self.register_buffer("log_lam2",
                                 torch.tensor(float(cfg.lam2)).log())
        k = gaussian_kernel2d(cfg.blur_ksize, cfg.eval_sigma, cfg.eval_sigma,
                              0.0)
        self.register_buffer("default_kernel", k.float())
        self.register_buffer("srf", torch.zeros(cfg.bands, cfg.msi_bands))

    def set_srf(self, srf: torch.Tensor) -> None:
        s = srf if srf.shape[0] == self.cfg.bands else srf.t().contiguous()
        self.srf.data = s.float()

    def _dt(self) -> torch.Tensor:
        dt = F.softplus(self.log_dt)
        if self.cfg.scale_adaptive_dt:
            dt = dt / self.cfg.scale
        return dt

    def forward(self, x: torch.Tensor, m: torch.Tensor,
                kernel: Optional[torch.Tensor] = None) -> dict:
        kernel = self.default_kernel if kernel is None else kernel
        scale = self.cfg.scale
        u = F.interpolate(x, scale_factor=scale, mode="bicubic",
                          align_corners=False)
        x_up = F.interpolate(x, scale_factor=scale, mode="bicubic",
                             align_corners=False)
        dt = self._dt()
        lam1 = F.softplus(self.log_lam1)
        lam2 = F.softplus(self.log_lam2)
        residuals = []

        for _ in range(self.cfg.pde_steps):
            diff = torch.zeros_like(u)
            if self.cfg.use_diffusion:
                if self.tensor_net is not None:
                    gxy, gz = self.tensor_net(u, x_up, m)
                else:
                    gxy = u.new_full((), self.cfg.iso_diff).expand_as(u)
                    gz = u.new_full((), self.cfg.iso_diff).expand_as(u)
                diff = divergence(u, gxy, gz)
            r1 = self.op.Dt(self.op.D(u, kernel) - x, kernel)
            r2 = self.op.St(self.op.S(u, self.srf) - m, self.srf)
            u = u + dt * (diff - lam1 * r1 - lam2 * r2)
            residuals.append(torch.linalg.vector_norm(r1 + r2, dim=(1, 2, 3)))

        return {"out": u.clamp(0, 1), "residuals": residuals}