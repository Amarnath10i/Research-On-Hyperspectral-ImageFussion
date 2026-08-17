"""ManifoldFlow model: rectified flow with a tangent-space (null-space) field.

The consistent set is ``D(y) = X``.  ``RangeNullProjector`` (reused from
proposal5.spectralflow) gives the split

    y = D_pinv(X) + P_perp(v),   D(P_perp v) = 0,

so ``D(y) = X`` for any ``v``.  The flow starts at ``y0 = D_pinv(X)`` and
integrates ``dy/dt = P_perp(v_theta(y, m, t))``; because every velocity lies in
the null space, every iterate stays consistent by construction.  Training is
rectified flow matching: ``v_theta`` regresses ``u = gt - y0`` along the
straight interpolation ``y_t = y0 + t u``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from proposal5.spectralflow.nullspace import RangeNullProjector

from .config import Config


class VelocityNet(nn.Module):
    def __init__(self, bands: int, msi_bands: int, hidden: int = 32,
                 blocks: int = 3):
        super().__init__()
        self.proj_m = nn.Conv2d(msi_bands, bands, 1)
        body = []
        ch = 2 * bands + 1
        body.append(nn.Conv2d(ch, hidden, 3, padding=1))
        for _ in range(blocks - 1):
            body += [nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1)]
        body.append(nn.SiLU())
        body.append(nn.Conv2d(hidden, bands, 3, padding=1))
        self.net = nn.Sequential(*body)

    def forward(self, y: torch.Tensor, m: torch.Tensor,
                t) -> torch.Tensor:
        B, C, H, W = y.shape
        if not torch.is_tensor(t):
            t = torch.full((B,), float(t), device=y.device, dtype=y.dtype)
        if t.dim() == 0:
            t = t.expand(B)
        tch = t.reshape(B, 1, 1, 1).expand(B, 1, H, W)
        return self.net(torch.cat([y, self.proj_m(m), tch], dim=1))


class ManifoldFlow(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.projector = RangeNullProjector(cfg.scale, cfg.blur_ksize,
                                            cfg.eval_sigma, cfg.cg_steps,
                                            cfg.cg_ridge)
        self.velocity = VelocityNet(cfg.bands, cfg.msi_bands, cfg.hidden,
                                    cfg.n_flow_blocks)
        self.register_buffer("srf", torch.zeros(cfg.bands, cfg.msi_bands))

    @property
    def D(self):
        return self.projector.D

    def set_srf(self, srf: torch.Tensor) -> None:
        s = srf if srf.shape[0] == self.cfg.bands else srf.t().contiguous()
        self.srf.data = s.float()

    def pinv(self, x_lr: torch.Tensor, out_hw, kernel=None) -> torch.Tensor:
        return self.projector.pinv(x_lr, out_hw, kernel)

    def velocity_field(self, y: torch.Tensor, m: torch.Tensor,
                       t: torch.Tensor, kernel=None) -> torch.Tensor:
        v = self.velocity(y, m, t)
        if self.cfg.tangent:
            v = self.projector.project_null(v, kernel)
        return v

    def training_step(self, x_lr: torch.Tensor, m: torch.Tensor,
                      gt: torch.Tensor, t: torch.Tensor,
                      kernel=None) -> dict:
        hw = (gt.shape[-2], gt.shape[-1])
        y0 = self.pinv(x_lr, hw, kernel)
        u = gt - y0
        yt = y0 + t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * u
        v = self.velocity_field(yt, m, t, kernel)
        return {"y0": y0, "target": u, "velocity": v, "t": t}

    def sample(self, x_lr: torch.Tensor, m: torch.Tensor,
               kernel=None, steps: Optional[int] = None,
               tangent: Optional[bool] = None) -> torch.Tensor:
        steps = steps or self.cfg.sample_steps
        tangent = self.cfg.tangent if tangent is None else tangent
        hw = (x_lr.shape[-2] * self.cfg.scale, x_lr.shape[-1] * self.cfg.scale)
        y = self.pinv(x_lr, hw, kernel)
        with torch.no_grad():
            for k in range(steps):
                t = (k + 0.5) / steps
                v = self.velocity(y, m, t)
                if tangent:
                    v = self.projector.project_null(v, kernel)
                y = y + v / steps
        return y.clamp(0, 1)