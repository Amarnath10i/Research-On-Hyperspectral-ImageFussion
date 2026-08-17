"""Combined observation operator A = [D; R] for P1 (identifiability-aware fusion).

Extends the spatial-only RangeNullProjector (proposal5.spectralflow) to the
joint operator

    A(X) = [ D(X) ;  R^T X ],          A^T[Y_H; Y_M] = D^T Y_H + R Y_M,

where D is the per-band blur-and-decimate operator and R (bands x msi_bands)
is the spectral response.  Because D (spatial, per-band) and R (spectral,
per-pixel) commute, the normal operator

    A A^T (z_H, z_M) = ( D D^T z_H + D R^T z_M,  R D^T z_H + R R^T z_M )

is well defined and symmetric, and the range/null decomposition

    X_obs = A^T (A A^T)^-1 Y        (pinned by the data)
    X_amb = P_N v = (I - A^T(A A^T)^-1 A) v   (genuinely free)

satisfies A(X_obs + X_amb) = Y for ANY v.  This is the P1 structural identity:
the observation-constrained component is preserved exactly, and the model only
ever fills the ambiguous (null-space) component.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from proposal5.spectralflow.nullspace import DegradationOperator


def default_srf(bands: int, msi_bands: int, width: float = 0.05) -> torch.Tensor:
    lam = torch.linspace(0.0, 1.0, bands)
    centers = torch.linspace(0.15, 0.85, msi_bands)
    R = torch.exp(-((lam[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
    return R / R.sum(0, keepdim=True).clamp_min(1e-12)


def block_cg(applyA: Callable[[Tuple], Tuple], rhs: Tuple,
             steps: int, tol: float = 1e-10) -> Tuple:
    """Batched CG over a pair of observation tensors (z_H, z_M)."""
    z = tuple(torch.zeros_like(r) for r in rhs)
    ap0 = applyA(z)
    r = tuple(rh - a for rh, a in zip(rhs, ap0))
    p = tuple(ri.clone() for ri in r)
    rs = sum((ri * ri).flatten(1).sum(1) for ri in r)
    shape = (rhs[0].shape[0],) + (1,) * (rhs[0].dim() - 1)
    for _ in range(steps):
        ap = applyA(p)
        denom = sum((pi * ai).flatten(1).sum(1) for pi, ai in zip(p, ap))
        alpha = (rs / denom.clamp_min(tol)).reshape(*shape)
        z = tuple(zi + alpha * pi for zi, pi in zip(z, p))
        r = tuple(ri - alpha * ai for ri, ai in zip(r, ap))
        rs_new = sum((ri * ri).flatten(1).sum(1) for ri in r)
        beta = (rs_new / rs.clamp_min(tol)).reshape(*shape)
        p = tuple(ri + beta * pi for ri, pi in zip(r, p))
        rs = rs_new
    return z


class CombinedOperator(nn.Module):
    """A = [D; R]: forward, exact adjoint, normal operator, and the joint
    range/null projector (one block-CG solve in the observation space)."""

    def __init__(self, scale: int, bands: int, msi_bands: int,
                 ksize: int = 9, sigma: float = 1.2, srf: Optional[torch.Tensor] = None,
                 cg_steps: int = 60, ridge: float = 1e-6):
        super().__init__()
        self.D = DegradationOperator(scale, ksize, sigma)
        self.scale = scale
        self.bands = bands
        self.msi_bands = msi_bands
        self.cg_steps = cg_steps
        self.ridge = ridge
        if srf is None:
            srf = default_srf(bands, msi_bands)
        self.register_buffer("srf", srf.float())

    # -- spectral response operators (exact adjoints) -------------------------
    @staticmethod
    def _flatten_lead(x):
        return x.reshape(-1, *x.shape[-3:]), x.shape[:-3]

    def R(self, x: torch.Tensor) -> torch.Tensor:
        """R^T x: (...,B,H,W) -> (...,M,H,W)."""
        x3, lead = self._flatten_lead(x)
        m = torch.einsum("nbhw,bm->nmhw", x3, self.srf)
        return m.reshape(*lead, -1, *x.shape[-2:])

    def Rt(self, m: torch.Tensor) -> torch.Tensor:
        """R m: (...,M,H,W) -> (...,B,H,W)."""
        m3, lead = self._flatten_lead(m)
        x = torch.einsum("nmhw,bm->nbhw", m3, self.srf)
        return x.reshape(*lead, -1, *m.shape[-2:])

    # -- combined operator -----------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """A x -> (Y_H, Y_M)."""
        return self.D(x), self.R(x)

    def adjoint(self, yH: torch.Tensor, yM: torch.Tensor,
                out_hw: Tuple[int, int]) -> torch.Tensor:
        """A^T [Y_H; Y_M] -> (B,H,W)."""
        return self.D.transpose(yH, out_hw) + self.Rt(yM)

    def apply_gram(self, zH: torch.Tensor, zM: torch.Tensor,
                   out_hw: Tuple[int, int]):
        """(A A^T + ridge I)(z_H, z_M)."""
        Dt = self.D.transpose(zH, out_hw)
        RtM = self.Rt(zM)
        wH = self.D(Dt) + self.D(RtM) + self.ridge * zH
        wM = self.R(Dt) + self.R(RtM) + self.ridge * zM
        return wH, wM

    # -- projectors ------------------------------------------------------------
    def pinv(self, yH: torch.Tensor, yM: torch.Tensor,
             out_hw: Tuple[int, int]) -> torch.Tensor:
        """A^T (A A^T)^-1 [Y_H; Y_M] - the component the data determines."""
        zH, zM = block_cg(lambda p: self.apply_gram(p[0], p[1], out_hw),
                          (yH, yM), self.cg_steps)
        return self.adjoint(zH, zM, out_hw)

    def project_range(self, x: torch.Tensor) -> torch.Tensor:
        """X_obs = A^T (A A^T)^-1 A X."""
        out_hw = (x.shape[-2], x.shape[-1])
        yH, yM = self.forward(x)
        return self.pinv(yH, yM, out_hw)

    def project_null(self, v: torch.Tensor) -> torch.Tensor:
        """P_N v = v - A^T (A A^T)^-1 A v."""
        out_hw = (v.shape[-2], v.shape[-1])
        yH, yM = self.forward(v)
        return v - self.pinv(yH, yM, out_hw)

    def consistent(self, yH: torch.Tensor, yM: torch.Tensor,
                   v: torch.Tensor) -> torch.Tensor:
        """A^T(A A^T)^-1 Y + P_N v: the consistent set."""
        out_hw = (v.shape[-2], v.shape[-1])
        return self.pinv(yH, yM, out_hw) + self.project_null(v)