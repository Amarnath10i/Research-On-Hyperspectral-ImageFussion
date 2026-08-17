r"""Range/null-space decomposition and the observed-operator projector.

THE FORMULATION
---------------
The low-resolution hyperspectral observation fixes part of the solution:

    X = D(Y),     D = blur then decimate,

and D has a non-trivial null space: many HR cubes explain the same LR cube
exactly.  Writing the split

    Y_hat = D_pinv(X) + P_perp( v )
            \_______/    \_______/
           fixed by        genuinely free
           the data       (what a fusion
                          method must supply)

with D_pinv = D^T (D D^T)^-1 and P_perp = I - D_pinv D, the reconstruction
satisfies

    D(Y_hat) = D(D_pinv X) + D(P_perp v) = X + 0 = X

for *any* v whatsoever.  Data consistency is an algebraic identity, not a
penalty.

WHAT THIS MEANS FOR A GENERATIVE METHOD
---------------------------------------
SpectralFlow never asks a network to predict the whole cube.  The range
component D_pinv(X) is computed in closed form from the measured LR-HSI.  The
generative model only ever produces the null-space component - the degrees of
freedom the observation genuinely leaves free - and the sampler re-projects
onto the consistent set at every reverse step.  This is what separates the
formulation from a diffusion method that generates freely and hopes the
observation is satisfied.

The cost of the projection is one small CG solve in *low-resolution* space per
call.  The inner system (D D^T + ridge I) lives on the LR grid, which is small,
so the projector runs inside every reverse step without dominating the cost.

The operator D may be fixed (a known eval kernel) or estimated by the network
(blind setting).  The identity D P_perp = 0 holds with respect to whichever D
is supplied; estimating the kernel and building the projector from it couples
the two, which is what makes the degradation head load-bearing.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- operators
def _kernel_weight(kernel: torch.Tensor, channels: int, ksize: int,
                   batch: int) -> torch.Tensor:
    """Expand a per-sample kernel into grouped-conv weights."""
    k = kernel.reshape(batch, 1, 1, ksize, ksize)
    return k.expand(batch, channels, 1, ksize, ksize).reshape(
        batch * channels, 1, ksize, ksize)


class DegradationOperator(nn.Module):
    """D and its exact adjoint D^T.

    Zero padding throughout: reflect padding is not self-adjoint, and mixing it
    with a transposed-convolution adjoint silently breaks <Dx,y> = <x,D^T y>,
    which would make every CG solve here converge to the wrong answer while
    still looking healthy.
    """

    def __init__(self, scale: int, ksize: int = 9, sigma: float = 1.2):
        super().__init__()
        self.scale, self.ksize = scale, ksize
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
        g = torch.exp(-0.5 * (ax / sigma) ** 2)
        k = torch.outer(g, g)
        self.register_buffer("default_kernel", (k / k.sum())[None])

    def _k(self, kernel: Optional[torch.Tensor], batch: int,
           device, dtype) -> torch.Tensor:
        k = self.default_kernel if kernel is None else kernel
        k = k.to(device=device, dtype=dtype)
        if k.dim() == 2:
            k = k[None]
        if k.shape[0] == 1 and batch > 1:
            k = k.expand(batch, -1, -1)
        return k.reshape(batch, self.ksize, self.ksize)

    def forward(self, y: torch.Tensor, kernel: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """D: [B,C,H,W] -> [B,C,H/s,W/s]."""
        b, c, h, w = y.shape
        k = self._k(kernel, b, y.device, y.dtype)
        wgt = _kernel_weight(k, c, self.ksize, b)
        pad = self.ksize // 2
        yr = F.pad(y.reshape(1, b * c, h, w), (pad,) * 4, mode="constant")
        out = F.conv2d(yr, wgt, groups=b * c).reshape(b, c, h, w)
        return out[..., ::self.scale, ::self.scale].contiguous()

    def transpose(self, x: torch.Tensor, out_hw: Tuple[int, int],
                  kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """D^T: [B,C,h,w] -> [B,C,H,W]. Zero-insert, then correlate with the
        flipped kernel."""
        b, c, h, w = x.shape
        up = x.new_zeros(b, c, out_hw[0], out_hw[1])
        up[..., ::self.scale, ::self.scale] = x
        k = self._k(kernel, b, x.device, x.dtype)
        k = torch.flip(k, dims=(-2, -1))
        wgt = _kernel_weight(k, c, self.ksize, b)
        pad = self.ksize // 2
        upr = F.pad(up.reshape(1, b * c, *out_hw), (pad,) * 4, mode="constant")
        return F.conv2d(upr, wgt, groups=b * c).reshape(b, c, *out_hw)


# ------------------------------------------------------------------ solver
def conjugate_gradient(applyA: Callable[[torch.Tensor], torch.Tensor],
                       rhs: torch.Tensor, steps: int,
                       tol: float = 1e-10) -> torch.Tensor:
    """Batched CG for a symmetric positive-definite operator.

    Per-sample scalars, so one easy sample in a batch cannot stall while a hard
    one converges (or vice versa).
    """
    z = torch.zeros_like(rhs)
    r = rhs - applyA(z)
    p = r.clone()
    rs = (r * r).flatten(1).sum(1)
    for _ in range(steps):
        ap = applyA(p)
        denom = (p * ap).flatten(1).sum(1)
        alpha = (rs / denom.clamp_min(tol)).reshape(-1, 1, 1, 1)
        z = z + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).flatten(1).sum(1)
        beta = (rs_new / rs.clamp_min(tol)).reshape(-1, 1, 1, 1)
        p = r + beta * p
        rs = rs_new
    return z


# --------------------------------------------------------------- projector
class RangeNullProjector(nn.Module):
    """Computes D_pinv(x) and P_perp(v) for the blur-and-decimate operator.

    Both need the same inner solve, (D D^T + ridge I) z = rhs, taken in
    low-resolution space where the system is small.  Defaults are the ones
    measured in proposal1/docs: 8 CG steps put the identity at solver
    tolerance, and ridge=1e-6 keeps the solve well conditioned at negligible
    cost to the exactness claim.
    """

    def __init__(self, scale: int, ksize: int = 9, sigma: float = 1.2,
                 cg_steps: int = 8, ridge: float = 1e-6):
        super().__init__()
        self.D = DegradationOperator(scale, ksize, sigma)
        self.scale, self.cg_steps, self.ridge = scale, cg_steps, ridge

    def _normal_op(self, out_hw: Tuple[int, int],
                   kernel: Optional[torch.Tensor]) -> Callable:
        """z -> (D D^T + ridge I) z, acting in low-resolution space."""
        def applyA(z: torch.Tensor) -> torch.Tensor:
            return self.D(self.D.transpose(z, out_hw, kernel), kernel) \
                + self.ridge * z
        return applyA

    def pinv(self, x_lr: torch.Tensor, out_hw: Tuple[int, int],
             kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """D_pinv x = D^T (D D^T)^-1 x - the component the data determines."""
        z = conjugate_gradient(self._normal_op(out_hw, kernel), x_lr,
                               self.cg_steps)
        return self.D.transpose(z, out_hw, kernel)

    def project_null(self, v: torch.Tensor,
                     kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """P_perp v = v - D_pinv(D v) - the component the data cannot see."""
        hw = (v.shape[-2], v.shape[-1])
        return v - self.pinv(self.D(v, kernel), hw, kernel)

    def consistent(self, x_lr: torch.Tensor, v: torch.Tensor,
                   kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """D_pinv(x) + P_perp(v): the consistent set the sampler lives on."""
        hw = (v.shape[-2], v.shape[-1])
        return self.pinv(x_lr, hw, kernel) + self.project_null(v, kernel)


# ------------------------------------------------------------- degradation
def decode_degradation_params(raw: torch.Tensor, min_sigma: float = 0.3
                              ) -> torch.Tensor:
    """Convert unconstrained head outputs into physical degradation parameters.

    Returns physical ``[sigma_x, sigma_y, sin(2 theta), cos(2 theta), noise]``
    values.  The doubled angle avoids the half-turn ambiguity of an ellipse.
    """
    if raw.ndim != 2 or raw.shape[1] < 5:
        raise ValueError("expected raw degradation parameters [B, >=5]")
    sigma_x = F.softplus(raw[:, 0]) + min_sigma
    sigma_y = F.softplus(raw[:, 1]) + min_sigma
    orient = raw[:, 2:4]
    orient_norm = orient.norm(dim=1, keepdim=True)
    orient_unit = orient / orient_norm.clamp_min(1e-6)
    default_orient = torch.zeros_like(orient)
    default_orient[:, 1] = 1.0
    orient = torch.where(orient_norm > 1e-6, orient_unit, default_orient)
    noise = F.softplus(raw[:, 4])
    return torch.cat([sigma_x[:, None], sigma_y[:, None], orient, noise[:, None]], dim=1)


def kernel_from_params(params: torch.Tensor, ksize: int = 9,
                       min_sigma: float = 0.3) -> torch.Tensor:
    """Build a differentiable anisotropic Gaussian kernel from *physical*
    degradation parameters (sx, sy, sin2t, cos2t, ...).  ``params`` must have
    passed through :func:`decode_degradation_params`."""
    sx = params[:, 0].clamp_min(min_sigma)
    sy = params[:, 1].clamp_min(min_sigma)
    s2, c2 = params[:, 2], params[:, 3]
    norm = torch.sqrt(s2 ** 2 + c2 ** 2).clamp_min(1e-6)
    theta = 0.5 * torch.atan2(s2 / norm, c2 / norm)

    ax = torch.arange(ksize, device=params.device, dtype=params.dtype)
    ax = ax - (ksize - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    xx, yy = xx[None], yy[None]                      # [1,k,k]
    cos_t = torch.cos(theta)[:, None, None]
    sin_t = torch.sin(theta)[:, None, None]
    xr = xx * cos_t + yy * sin_t
    yr = -xx * sin_t + yy * cos_t
    k = torch.exp(-0.5 * ((xr / sx[:, None, None]) ** 2
                          + (yr / sy[:, None, None]) ** 2))
    return k / k.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


# ------------------------------------------------------------- verification
@torch.no_grad()
def check_adjoint(scale: int = 4, ksize: int = 9, size: int = 32,
                  channels: int = 5, tol: float = 1e-5) -> float:
    """<D y, x> must equal <y, D^T x>."""
    torch.manual_seed(0)
    D = DegradationOperator(scale, ksize)
    y = torch.randn(2, channels, size, size)
    x = torch.randn(2, channels, size // scale, size // scale)
    lhs = (D(y) * x).sum()
    rhs = (y * D.transpose(x, (size, size))).sum()
    err = abs((lhs - rhs).item()) / max(abs(lhs.item()), 1e-12)
    print(f"[check] adjoint <Dy,x> vs <y,D^Tx>: rel err {err:.2e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


@torch.no_grad()
def check_consistency(scale: int = 4, size: int = 32, channels: int = 5,
                      tol: float = 1e-3) -> float:
    """THE claim: D(consistent(x, v)) == x for an arbitrary v."""
    torch.manual_seed(0)
    P = RangeNullProjector(scale, cg_steps=30, ridge=0.0)
    x = torch.rand(2, channels, size // scale, size // scale)
    v = torch.randn(2, channels, size, size) * 3.0      # arbitrary, large
    out = P.consistent(x, v)
    err = (P.D(out) - x).abs().max().item() / max(x.abs().max().item(), 1e-12)
    print(f"[check] data consistency  max|D(Y_hat) - X| / max|X| = {err:.2e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


def run_all(verbose: bool = True) -> bool:
    ok = True
    ok &= check_adjoint() < 1e-5
    ok &= check_consistency() < 1e-3
    print(f"\n[nullspace] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()