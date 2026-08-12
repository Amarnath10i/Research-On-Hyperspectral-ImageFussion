r"""Range/null-space decomposition for HSI-MSI fusion.

THE FORMULATION
---------------
The low-resolution observation constrains the solution through

    X = D(Y)                      D = blur then decimate

D has a non-trivial null space: many high-resolution cubes explain the same
observation exactly. Splitting the solution along that structure,

    Y_hat = D_pinv(X)  +  P_perp( F_theta(X, M) )
            \_________/     \___________________/
             determined       genuinely unknown
             by the data      (learned)

where D_pinv = D^T (D D^T)^-1 is the Moore-Penrose pseudo-inverse and
P_perp = I - D_pinv D projects onto the null space of D.

WHY THIS IS DIFFERENT FROM A PHYSICS LOSS
-----------------------------------------
Because D P_perp = D - (D D_pinv) D = D - D = 0, the reconstruction satisfies

    D(Y_hat) = D(D_pinv X) + D(P_perp F) = X + 0 = X

*exactly, for any network output whatsoever.* Data consistency is an algebraic
identity, not a penalty the optimiser trades against other terms. A network
that has memorised the source domain cannot violate the target observation,
because the architecture gives it no way to express a violation.

This also changes what the network is asked to learn. Under a physics *loss*
the network predicts the whole cube and is punished when it disagrees with the
data - so most of its capacity goes on reproducing the component the data
already determines. Here that component is computed in closed form, and the
network only ever supplies the part the observation genuinely leaves free.

COMPUTING THE PSEUDO-INVERSE
----------------------------
D_pinv X = D^T (D D^T)^-1 X. The inner system lives in *low-resolution* space,
so it is small, and D D^T is symmetric positive definite - conjugate gradients
solves it in a handful of matrix-vector products with no matrix ever formed.

A ridge term makes the solve well conditioned when the blur is close to
singular. It is a genuine trade-off, stated plainly: with ridge = 0 the
consistency identity holds to solver tolerance, and with ridge > 0 it holds to
O(ridge). `check_consistency` below measures the actual residual so the claim
is never taken on trust.
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
                       z0: Optional[torch.Tensor] = None,
                       tol: float = 1e-10) -> torch.Tensor:
    """Batched CG for a symmetric positive-definite operator.

    Per-sample scalars, so one easy sample in a batch cannot stall while a hard
    one converges (or vice versa).
    """
    z = torch.zeros_like(rhs) if z0 is None else z0
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
    low-resolution space where the system is small.
    """

    # Defaults chosen from the measured sweeps in this module, not by taste:
    #   cg_steps: 1->5.8e-01, 2->2.0e-01, 4->8.5e-03, 8->6.2e-06, 32->1.9e-06
    #     -> 8 is where the identity is already at solver tolerance; more is
    #        wasted compute inside every forward pass.
    #   ridge:    0->1.9e-06, 1e-6->5.3e-05, 1e-4->5.1e-03, 1e-2->4.3e-01
    #     -> 1e-4 costs three orders of magnitude of consistency, which would
    #        have quietly turned the exactness claim into a rounding argument.
    #        1e-6 keeps the solve well conditioned at negligible cost.
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
             kernel: Optional[torch.Tensor] = None,
             cg_steps: Optional[int] = None) -> torch.Tensor:
        """D_pinv x = D^T (D D^T)^-1 x - the component the data determines."""
        z = conjugate_gradient(self._normal_op(out_hw, kernel), x_lr,
                               cg_steps or self.cg_steps)
        return self.D.transpose(z, out_hw, kernel)

    def project_null(self, v: torch.Tensor,
                     kernel: Optional[torch.Tensor] = None,
                     cg_steps: Optional[int] = None) -> torch.Tensor:
        """P_perp v = v - D_pinv(D v) - the component the data cannot see."""
        hw = (v.shape[-2], v.shape[-1])
        return v - self.pinv(self.D(v, kernel), hw, kernel, cg_steps)

    def compose(self, x_lr: torch.Tensor, v: torch.Tensor,
                kernel: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Y_hat = D_pinv(x) + P_perp(v), with both parts returned separately so
        the split can be inspected and visualised."""
        hw = (v.shape[-2], v.shape[-1])
        range_part = self.pinv(x_lr, hw, kernel)
        null_part = self.project_null(v, kernel)
        return {"out": range_part + null_part,
                "range": range_part, "null": null_part}


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
                      cg_steps: int = 30, ridge: float = 0.0,
                      tol: float = 1e-3) -> float:
    """THE claim: D(Y_hat) == X for an arbitrary network output.

    This is what separates the formulation from a physics loss. If it fails,
    the decomposition is not doing what the paper says it does.
    """
    torch.manual_seed(0)
    P = RangeNullProjector(scale, cg_steps=cg_steps, ridge=ridge)
    x = torch.rand(2, channels, size // scale, size // scale)
    v = torch.randn(2, channels, size, size) * 3.0      # arbitrary, large
    res = P.compose(x, v)
    back = P.D(res["out"])
    err = (back - x).abs().max().item() / max(x.abs().max().item(), 1e-12)
    print(f"[check] data consistency  max|D(Y_hat) - X| / max|X| = {err:.2e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


@torch.no_grad()
def check_null_annihilation(scale: int = 4, size: int = 32, channels: int = 5,
                            cg_steps: int = 30, ridge: float = 0.0,
                            tol: float = 1e-3) -> float:
    """D P_perp = 0: the learned component is invisible to the observation."""
    torch.manual_seed(1)
    P = RangeNullProjector(scale, cg_steps=cg_steps, ridge=ridge)
    v = torch.randn(2, channels, size, size)
    dn = P.D(P.project_null(v))
    err = dn.abs().max().item() / max(P.D(v).abs().max().item(), 1e-12)
    print(f"[check] null annihilation max|D(P_perp v)| = {err:.2e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


@torch.no_grad()
def check_idempotent(scale: int = 4, size: int = 32, channels: int = 5,
                     cg_steps: int = 30, ridge: float = 0.0,
                     tol: float = 1e-3) -> float:
    """P_perp must be a projector: P_perp(P_perp v) == P_perp v."""
    torch.manual_seed(2)
    P = RangeNullProjector(scale, cg_steps=cg_steps, ridge=ridge)
    v = torch.randn(2, channels, size, size)
    p1 = P.project_null(v)
    p2 = P.project_null(p1)
    err = (p2 - p1).abs().max().item() / max(p1.abs().max().item(), 1e-12)
    print(f"[check] idempotence |P(Pv) - Pv| = {err:.2e} "
          f"({'PASS' if err < tol else 'FAIL'})")
    return err


@torch.no_grad()
def check_ridge_tradeoff(scale: int = 4, size: int = 32, channels: int = 5
                         ) -> None:
    """Consistency degrades as O(ridge). Reported rather than hidden."""
    print("[check] ridge vs consistency (the stability/exactness trade-off):")
    for ridge in (0.0, 1e-6, 1e-4, 1e-2):
        e = check_consistency_quiet(scale, size, channels, 30, ridge)
        print(f"          ridge {ridge:<8g} -> rel err {e:.2e}")


@torch.no_grad()
def check_consistency_quiet(scale, size, channels, cg_steps, ridge) -> float:
    torch.manual_seed(0)
    P = RangeNullProjector(scale, cg_steps=cg_steps, ridge=ridge)
    x = torch.rand(2, channels, size // scale, size // scale)
    v = torch.randn(2, channels, size, size) * 3.0
    back = P.D(P.compose(x, v)["out"])
    return (back - x).abs().max().item() / max(x.abs().max().item(), 1e-12)


@torch.no_grad()
def check_cg_convergence(scale: int = 4, size: int = 32, channels: int = 5
                         ) -> None:
    """How many CG steps the identity actually needs - the number that sets the
    cost of the whole formulation."""
    print("[check] CG steps vs consistency:")
    for steps in (1, 2, 4, 8, 16, 32):
        e = check_consistency_quiet(scale, size, channels, steps, 0.0)
        print(f"          {steps:2d} steps -> rel err {e:.2e}")


def run_all(verbose: bool = True) -> bool:
    ok = True
    ok &= check_adjoint() < 1e-5
    ok &= check_consistency() < 1e-3
    ok &= check_null_annihilation() < 1e-3
    ok &= check_idempotent() < 1e-3
    if verbose:
        check_cg_convergence()
        check_ridge_tradeoff()
    print(f"\n[nullspace] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()
