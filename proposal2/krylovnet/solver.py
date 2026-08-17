"""The unrolled Krylov solver and the fusion operator.

Fusion is the normal equation of the two observation models:

    A x = b,   A = D^T D + S^T S + rho I,   b = D^T X + S^T M

with D the LR-HSI operator (blur + decimate) and S the SRF-to-MSI operator.
D and S are implemented with zero padding so D^T / S^T are *exact* adjoints
(conv_transpose / einsum transpose), which the selfcheck verifies numerically.
The unrolled GMRES grows the Krylov basis one vector per stage and re-solves the
residual-minimising combination; the learned attention blend and the spectral
preconditioner are the network's only learned pieces.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionOperator(nn.Module):
    """D, S and their exact adjoints; A and b of the normal equation."""

    def __init__(self, scale: int, rho: float):
        super().__init__()
        self.scale = scale
        self.rho = rho

    @staticmethod
    def _kernels(kernel: torch.Tensor, b: int) -> torch.Tensor:
        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).expand(b, -1, -1)
        k = kernel.shape[-1]
        return (kernel.to(kernel.device).reshape(b, 1, 1, k, k)
                .expand(b, 1, 1, k, k)), k

    def D(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Blur then decimate (zero padding => exact adjoint)."""
        b, c, h, w = x.shape
        w_, k = self._kernels(kernel, b)
        w_ = w_.expand(b, c, 1, k, k).reshape(b * c, 1, k, k)
        pad = k // 2
        xr = F.pad(x.reshape(1, b * c, h, w), (pad, pad, pad, pad))
        out = F.conv2d(xr, w_, groups=b * c).reshape(b, c, *x.shape[-2:])
        return out[..., ::self.scale, ::self.scale].contiguous()

    def Dt(self, y: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Adjoint of D: zero-insert upsampling, then transposed blur."""
        b, c, h, w = y.shape
        w_, k = self._kernels(kernel, b)
        w_ = w_.expand(b, c, 1, k, k).reshape(b * c, 1, k, k)
        yup = y.new_zeros(b, c, h * self.scale, w * self.scale)
        yup[..., ::self.scale, ::self.scale] = y
        pad = k // 2
        out = F.conv_transpose2d(yup.reshape(1, b * c, *yup.shape[-2:]),
                                 w_, groups=b * c, padding=pad)
        return out.reshape(b, c, *yup.shape[-2:])

    def S(self, x: torch.Tensor, srf: torch.Tensor) -> torch.Tensor:
        """SRF projection to the MSI guide: x @ srf."""
        return torch.einsum("bchw,cm->bmhw", x, srf)

    def St(self, y: torch.Tensor, srf: torch.Tensor) -> torch.Tensor:
        """Adjoint of S."""
        return torch.einsum("bmhw,cm->bchw", y, srf)

    def A(self, v: torch.Tensor, kernel: torch.Tensor,
          srf: torch.Tensor) -> torch.Tensor:
        return (self.Dt(self.D(v, kernel), kernel)
                + self.St(self.S(v, srf), srf) + self.rho * v)

    def b(self, lr: torch.Tensor, msi: torch.Tensor, kernel: torch.Tensor,
          srf: torch.Tensor) -> torch.Tensor:
        return self.Dt(lr, kernel) + self.St(msi, srf)


def krylov_gmres(x0: torch.Tensor, b: torch.Tensor,
                 A: Callable[[torch.Tensor], torch.Tensor],
                 Pinv: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 m: int = 8,
                 blend: Optional[nn.Module] = None,
                 alpha_gates: Optional[torch.Tensor] = None,
                 ridge: float = 1e-6):
    """Differentiable GMRES unrolling.

    Each stage appends one orthonormalised Krylov vector and re-solves the
    residual-minimising combination over the growing basis (normal equations on
    the small Hessenberg system).  Because the subspace grows monotonically, the
    residual is non-increasing.  ``blend`` optionally replaces the combination
    with an attention-blended one; ``alpha_gates`` gives per-stage blend gates
    (hypernetwork).  Returns ``(x, residuals)`` with residuals differentiable.
    """
    B = x0.shape[0]
    dims = tuple(range(1, x0.ndim))
    r = b - A(x0)
    if Pinv is not None:
        r = Pinv(r)
    beta = torch.linalg.vector_norm(r, dim=dims, keepdim=True).clamp_min(1e-12)
    V: List[torch.Tensor] = [r / beta]
    x = x0
    residuals: List[torch.Tensor] = []
    Hbar = None

    def op(v):                      # left-preconditioned operator P^-1 A
        w = A(v)
        return Pinv(w) if Pinv is not None else w

    for k in range(m):
        w = op(V[k])
        cols: List[torch.Tensor] = []
        for j in range(k + 1):
            h = (w * V[j]).sum(dim=dims)
            cols.append(h)
            w = w - h.reshape(B, *([1] * (w.ndim - 1))) * V[j]
        hk1 = torch.linalg.vector_norm(w, dim=dims)
        converged = float(hk1.detach().abs().max()) < 1e-9
        cols.append(hk1 * 0 if converged else hk1)   # last row ~ 0 on breakdown

        # grow the Hessenberg matrix with the new column (k+2) x (k+1)
        Hbar_new = torch.zeros(B, k + 2, k + 1,
                               device=x0.device, dtype=x0.dtype)
        if Hbar is not None:
            Hbar_new[:, :k + 1, :k] = Hbar
        for j, c in enumerate(cols):
            Hbar_new[:, j, k] = c
        Hbar = Hbar_new
        gg = torch.zeros(B, k + 2, 1, device=x0.device, dtype=x0.dtype)
        gg[:, 0, 0] = beta.reshape(B)

        # pseudo-inverse least squares (robust to rank-deficient Hbar)
        c = torch.linalg.pinv(Hbar) @ gg           # (B, k+1, 1)

        if blend is not None:
            c = _blend(blend, V, c, k + 1, alpha_gates, k)

        xk = x0
        for j in range(k + 1):
            xk = xk + c[:, j].reshape(B, *([1] * (x0.ndim - 1))) * V[j]
        residuals.append(b - A(xk))
        x = xk
        if converged:                              # Krylov space exhausted
            break
        V.append(w / hk1.reshape(B, *([1] * (w.ndim - 1))).clamp_min(1e-12))
    return x, residuals


def _blend(blend: nn.Module, V: List[torch.Tensor], c: torch.Tensor, k1: int,
           alpha_gates: Optional[torch.Tensor], k: int) -> torch.Tensor:
    """Attention blend: ``(1-alpha) c_gmres + alpha c_learned``.

    The basis size ``k1`` grows with the stage, so features are zero-padded to
    the module's fixed input width before the attention, then sliced back.
    """
    B = c.shape[0]
    feats = torch.stack(
        [torch.linalg.vector_norm(v, dim=tuple(range(1, v.ndim)))
         for v in V[:k1]], dim=-1)                          # (B, k1)
    n_in = blend.attn.in_features
    if k1 < n_in:
        pad = torch.zeros(B, n_in - k1, device=feats.device, dtype=feats.dtype)
        feats = torch.cat([feats, pad], dim=-1)
    a = blend.attn(feats)[:, :k1].unsqueeze(-1)          # (B, k1, 1)
    alpha = torch.sigmoid(blend.alpha)
    if alpha_gates is not None:
        alpha = alpha * alpha_gates[:, k].unsqueeze(-1).unsqueeze(-1)
    return (1 - alpha) * c + alpha * a


def richardson_solve(x0: torch.Tensor, b: torch.Tensor,
                     A: Callable[[torch.Tensor], torch.Tensor],
                     Pinv: Optional[Callable[[torch.Tensor], torch.Tensor]],
                     steps: int, alpha: float):
    """Stage-1 baseline: fixed-step fixed-point iteration (no learning)."""
    x = x0
    residuals: List[torch.Tensor] = []
    for _ in range(steps):
        r = b - A(x)
        if Pinv is not None:
            r = Pinv(r)
        x = x + alpha * r
        residuals.append(b - A(x))
    return x, residuals


class Blend(nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self.attn = nn.Linear(m, m)
        self.alpha = nn.Parameter(torch.tensor(-4.0))   # start near pure GMRES

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.attn(feats).unsqueeze(-1)


class Hypernet(nn.Module):
    """Condition-adaptive stage gating: reads a conditioning proxy and gates
    the blend strength per stage."""

    def __init__(self, m: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, m))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(cond))