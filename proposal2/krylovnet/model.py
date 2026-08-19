"""KrylovNet model: the unrolled Krylov solver plus the learned pieces.

The network has no spatial feature encoder; capacity lives in (a) the unrolled
GMRES stages and (b) the spectral-graph GNN that builds a per-scene
preconditioner from MSI band statistics.  The SRF and default evaluation kernel
are registered as buffers (set_srf / defaults), and the training kernel is
passed per-batch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from proposal1.daetf.degrade import gaussian_kernel2d

from .config import Config
from .solver import (Blend, FusionOperator, Hypernet, krylov_gmres,
                     richardson_solve)


class SpectralPreconditioner(nn.Module):
    """GNN over the spectral band graph; emits a positive per-band scale.

    Nodes = spectral bands, edges = kNN affinities computed from band-level
    statistics of the MSI guide.  GCN layers (row-stochastic normalisation)
    refine the features and the head emits ``s = exp(...)``.  A linear skip
    connection lets the net trivially fit a prescribed diagonal target (used by
    the selfcheck to show conditioning improves).
    """

    def __init__(self, bands: int, graph_k: int = 4, hidden: int = 32,
                 gcn_layers: int = 2, feat_dim: int = 2):
        super().__init__()
        self.bands = bands
        self.graph_k = graph_k
        self.embed = nn.Linear(feat_dim, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(gcn_layers)])
        self.head = nn.Linear(hidden, 1)
        self.skip = nn.Linear(feat_dim, 1)

    def build_affinity(self, feats: torch.Tensor) -> torch.Tensor:
        """kNN band graph, symmetric, row-stochastic, with self loops."""
        b = feats.shape[0]
        d = torch.cdist(feats, feats)                      # (b, n, n)
        k = min(self.graph_k, self.bands - 1)
        idx = torch.topk(d, k=k, dim=-1, largest=False).indices
        adj = torch.zeros(b, self.bands, self.bands,
                          device=feats.device, dtype=feats.dtype)
        ar = torch.arange(self.bands, device=feats.device)
        adj[torch.arange(b).reshape(b, 1, 1), ar.reshape(1, self.bands, 1),
            idx] = 1.0
        adj = adj + adj.transpose(1, 2)
        adj = torch.clamp(adj, max=1.0) + torch.eye(self.bands,
                                                    device=feats.device)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return adj / deg                                   # (b, n, n)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        adj = self.build_affinity(feats)
        h = F.relu(self.embed(feats))                      # (b, n, h)
        for layer in self.layers:
            h = F.relu(adj @ layer(h))
        s = torch.exp(self.head(h).squeeze(-1) + self.skip(feats).squeeze(-1))
        return s                                           # (b, bands)


class KrylovNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.op = FusionOperator(cfg.scale, cfg.rho)
        self.precond = SpectralPreconditioner(
            cfg.bands, cfg.graph_k, cfg.hidden, cfg.gcn_layers)
        self.blend = Blend(cfg.n_stages)
        # Learned proximal prior. Without it the model is a pure Tikhonov
        # solver with no image prior and loses to bicubic; see ResidualDenoiser.
        self.prior = (ResidualDenoiser(cfg.bands, cfg.prior_width,
                                       cfg.prior_blocks)
                      if getattr(cfg, "use_prior", True) else None)
        self.hypernet = Hypernet(cfg.n_stages) if cfg.use_hypernet else None
        k = gaussian_kernel2d(cfg.blur_ksize, cfg.eval_sigma, cfg.eval_sigma,
                              0.0)
        self.register_buffer("default_kernel", k.float())
        self.register_buffer("srf", torch.zeros(cfg.msi_bands, cfg.bands))

    def set_srf(self, srf: torch.Tensor) -> None:
        """srf: [bands, msi_bands] (or transposed; stored as [bands, msi_bands])."""
        s = srf if srf.shape[0] == self.cfg.bands else srf.t().contiguous()
        self.srf.data = s.float()

    @staticmethod
    def _band_feats(hsi: torch.Tensor) -> torch.Tensor:
        """Per-band mean/std of the LR HSI -> (B, bands, 2)."""
        mu = hsi.mean(dim=(2, 3))
        sd = hsi.std(dim=(2, 3))
        return torch.stack([mu, sd], dim=-1)

    def forward(self, lr: torch.Tensor, msi: torch.Tensor,
                kernel: Optional[torch.Tensor] = None) -> dict:
        kernel = self.default_kernel if kernel is None else kernel
        B = lr.shape[0]
        b = self.op.b(lr, msi, kernel, self.srf)
        x0 = F.interpolate(lr, scale_factor=self.cfg.scale, mode="bicubic",
                           align_corners=False)
        A = lambda v: self.op.A(v, kernel, self.srf)

        Pinv = None
        if self.cfg.use_precond:
            s = self.precond(self._band_feats(lr))
            Pinv = lambda v: v * s.reshape(B, self.cfg.bands,
                                           *([1] * (v.ndim - 2)))

        alpha_gates = None
        if self.cfg.use_hypernet and self.hypernet is not None:
            r0 = (torch.linalg.vector_norm(b - A(x0), dim=(1, 2, 3))
                  / torch.linalg.vector_norm(b, dim=(1, 2, 3)))
            alpha_gates = self.hypernet(r0.unsqueeze(-1))

        def _solve(x_init, rhs, n_stages):
            Ak = lambda v: self.op.A(v, kernel, self.srf)
            if self.cfg.use_krylov:
                blend = self.blend if self.cfg.use_learned_combo else None
                return krylov_gmres(x_init, rhs, Ak, Pinv, n_stages, blend,
                                    alpha_gates)
            return richardson_solve(x_init, rhs, Ak, Pinv, n_stages,
                                    self.cfg.rich_alpha)

        if self.prior is None:
            out, residuals = _solve(x0, b, self.cfg.n_stages)
        else:
            # Plug-and-play ordering: data step, then prior step, ending on the
            # PRIOR.
            #
            # Ending on a data step destroys the prior's contribution: A already
            # contains rho*I and b does not reference the prior, so the solver's
            # fixed point is the Tikhonov solution shrinking toward zero, and any
            # detail the prior added is converged away. Measured: overfitting a
            # single patch plateaued at 23.18 dB with 154k parameters.
            #
            # Feeding the prior back through the RHS as b + rho*v was tried and
            # diverged (L1 213) - with rho = 1e-3 the anchor is far too weak to
            # constrain the solve while still coupling the iterations, so the
            # outer loop is unstable. Ending on the prior keeps the prior's
            # output intact; data consistency is then carried by the physics
            # terms in the loss rather than by a final projection.
            n_outer = max(1, self.cfg.n_outer)
            inner = max(1, self.cfg.n_stages // n_outer)
            out, residuals = x0, []
            for _ in range(n_outer):
                out, res = _solve(out, b, inner)
                residuals = residuals + list(res)
                out = self.prior(out)

        # Hard clamping during training zeroes the gradient wherever the solver
        # overshoots, which is precisely where it most needs to be corrected.
        out = out.clamp(0, 1) if not self.training else out
        return {"out": out, "residuals": residuals}

class ResidualDenoiser(nn.Module):
    """Learned proximal operator applied between Krylov data steps.

    WHY THIS IS NEEDED
    ------------------
    The solver alone minimises ||D x - X||^2 + ||S x - M||^2 + rho||x||^2.
    That is a Tikhonov least-squares fit to the observations, and it carries no
    image prior at all - so it returns the smooth minimum-norm member of the
    solution set and cannot recover high-frequency spectral detail that the
    observations underdetermine. Measured: the solver-only model scored 29.49 dB
    against plain bicubic at 31.31 dB, i.e. worse than doing nothing.

    Interleaving a learned denoiser turns the unrolling into half-quadratic
    splitting / plug-and-play: the data step enforces agreement with the
    observations, the prior step supplies what the observations cannot. This is
    where SOTA unfolding methods put their capacity, and it is the difference
    between a 2.3k-parameter solver and a competitive model.

    Zero-initialised output, so the network starts exactly at the solver's
    answer and can only improve on it - training never has to first undo a
    random perturbation of an already-reasonable estimate.
    """

    def __init__(self, bands: int, width: int = 64, blocks: int = 4):
        super().__init__()
        self.head = nn.Conv2d(bands, width, 3, 1, 1)
        self.body = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, 1, 1),
                          nn.LeakyReLU(0.1, True),
                          nn.Conv2d(width, width, 3, 1, 1))
            for _ in range(blocks)
        ])
        self.tail = nn.Conv2d(width, bands, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
        self.act = nn.LeakyReLU(0.1, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.head(x))
        for blk in self.body:
            h = h + blk(h)
        return x + self.tail(h)
