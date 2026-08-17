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

        if self.cfg.use_krylov:
            blend = self.blend if self.cfg.use_learned_combo else None
            out, residuals = krylov_gmres(x0, b, A, Pinv, self.cfg.n_stages,
                                          blend, alpha_gates)
        else:
            out, residuals = richardson_solve(
                x0, b, A, Pinv, self.cfg.n_stages, self.cfg.rich_alpha)

        return {"out": out.clamp(0, 1), "residuals": residuals}