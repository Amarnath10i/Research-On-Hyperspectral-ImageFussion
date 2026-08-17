"""GraphDIP model: a GNN prior over the superpixel graph.

Nodes = superpixels; edges = kNN in node-feature space.  Message passing
aggregates neighbour representations; the mixing stage is one of

  linear       h' = W (agg h)                    (linear unmixing, bias-free)
  nonlinear    h' = relu(W (agg h))              (non-linear mixing)
  attention    h' = relu( sum_j softmax(q_i.k_j) W h_j )

Each node emits a C-band spectrum; the fused image gathers node outputs by the
hard superpixel labels (differentiable w.r.t. the node outputs).  The physics-
only objective is applied per scene at inference (deep-image-prior style).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from proposal1.daetf.degrade import gaussian_kernel2d

from proposal2.krylovnet.solver import FusionOperator

from .config import Config


class MixingLayer(nn.Module):
    def __init__(self, hidden: int, mix_type: str = "attention"):
        super().__init__()
        self.mix_type = mix_type
        bias = mix_type != "linear"          # linear stage must stay linear
        self.W = nn.Linear(hidden, hidden, bias=bias)
        if mix_type == "attention":
            self.qk = nn.Linear(hidden, hidden, bias=False)

    def forward(self, h: torch.Tensor, hn: torch.Tensor) -> torch.Tensor:
        """h: (n, hidden) own reps; hn: (n, k, hidden) neighbour reps."""
        if self.mix_type == "linear":
            return self.W(hn.mean(dim=1))
        if self.mix_type == "nonlinear":
            return F.relu(self.W(hn.mean(dim=1)))
        q = self.qk(h)
        k = self.qk(hn)
        scores = (q.unsqueeze(1) * k).sum(-1) / (k.shape[-1] ** 0.5)
        w = F.softmax(scores, dim=1)
        return F.relu((w.unsqueeze(-1) * self.W(hn)).sum(dim=1))


class GraphDIP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.op = FusionOperator(cfg.scale, rho=0.0)
        feat_dim = 2 + cfg.msi_bands
        self.node_embed = nn.Linear(feat_dim, cfg.hidden,
                                    bias=cfg.mix_type != "linear")
        self.layers = nn.ModuleList(
            [MixingLayer(cfg.hidden, cfg.mix_type) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(cfg.hidden, cfg.bands,
                              bias=cfg.mix_type != "linear")
        k = gaussian_kernel2d(cfg.blur_ksize, cfg.eval_sigma, cfg.eval_sigma,
                              0.0)
        self.register_buffer("default_kernel", k.float())
        self.register_buffer("srf", torch.zeros(cfg.bands, cfg.msi_bands))

    def set_srf(self, srf: torch.Tensor) -> None:
        s = srf if srf.shape[0] == self.cfg.bands else srf.t().contiguous()
        self.srf.data = s.float()

    @staticmethod
    def neighbors(feats: torch.Tensor, k: int) -> torch.Tensor:
        """kNN neighbour table (n, k) excluding self, from node features."""
        d = torch.cdist(feats, feats)
        d.fill_diagonal_(float("inf"))
        return torch.topk(d, min(k, feats.shape[0] - 1), largest=False).indices

    def tv(self, img: torch.Tensor) -> torch.Tensor:
        return ((img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
                + (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean())

    def forward(self, feats: torch.Tensor, labels: torch.Tensor,
                nb: Optional[torch.Tensor] = None) -> dict:
        if nb is None:
            nb = self.neighbors(feats, self.cfg.graph_k)
        h = self.node_embed(feats)                     # (n, hidden)
        if self.cfg.mix_type != "linear":
            h = F.relu(h)
        for layer in self.layers:
            hn = h[nb]                                 # (n, k, hidden)
            h = layer(h, hn)
        out_nodes = self.head(h)                       # (n, C)
        img = out_nodes[labels]                        # (H, W, C)
        img = img.permute(2, 0, 1).unsqueeze(0)        # (1, C, H, W)
        return {"out": img.clamp(0, 1), "nodes": out_nodes}

    def physics_objective(self, out: torch.Tensor, lr: torch.Tensor,
                          msi: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        op = self.op
        return (F.mse_loss(op.D(out, kernel), lr)
                + F.mse_loss(op.S(out, self.srf), msi))