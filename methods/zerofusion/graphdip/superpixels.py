"""Superpixel extraction from the MSI guide via seeded k-means.

Each HR pixel carries (normalised position, per-band MSI colour); k-means
clusters these into ``n_seg`` superpixels.  The output is the hard label map and
the cluster centres (used to build node features).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kmeans_superpixels(msi: torch.Tensor, n_seg: int, spatial_weight: float = 1.0,
                       iters: int = 30, seed: int = 0) -> tuple:
    """msi: (M, H, W).  Returns ``(labels (H,W), centres (n_seg, 2+M))``."""
    torch.manual_seed(seed)
    M, H, W = msi.shape
    ys = torch.linspace(0, 1, H, device=msi.device)[:, None].expand(H, W)
    xs = torch.linspace(0, 1, W, device=msi.device)[None, :].expand(H, W)
    mm = msi.reshape(M, -1)
    lo = mm.min(-1, keepdim=True).values
    hi = mm.max(-1, keepdim=True).values
    mm = ((mm - lo) / (hi - lo + 1e-9)).reshape(M, H, W)
    feats = torch.stack(
        [spatial_weight * ys, spatial_weight * xs] + [mm[i] for i in range(M)],
        dim=0).reshape(2 + M, -1).t()                      # (H*W, d)

    idx = torch.randperm(H * W, device=msi.device)[:n_seg]
    centres = feats[idx].clone()
    for _ in range(iters):
        d = torch.cdist(feats, centres)                     # (H*W, n_seg)
        lab = d.argmin(-1).reshape(H, W)
        oh = F.one_hot(lab.reshape(-1), n_seg).float()
        counts = oh.sum(0).clamp_min(1.0)
        centres = (oh.t() @ feats) / counts.unsqueeze(-1)
    labels = d.argmin(-1).reshape(H, W)
    return labels, centres


def node_features(msi: torch.Tensor, labels: torch.Tensor,
                  centres: torch.Tensor) -> torch.Tensor:
    """Per-node features (position + mean MSI colour) from the cluster centres."""
    return centres