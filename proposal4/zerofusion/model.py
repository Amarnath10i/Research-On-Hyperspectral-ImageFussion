"""ZeroFusion - self-supervised, per-scene HSI-MSI fusion. No training set.

WHERE THIS DIFFERS FROM PROPOSALS 1-3
-------------------------------------
Proposals 1, 2 and 3 all learn from a training split and are then asked to
generalise to a new dataset. ZeroFusion has no training split at all. Given a
single pair (LR-HSI, MSI) it optimises a small network from scratch, on that
pair alone, using only losses that need no ground truth:

    ||B(X) - Y_h||  +  ||R(X) - Y_m||  +  priors

WHY THIS IS THE MOST IMPORTANT OF THE FOUR
------------------------------------------
It is the control arm. Every cross-domain number the other three report is an
attempt to answer "how much does training on CAVE hurt you on Harvard?".
ZeroFusion never trains on CAVE, so it *cannot* suffer domain shift - its
Harvard score is by construction the same kind of quantity as its CAVE score.

That makes it the reference line for the entire study:

  * If a trained proposal beats ZeroFusion cross-domain, the training genuinely
    transferred something useful.
  * If it does not, then whatever that model learned on the source dataset was
    worth less than nothing on the target, and the honest conclusion is that
    per-scene optimisation is the better method - not that our model is good.

Very few fusion papers include this comparison. It is easy to look strong
cross-domain until someone asks whether a method that ignores the training set
entirely would have done better.

THE MODEL
---------
Physically-grounded rather than a generic denoiser, following linear spectral
unmixing:

    X = E @ A       E: [bands, k] endmember spectra
                    A: [k, H, W]  abundance maps, non-negative, sum-to-one

* E is initialised from an SVD of the LR-HSI, so it starts at the scene's own
  spectral subspace, then refined.
* A is produced by a small CNN reading the MSI, which is where the spatial
  detail lives, at full resolution.
* Sum-to-one and non-negativity are imposed by a softmax over k, so every
  reconstructed pixel is a convex combination of physically plausible spectra.
  This is a strong prior that costs nothing and rules out the spectral
  hallucination that shows up as large SAM.

The deep-image-prior effect does the rest: a small convolutional generator
fits smooth structure long before it fits noise, so early stopping regularises.

COST
----
Optimisation is per scene, so inference is expensive (seconds to a minute per
scene) while training cost is zero. That trade is the honest headline and is
reported explicitly rather than hidden.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbundanceNet(nn.Module):
    """(MSI, upsampled LR-HSI) -> abundance maps and a brightness factor.

    Both observations feed the abundances. Reading only the MSI leaves the
    LR-HSI's spectral content to arrive solely through the loss gradient, which
    is far too slow for a per-scene fit with a few hundred steps - measured, it
    lost to bicubic by 12 dB. Conditioning on the upsampled cube gives the
    network the spectra directly and the MSI supplies the spatial detail.

    The brightness head matters too: a strict convex combination of endmembers
    cannot represent illumination changes, so a shaded region of an otherwise
    uniform material is unreachable. A positive per-pixel scale restores that
    degree of freedom while keeping the physical prior on spectral *shape*.

    Capacity stays small on purpose: with no training set, a large network
    simply memorises the observations, noise included.
    """

    def __init__(self, msi_bands: int, hsi_bands: int, k: int,
                 width: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Conv2d(msi_bands + hsi_bands, width, 3, 1, 1),
                  nn.ReLU(inplace=True)]
        for _ in range(depth):
            layers += [nn.Conv2d(width, width, 3, 1, 1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, k, 1)
        self.scale_head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)          # start at scale 1

    def forward(self, msi: torch.Tensor, hsi_up: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(torch.cat([msi, hsi_up], dim=1))
        # Scale is bounded to roughly [0.37, 2.7]. An unbounded exp lets the
        # brightness head run away in the first few steps, saturate the output
        # and - because saturation used to be clamped - kill every gradient.
        return self.head(h), torch.exp(self.scale_head(h).clamp(-1.0, 1.0))


class ZeroFusion(nn.Module):
    """Per-scene linear unmixing model: X = E @ softmax(A)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.endmembers
        self.net = AbundanceNet(cfg.msi_bands, cfg.bands, cfg.endmembers,
                                cfg.width, cfg.depth)
        # endmembers are parameters, initialised per scene from the LR-HSI
        self.E = nn.Parameter(torch.rand(cfg.bands, cfg.endmembers) * 0.1 + 0.4)
        self.register_buffer("_initialised", torch.zeros(1))

    @torch.no_grad()
    def init_endmembers(self, lr_hsi: torch.Tensor) -> None:
        """Start from the scene's own spectral subspace.

        Random endmembers make the optimisation slow and unstable; the leading
        singular vectors of the observed cube are already close to the answer.
        Absolute value keeps them non-negative, as radiance must be.
        """
        c = lr_hsi.shape[1]
        y = lr_hsi[0].reshape(c, -1).float()
        u, _, _ = torch.linalg.svd(y @ y.t())
        e = u[:, :self.k].abs()
        e = e / e.amax(dim=0, keepdim=True).clamp_min(1e-6)
        self.E.data.copy_(e.to(self.E.dtype))
        self._initialised.fill_(1)

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        hsi_up = F.interpolate(lr_hsi, size=msi.shape[-2:], mode="bicubic",
                               align_corners=False).clamp(0, 1)
        logits, scale = self.net(msi, hsi_up)
        a = torch.softmax(logits, dim=1)             # [B,k,H,W] convex weights
        e = self.E.clamp(0, 1).to(a.dtype)           # [bands,k]
        out = torch.einsum('ck,bkhw->bchw', e, a) * scale
        # NOT clamped here. torch.clamp has zero gradient outside its range, so
        # clamping inside the forward pass silently freezes optimisation the
        # moment the estimate saturates - measured as byte-identical results
        # across 200, 600 and 1500 steps. Callers clamp for display and metrics.
        return {"out": out, "abundance": a, "endmembers": e,
                "scale": scale, "feat": a.mean(dim=(2, 3))}

    def features(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        return self.forward(lr_hsi, msi)["feat"]


def abundance_priors(out: Dict, cfg, **kw) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sparsity, smoothness, and a range penalty on the reconstruction.

    Sparsity uses the L1/2 quasi-norm, not entropy. Under a sum-to-one softmax
    the entropy has a degenerate minimum at one-hot that is reachable regardless
    of how badly the data is fitted, and with any appreciable weight the fit
    collapses to a single endmember everywhere - measured, entropy fell 2.08 to
    0.00 while the reconstruction went spatially constant. L1/2 rewards sparsity
    without that free lunch, since it still has to explain the observations.

    Total variation keeps abundances piecewise smooth, suppressing the speckle
    a per-scene fit would otherwise chase.

    The range term replaces the forward-pass clamp: it discourages values
    outside [0,1] while keeping a usable gradient when they stray.
    """
    a = out["abundance"]
    sparse = torch.sqrt(a.clamp_min(1e-8)).sum(dim=1).mean()        # L1/2
    tv = ((a[..., 1:] - a[..., :-1]).abs().mean()
          + (a[..., 1:, :] - a[..., :-1, :]).abs().mean())
    x = out["out"]
    rng = (F.relu(x - 1.0) + F.relu(-x)).mean()
    loss = cfg.w_sparse * sparse + cfg.w_tv * tv + cfg.w_range * rng
    return loss, {"sparse": float(sparse.item()), "tv": float(tv.item()),
                  "range": float(rng.item())}


def build_model(cfg) -> ZeroFusion:
    return ZeroFusion(cfg)
