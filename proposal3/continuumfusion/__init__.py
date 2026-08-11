"""ContinuumFusion - arbitrary-scale HSI-MSI fusion by implicit representation.

Proposal 3. Proposals 1 and 2 are welded to one scale factor; this one treats
resolution as a query parameter, so a single set of weights serves x4 through
x32 - which is what makes a like-for-like comparison across factors possible
at all.

    import continuumfusion as C
    cfg = C.Config().resolve()
    model, hist = C.train(cfg)
    out = model(lr, msi, out_hw=(1024, 1024))["out"]   # any resolution
"""

from dataclasses import dataclass, field
from typing import Tuple

from hsifusion import BaseConfig, FusionLoss
from hsifusion import engine as _engine

from .model import (ContinuumFusion, DetailEncoder, Encoder, SpectralCoordMLP,
                    build_model, make_coord)

__version__ = "1.0.0"


@dataclass
class Config(BaseConfig):
    name: str = "continuumfusion"
    width: int = 64            # LR latent width
    enc_depth: int = 4
    detail_width: int = 32     # HR MSI detail width
    mlp_width: int = 128
    mlp_depth: int = 4
    band_dim: int = 24         # spectral coordinate embedding
    unfold: bool = True        # 3x3 latent neighbourhood as decoder context
    # scale factors sampled during training. Randomising the factor is what
    # makes the resolution ratio a nuisance variable rather than a constant.
    train_scales: Tuple[int, ...] = (2, 3, 4, 6, 8)
    out_dir: str = "./continuum_out"


def build_loss(cfg, srf):
    import torch
    return FusionLoss(cfg, torch.from_numpy(srf))


def train(cfg, device: str = "cuda", **kw):
    return _engine.train(cfg, build_model=build_model, build_loss=build_loss,
                         device=device, **kw)


def evaluate_scales(model, root, cfg, scales=(4, 8, 16, 32), device="cuda",
                    limit=None, verbose=True):
    """Score one trained model at several scale factors.

    This is the experiment the grid-based proposals cannot run without
    retraining, and the one that makes the x4/x8/x16/x32 confusion in
    `existing/` addressable.
    """
    import copy

    from hsifusion.engine import evaluate_dataset

    rows = []
    for s in scales:
        c = copy.deepcopy(cfg)
        c.scale = s
        c.patch = max(c.patch, s * 8)
        if verbose:
            print(f"\n--- scale x{s} ---")
        m = evaluate_dataset(model, root, c, "Test", device, limit=limit,
                             verbose=verbose)
        rows.append({"scale": s, **m})
    return rows


__all__ = ["Config", "ContinuumFusion", "build_model", "build_loss", "train",
           "evaluate_scales", "Encoder", "DetailEncoder", "SpectralCoordMLP",
           "make_coord", "__version__"]
