"""UnfoldFusion - deep-unfolded variational solver for HSI-MSI fusion.

Proposal 2. Where DAETF-Net (proposal 1) is a feed-forward network told about
the physics through the loss, this is an optimisation algorithm whose
iterations are unrolled into layers, so the observation model is structural.

    import unfoldfusion as U
    cfg = U.Config().resolve()
    model, hist = U.train(cfg)
"""

from dataclasses import dataclass

from hsifusion import BaseConfig, FusionLoss
from hsifusion import engine as _engine

from .model import (BlurDecimate, NoiseConditionedDenoiser, SpectralProject,
                    UnfoldFusion, build_model, deep_supervision)

__version__ = "1.0.0"


@dataclass
class Config(BaseConfig):
    """Protocol fields are inherited unchanged so the comparison stays fair."""
    name: str = "unfoldfusion"
    stages: int = 6            # unrolled HQS iterations
    cg_steps: int = 4          # conjugate-gradient steps per data step
    rank: int = 10             # spectral subspace dimension
    width: int = 64            # denoiser width
    denoise_depth: int = 4
    w_stage: float = 0.20      # deep supervision on intermediate stages
    out_dir: str = "./unfold_out"


def build_loss(cfg, srf):
    """Shared fidelity+physics objective plus per-stage deep supervision."""
    import torch

    def extra(out, target, cfg=cfg, supervised=True, **kw):
        return deep_supervision(out, target, cfg, supervised)

    loss = FusionLoss(cfg, torch.from_numpy(srf),
                      extra_terms=lambda **kw: extra(**kw))
    return loss


def train(cfg, device: str = "cuda", **kw):
    """Train, wiring the measured SRF into the model's projection operator."""
    import numpy as np
    import torch
    from hsifusion.data import estimate_srf

    def factory(c):
        m = build_model(c)
        srf = estimate_srf(c.source_root, "Train", c)
        m.set_srf(torch.from_numpy(srf))
        return m

    return _engine.train(cfg, build_model=factory, build_loss=build_loss,
                         device=device, **kw)


__all__ = ["Config", "UnfoldFusion", "build_model", "build_loss", "train",
           "BlurDecimate", "SpectralProject", "NoiseConditionedDenoiser",
           "deep_supervision", "__version__"]
