"""Spectral Lie Transport (SLT): fusion as geodesic transport on the spectral
manifold.

Proposal 1 (Q1 redesign).  Normalised spectra live on the unit sphere
S^{B-1}: SAM *is* the geodesic distance there.  The base point is the
LR-derived spectrum direction; a learned tangent vector is transported through
the spherical exponential map; intensity is carried by the observation.  The
network's only degrees of freedom are geodesic displacements, so illumination
cannot mask spectral error - the root cause behind the cross-domain SAM
collapse reported in proposal1/results/RESULTS.md.

    import slt
    cfg = slt.Config().resolve()
    slt.selfcheck()                 # proves the algebraic/structural claims
    model, hist = slt.train(cfg)
    slt.evaluate_dataset(model, cfg.source_root, cfg)
"""

from . import selfcheck as _selfcheck
from .config import Config
from .engine import (evaluate_dataset, load_checkpoint, set_seed,
                     tiled_inference, train)
from .losses import SLTLoss
from .manifold import (exp_map, geodesic_distance, l2_normalize, log_map,
                       tangent_projection)
from .model import SLTNet

__all__ = ["Config", "SLTNet", "SLTLoss", "train", "evaluate_dataset",
           "tiled_inference", "load_checkpoint", "set_seed",
           "l2_normalize", "tangent_projection", "exp_map", "log_map",
           "geodesic_distance"]


def selfcheck(device: str = "cpu") -> bool:
    return _selfcheck.run_all(device)