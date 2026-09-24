"""ConsistentFlow: one-step null-space-consistent generative HSI-MSI fusion.

Proposal 6.  The fast sibling of Proposal 5 (SpectralFlow): it shares the
spectral score prior, degradation head, and null-space projector, but replaces
the multi-step DDIM sampler with a distilled consistency map, so inference is
one forward pass at ~regressor cost while D(Y_hat) = X still holds
algebraically.
"""

from . import selfcheck as _selfcheck
from .config import Config
from .engine import (ConsistencyModel, build_model, evaluate_dataset,
                     load_checkpoint, train)
from .sampler import ConsistencySampler

__all__ = ["Config", "ConsistencyModel", "ConsistencySampler",
           "build_model", "train", "evaluate_dataset", "load_checkpoint"]


def selfcheck(device: str = "cpu") -> bool:
    return _selfcheck.run_all(device)
