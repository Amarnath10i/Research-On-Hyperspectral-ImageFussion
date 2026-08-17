"""Neural Spectral PDE (NSP) - Proposal 3 (Q1 redesign).

Fusion posed as the steady state of a diffusion-reaction PDE:

    du/dt = div(D(u) grad u) - lambda1 D^T(Du - X) - lambda2 S^T(S u - M)

with D(u) a learned **cross-spectral** diffusion tensor (couples adjacent
spectral bands) and the observation terms as soft *penalties* (not hard
constraints -- the honest limitation documented in ARCHITECTURE.md).  The PDE is
discretised with explicit Euler; ``dt`` is the discretisation step and, being a
continuous-space model, only the step adapts to the LR/HR scale.

    import nsp
    cfg = nsp.Config().resolve()
    nsp.selfcheck()                 # adjointness, PDE->fusion, scale transfer
    model, hist = nsp.train(cfg)
"""

from . import selfcheck as _selfcheck
from .config import Config
from .engine import (evaluate_dataset, load_checkpoint, set_seed,
                     tiled_inference, train)
from .losses import NSPLoss
from .model import NSPModel
from .pde import divergence

__all__ = ["Config", "NSPModel", "divergence", "NSPLoss",
           "train", "evaluate_dataset", "tiled_inference", "load_checkpoint",
           "set_seed"]


def selfcheck(device: str = "cpu") -> bool:
    return _selfcheck.run_all(device)