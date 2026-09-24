"""ManifoldFlow - Proposal 5 (Q1 redesign).

Rectified flow matching for HSI fusion with a *tangent-space manifold
constraint*: the velocity field is projected onto the null space of the
observation operator at every step, so every iterate stays on the consistent
set D(y) = X (an algebraic identity, not a penalty).  Because the flow is
rectified, few Euler steps suffice (the "~10x fewer steps" claim).

    import manifoldflow
    cfg = manifoldflow.Config().resolve()
    manifoldflow.selfcheck()      # tangent, flow matching, straightness, consistency
    model, hist = manifoldflow.train(cfg)
"""

from . import selfcheck as _selfcheck
from .config import Config
from .engine import (evaluate_dataset, load_checkpoint, set_seed,
                     tiled_inference, train)
from .losses import FlowLoss
from .model import ManifoldFlow, VelocityNet

__all__ = ["Config", "ManifoldFlow", "VelocityNet", "FlowLoss",
           "train", "evaluate_dataset", "tiled_inference", "load_checkpoint",
           "set_seed"]


def selfcheck(device: str = "cpu") -> bool:
    return _selfcheck.run_all(device)