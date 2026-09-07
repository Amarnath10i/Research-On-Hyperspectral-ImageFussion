"""KrylovNet: differentiable Krylov-subspace fusion with a learned spectral
graph preconditioner.

Proposal 2 (Q1 redesign).  Instead of unrolling ADMM (the old UnfoldFusion), we
unroll a GMRES-like Krylov solver for the fusion normal equation
A x = b,  A = D^T D + S^T S + rho I,  b = D^T X + S^T M, where D is the
LR-HSI operator and S the SRF-to-MSI operator.  Each stage grows the Krylov
basis by one vector; a learned attention blend adjusts the GMRES combination,
and a GNN over the spectral band graph builds a per-scene preconditioner.

    import krylovnet
    cfg = krylovnet.Config().resolve()
    krylovnet.selfcheck()               # adjointness, exact solve, precond
    model, hist = krylovnet.train(cfg)
"""

from . import selfcheck as _selfcheck
from .config import Config
from .engine import (evaluate_dataset, load_checkpoint, set_seed,
                     tiled_inference, train)
from .losses import KrylovLoss
from .model import KrylovNet, SpectralPreconditioner
from .solver import FusionOperator, krylov_gmres, richardson_solve

__all__ = ["Config", "KrylovNet", "SpectralPreconditioner", "FusionOperator",
           "krylov_gmres", "richardson_solve", "KrylovLoss",
           "train", "evaluate_dataset", "tiled_inference", "load_checkpoint",
           "set_seed"]


def selfcheck(device: str = "cpu") -> bool:
    return _selfcheck.run_all(device)