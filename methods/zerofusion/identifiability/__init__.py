"""P4 (capstone) identifiability phase diagram.

Two complementary views of how identifiable the HSI-MSI fusion problem is,
driven by the same knobs (scene rank, MSI band count, SRF overlap, noise):

    score     = r_id_hat / r     (P2: spectral DOF the observations pin down)
    null_frac = ||P_N X|| / ||X|| (P1: fraction of the scene left ambiguous)

The phase diagram maps (M, SNR) and (SRF width, SNR) onto
Identifiable / Weakly / Non-identifiable.
"""

from . import selfcheck as _selfcheck
from . import simulator
from .simulator import regime, simulate

__all__ = ["simulate", "regime", "simulator"]