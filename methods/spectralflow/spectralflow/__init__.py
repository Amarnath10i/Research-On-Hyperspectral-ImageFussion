"""SpectralFlow: null-space-consistent score-based HSI-MSI fusion.

The observation X = D(Y) fixes the range component D_pinv(X) in closed form.
SpectralFlow generates the remaining null-space component with a
multispectral-guided score-based prior, and re-projects onto the consistent set
D_pinv(X) + P_perp(.) at every reverse step, so D(Y_hat) = X holds by
construction.  See docs/ARCHITECTURE.md for the full paper design.
"""

from .config import Config
from .engine import SamplingModel, build_model, evaluate_dataset, train
from . import nullspace
from . import sampler
from . import score
from . import selfcheck as _selfcheck

__all__ = [
    "Config", "SamplingModel", "build_model", "evaluate_dataset", "train",
    "nullspace", "sampler", "score", "selfcheck",
]


def selfcheck(device: str = "cpu") -> bool:
    return _selfcheck.run_all(device)