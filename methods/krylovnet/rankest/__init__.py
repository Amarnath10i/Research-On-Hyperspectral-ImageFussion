"""P2 rankest: identifiable spectral rank on controlled synthetic scenes.

    import krylovnet.rankest as rankest
    ok = rankest.selfcheck()
"""

from . import selfcheck as _selfcheck
from .generator import RankScene, make_srf
from krylovnet.metrics.identifiable_rank import estimate_ranks

__all__ = ["RankScene", "make_srf", "estimate_ranks"]


def selfcheck(device: str = "cpu") -> bool:
    return _selfcheck.run_all(device)