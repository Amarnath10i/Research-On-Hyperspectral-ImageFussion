"""P1 (priority-2 paper): identifiability-aware fusion.

Non-neural scaffold: the combined observation operator A=[D;R] and its joint
range/null-space projector (extending proposal5's spatial-only RangeNullProjector),
plus the observable/ambiguous decomposition, hallucination metric and ambiguity
map.  The learned admissible manifold M_theta subset N(A) is built on top of
this scaffold in a later stage.
"""

from . import selfcheck as _selfcheck
from . import metrics, operator
from .operator import CombinedOperator

__all__ = ["CombinedOperator", "metrics", "operator"]