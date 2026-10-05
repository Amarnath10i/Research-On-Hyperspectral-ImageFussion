"""Command-line options shared by train.py, test.py and eval_gaps.py: datasets, models, ablations, operators."""
from __future__ import annotations

from . import metrics
from .ops import Degradation, LearnedDegradation, dataset_srf

DATASETS = ["chikusei", "pavia", "cave", "harvard"]
MODELS = ["puformer", "ssrnet", "psrt", "dct", "mimformer", "fusformer", "dhif"]


def add_ablation_args(p):
    g = p.add_argument_group("PUFormer ablations (defaults = the full model)")
    g.add_argument("--physics", type=int, default=1, help="0: no data-consistency steps and no residual input")
    g.add_argument("--res_input", type=int, default=1, help="0: the prior does not get s^2 Dt(D Z - Y_H)")
    g.add_argument("--memory", type=int, default=1, help="0: no cross-stage memory")
    g.add_argument("--mixer", default="mdta", choices=["mdta", "window", "conv"], help="token mixer of the prior")
    g.add_argument("--op", default="exact", choices=["exact", "learned"],
                   help="learned: D (blur kernel) and R are learned with the network instead of the known operators")


def blur_of(scale: int) -> tuple[int, float]:
    """Gaussian blur of the protocol at a scale factor: a (2s-1) x (2s-1) kernel with sigma = s / 2, i.e. the
    7 x 7, sigma = 2 blur of TIP'26 at x4; the kernel grows with the factor so that x8 / x16 stay band-limited."""
    return 2 * scale - 1, scale / 2.0


def make_deg(a, sigma: float | None = None, shift_nm: float = 0.0) -> Degradation:
    """The true (simulation) operator of the protocol; `sigma` overrides the blur width (robustness study)."""
    k, s0 = blur_of(a.scale)
    return Degradation(dataset_srf(a.dataset, shift_nm=shift_nm), scale=a.scale, ksize=k,
                       sigma=s0 if sigma is None else sigma, pad=a.pad)


def build_model(a, deg: Degradation):
    """-> model whose operator is `deg`, or a learned operator for --op learned."""
    from .models import build
    if a.model != "puformer":
        return build(a.model, deg)
    op = deg
    if getattr(a, "op", "exact") == "learned":
        op = LearnedDegradation(*deg.srf.shape, scale=a.scale, ksize=deg.ksize)
    return build("puformer", op, width=a.width, stages=a.stages, physics=bool(a.physics),
                 res_input=bool(a.res_input), memory=bool(a.memory), mixer=a.mixer)


def set_metric_scale(a, dn_peak: float):
    metrics.DN_SCALE = dn_peak
    metrics.RATIO = a.scale
    metrics.Q2N_SCALE = 65535.0 if a.dataset in ("cave", "harvard") else None
