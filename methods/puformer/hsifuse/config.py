"""Command-line options shared by train.py, test.py and eval_gaps.py: datasets, models, ablations, operators."""
from __future__ import annotations

from . import metrics
from .ops import Degradation, LearnedDegradation, dataset_srf, dataset_wl, sensor_srf

DATASETS = ["chikusei", "pavia", "cave", "harvard", "neon"]
MODELS = ["puformer", "ssrnet", "psrt", "dct", "mimformer", "fusformer", "dhif", "smgu", "clsnet", "otpnet", "u2net", "mimosst",
          "caformer"]


def add_ablation_args(p):
    g = p.add_argument_group("PUFormer ablations (defaults = the full model)")
    g.add_argument("--physics", type=int, default=1, help="0: no data-consistency steps and no residual input")
    g.add_argument("--res_input", type=int, default=1, help="0: the prior does not get s^2 Dt(D Z - Y_H)")
    g.add_argument("--memory", type=int, default=1, help="0: no cross-stage memory")
    g.add_argument("--mixer", default="mdta", choices=["mdta", "window", "conv"], help="token mixer of the prior")
    g.add_argument("--op", default="exact", choices=["exact", "learned"],
                   help="learned: D (blur kernel) and R are learned with the network instead of the known operators")
    g.add_argument("--dstep", default="grad", choices=["grad", "cg", "kron", "kronlearn"],
                   help="data step: one gradient step, the exact proximal step by --cg_iters conjugate-gradient "
                   "iterations, the same step in closed form through the Kronecker eigendecomposition of the "
                   "separable exact operators, or that step with learned (fixed-point preserving) spectral gains")
    g.add_argument("--cg_iters", type=int, default=5)
    g.add_argument("--mres", type=int, default=0, help="1: the prior also gets the HR-MSI residual back-projected "
                   "through the spectral response, (B/M) Rt(R Z - Y_M)")
    g.add_argument("--omc", type=int, default=0, help="1: observation-model conditioning - the data-step sizes and "
                   "the prior (FiLM) depend on the operator's blur width and SRF shift and on the noise level "
                   "estimated from each observation")
    g.add_argument("--nsp", type=int, default=0, help="1: null-space-gated prior - the prior's correction is applied "
                   "exactly in the fusion null space and with a learned (noise-dependent with --omc) weight in the "
                   "observed subspace (needs a separable exact operator)")
    g.add_argument("--ntp", type=int, default=0, help="1: null-space transfer prior - the LR-HSI enters the U-Net "
                   "bottleneck and a per-pixel spectral transfer branch maps the MSI's fine detail to the other bands")
    g.add_argument("--nomsi", type=int, default=0, help="1: input ablation without the HR-MSI (single-image "
                   "super-resolution of the LR-HSI with the same network and data step)")
    g.add_argument("--nohsi", type=int, default=0, help="1: input ablation without the LR-HSI (spectral "
                   "super-resolution of the HR-MSI with the same network and data step)")
    g.add_argument("--prior", default="unet", choices=["unet", "eon"], help="prior network: the transposed-attention "
                   "U-Net, or the eigen-operator network (global mixing in the operator's eigenbasis, eigenvalue-"
                   "conditioned filters, no down-sampling)")
    g.add_argument("--unc", type=int, default=0, help="1: observability-aware uncertainty head - predicts the size of "
                   "the error left in the fusion null space (trained with --w_unc on the projected, detached error)")
    g.add_argument("--cwd", type=int, default=0, help="1: confidence-weighted data step - small networks down-weight "
                   "measurements that disagree locally with the estimate (changes, clouds, stripes; train with "
                   "--rand_corrupt)")
    g.add_argument("--sa", type=int, default=0, help="1: sensor-independent - the network receives the HR-MSI only "
                   "as R^+ Y_M in the hyperspectral bands (and, for PUFormer, through its exact data steps), so one "
                   "network serves any MSI sensor (train with --rand_srf)")
    g.add_argument("--sensor", default="", help="HR-MSI sensor response of the protocol (hsifuse.ops.SENSORS; "
                   "default: the dataset's own, WorldView-2 for Chikusei, IKONOS for Pavia Centre)")
    g.add_argument("--reg", type=int, default=0, help="1: the observation code also carries the column blur width "
                   "and the sub-pixel offsets of the LR-HSI grid (operator family with anisotropic blur and "
                   "misregistration; train with --rand_aniso / --rand_offset)")


def blur_of(scale: int) -> tuple[int, float]:
    """Gaussian blur of the protocol at a scale factor: a (2s-1) x (2s-1) kernel with sigma = s / 2, i.e. the
    7 x 7, sigma = 2 blur of TIP'26 at x4; the kernel grows with the factor so that x8 / x16 stay band-limited."""
    return 2 * scale - 1, scale / 2.0


def make_deg(a, sigma: float | None = None, shift_nm: float = 0.0, sigma_c: float | None = None,
             off: tuple = (0.0, 0.0), srf=None, scale: int | None = None) -> Degradation:
    """The true (simulation) operator of the protocol; `sigma` overrides the blur width (robustness study), `sigma_c`
    the width along columns (anisotropic blur), `off` the sub-pixel offsets of the LR-HSI grid (misregistration),
    `srf` the spectral response (another MSI sensor) and `scale` the resolution ratio (with its own blur)."""
    s = a.scale if scale is None else scale
    k, s0 = blur_of(s)
    sig = s0 if sigma is None else sigma
    if srf is None:
        srf = sensor_srf(a.sensor, dataset_wl(a.dataset)) if getattr(a, "sensor", "") else \
            dataset_srf(a.dataset, shift_nm=shift_nm)
    return Degradation(srf, scale=s, ksize=k, sigma=sig, pad=a.pad, shift_nm=shift_nm,
                       sigma_c=sig if sigma_c is None else sigma_c, off=off)


def build_model(a, deg: Degradation):
    """-> model whose operator is `deg`, or a learned operator for --op learned."""
    from .models import build
    if a.model != "puformer":
        return build(a.model, deg, sa=bool(getattr(a, "sa", 0)))
    op = deg
    if getattr(a, "op", "exact") == "learned":
        op = LearnedDegradation(*deg.srf.shape, scale=a.scale, ksize=deg.ksize)
    return build("puformer", op, width=a.width, stages=a.stages, physics=bool(a.physics),
                 res_input=bool(a.res_input), memory=bool(a.memory), mixer=a.mixer, omc=bool(getattr(a, "omc", 0)),
                 dstep=getattr(a, "dstep", "grad"), cg_iters=getattr(a, "cg_iters", 5), mres=bool(getattr(a, "mres", 0)),
                 reg=bool(getattr(a, "reg", 0)), sa=bool(getattr(a, "sa", 0)), nsp=bool(getattr(a, "nsp", 0)),
                 ntp=bool(getattr(a, "ntp", 0)), cwd=bool(getattr(a, "cwd", 0)), unc=bool(getattr(a, "unc", 0)),
                 prior=getattr(a, "prior", "unet"), nomsi=bool(getattr(a, "nomsi", 0)),
                 nohsi=bool(getattr(a, "nohsi", 0)))


def set_metric_scale(a, dn_peak: float):
    metrics.DN_SCALE = dn_peak
    metrics.RATIO = a.scale
    metrics.Q2N_SCALE = 65535.0 if a.dataset in ("cave", "harvard") else None
