"""Experiment configuration for ConsistentFlow (Proposal 6).

ConsistentFlow is the fast sibling of SpectralFlow (Proposal 5).  It shares
P5's spectral score prior, degradation head, and null-space projector, but
replaces the multi-step DDIM sampler with a **distilled consistency map**:
a network f(y_t, t, M, d) that maps any point on a noise trajectory straight to
the clean endpoint, so inference is one (or a few) forward passes at
~regressor cost while the null-space projection still guarantees D(Y_hat) = X
algebraically.

Only the consistency-distillation fields are new; everything else follows the
shared protocol (same degradation model, metrics, scene splits).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from hsifusion.io_utils import discover_dataset, infer_channels


@dataclass
class Config:
    # --- data (None => auto-discover) --------------------------------------
    source_root: Optional[str] = None    # domain the prior is distilled on
    target_root: Optional[str] = None    # unseen domain for cross-domain tests
    bands: Optional[int] = None
    msi_bands: Optional[int] = None
    scale: int = 4                       # super-resolution factor
    patch: int = 64                      # HR training patch (multiple of scale)

    # --- degradation simulation (shared protocol) ---------------------------
    blur_ksize: int = 9
    eval_sigma: float = 1.2
    sigma_range: Tuple[float, float] = (0.6, 2.4)
    aniso: float = 0.5
    noise_range: Tuple[float, float] = (0.0, 0.03)
    srf_jitter: float = 0.35

    # --- shared network (same spectral prior as P5) ---------------------------
    ch: int = 64                        # base channel width
    ch_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_res: int = 2
    cond_dim: int = 256                 # time + degradation conditioning width
    code_dim: int = 64                  # degradation code width
    dropout: float = 0.0

    # --- consistency model ----------------------------------------------------
    num_timesteps: int = 200            # schedule length for distillation
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ema_decay: float = 0.999            # EMA target network for CD targets

    # --- sampling / consistency ------------------------------------------------
    # The headline: `sample_steps` is 1 by default - one forward pass at
    # inference, the entire point of the proposal.  Use 2-4 for a small
    # quality/speed trade-off study (docs/ARCHITECTURE.md).
    sample_steps: int = 1
    use_projection: bool = True        # False = free one-step generation (control)
    cg_steps: int = 8                   # CG iterations for the pseudo-inverse
    ridge: float = 1e-6

    # --- optimisation -----------------------------------------------------------
    iters: int = 2000                    # ~1.2h on P100; scale up for full convergence
    batch: int = 8
    lr: float = 2e-4
    min_lr: float = 1e-6
    warmup: int = 200
    grad_clip: float = 1.0
    amp: bool = True
    workers: int = 2
    seed: int = 42
    cache_limit: int = 12

    # --- loss weights -----------------------------------------------------------
    w_deg: float = 0.05                 # degradation regression
    w_spec: float = 0.05                # SRF-consistency of the guide at train

    # --- bookkeeping -------------------------------------------------------------
    out_dir: str = "./consistentflow_out"
    val_every: int = 500
    log_every: int = 200
    val_scenes: int = 4
    name: str = "consistentflow"

    def __post_init__(self) -> None:
        assert self.patch % self.scale == 0, "patch must be divisible by scale"

    def resolve(self, source_hints: Sequence[str] = ("cave",),
                target_hints: Sequence[str] = ("harvard",),
                verbose: bool = True) -> "Config":
        """Fill in any field still None by inspecting the filesystem."""
        if self.source_root is None:
            self.source_root = discover_dataset(source_hints, verbose=verbose)
        if self.target_root is None:
            self.target_root = discover_dataset(target_hints, required=False,
                                                verbose=verbose)
        if self.bands is None or self.msi_bands is None:
            b, m = infer_channels(self.source_root)
            self.bands = self.bands or b
            self.msi_bands = self.msi_bands or m
            if verbose:
                print(f"[config] inferred bands={self.bands} msi_bands={self.msi_bands}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
