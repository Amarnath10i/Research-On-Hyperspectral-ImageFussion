"""Experiment configuration - KrylovNet.

The fusion problem is posed as the normal equation of the two observation
models and solved by an unrolled GMRES-style Krylov solver.  The config carries
the shared data pipeline fields (the proposal1.daetf modules are duck-typed)
plus the solver switches that define the ladder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from proposal1.daetf.io_utils import discover_dataset, infer_channels


@dataclass
class Config:
    # --- data (None => auto-discover) --------------------------------------
    source_root: Optional[str] = None
    target_root: Optional[str] = None
    bands: Optional[int] = None
    msi_bands: Optional[int] = None
    scale: int = 4
    patch: int = 96

    # --- solver -------------------------------------------------------------
    n_stages: int = 6                    # unrolled solver stages
    rho: float = 1e-3                    # ridge in A = D^T D + S^T S + rho I
    rich_alpha: float = 0.1              # fixed step for the stage-1 baseline
    use_krylov: bool = True              # False => Richardson/fixed-point
    use_learned_combo: bool = True       # attention blend over the basis
    use_precond: bool = True             # spectral-graph GNN preconditioner
    use_hypernet: bool = False           # condition-adaptive stage gating

    # --- learned proximal prior (the capacity that makes this competitive) ---
    # The solver alone is Tikhonov least squares with no image prior and scores
    # below bicubic. These control the plug-and-play denoiser between stages.
    use_prior: bool = True
    prior_width: int = 96                # SOTA push: was 64
    prior_blocks: int = 8                # SOTA push: was 4
    n_outer: int = 4                     # data/prior alternations

    # --- spectral graph preconditioner --------------------------------------
    graph_k: int = 4                     # kNN edges in the band graph
    hidden: int = 32
    gcn_layers: int = 2

    # --- observation model (protocol parity with P1) -------------------------
    blur_ksize: int = 9
    sigma_range: Tuple[float, float] = (0.6, 2.4)
    aniso: float = 0.5
    noise_range: Tuple[float, float] = (0.0, 0.03)
    srf_jitter: float = 0.35
    eval_sigma: float = 1.2

    # --- optimisation --------------------------------------------------------
    # Published CAVE x4 results (FeINFN 52.47, BDT 52.30) come from 1e5-1e6
    # iterations. The previous default of 2000 - whose own comment said "scale
    # up for full convergence" - is 50-500x short, and produced 40.85 dB.
    # This is the value to run for a headline number; drop it for debugging.
    iters: int = 100000
    batch: int = 12
    lr: float = 2e-4
    min_lr: float = 1e-6
    warmup: int = 2000
    grad_clip: float = 1.0
    amp: bool = True
    grad_accum: int = 2
    ema_decay: float = 0.999
    n_restarts: int = 3
    workers: int = 2
    seed: int = 42
    cache_limit: int = 12

    # --- loss weights --------------------------------------------------------
    w_phys: float = 1.0                  # ||D(ŷ) - X||^2
    w_spec: float = 1.0                  # ||S(ŷ) - M||^2
    # Physics consistency is satisfied by an entire family of solutions - that
    # is the whole point of the null-space view - so weighting it 10x above
    # fidelity gave the model little pressure to pick the right member.
    w_recon: float = 1.0                 # L1 vs ground truth (was 0.1)
    w_res: float = 0.1                   # final GMRES residual (regulariser)

    # --- bookkeeping ---------------------------------------------------------
    out_dir: str = "./krylovnet_out"
    val_every: int = 500
    log_every: int = 200
    val_scenes: int = 8
    name: str = "krylovnet"

    def __post_init__(self) -> None:
        assert self.patch % self.scale == 0, "patch must be divisible by scale"

    def resolve(self, source_hints: Sequence[str] = ("cave",),
                target_hints: Sequence[str] = ("harvard",),
                verbose: bool = True) -> "Config":
        """Fill in any field still set to None. Idempotent."""
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
                print(f"[config] inferred bands={self.bands} "
                      f"msi_bands={self.msi_bands}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)