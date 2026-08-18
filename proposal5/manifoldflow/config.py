"""Experiment configuration - ManifoldFlow (rectified flow on the consistent set)."""

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

    # --- flow ------------------------------------------------------------------
    hidden: int = 32
    n_flow_blocks: int = 3
    tangent: bool = True              # project velocity onto null(D) (s2+)
    straightness_reg: bool = False    # extra t-invariance loss (s3)
    sample_steps: int = 8             # Euler steps at inference (s4 uses 4)
    cg_steps: int = 8                 # CG solve inside the projector
    cg_ridge: float = 1e-6

    # --- observation model (protocol parity) -------------------------------------
    blur_ksize: int = 9
    sigma_range: Tuple[float, float] = (0.6, 2.4)
    aniso: float = 0.5
    noise_range: Tuple[float, float] = (0.0, 0.03)
    srf_jitter: float = 0.35
    eval_sigma: float = 1.2

    # --- optimisation ------------------------------------------------------------
    iters: int = 2000                    # ~1.2h on P100; scale up for full convergence
    batch: int = 12
    lr: float = 2e-4
    min_lr: float = 1e-6
    warmup: int = 200
    grad_clip: float = 1.0
    amp: bool = True
    grad_accum: int = 2
    ema_decay: float = 0.999
    n_restarts: int = 1
    workers: int = 2
    seed: int = 42
    cache_limit: int = 12

    # --- loss weights ------------------------------------------------------------
    w_flow: float = 1.0               # velocity regression ||v - u||^2
    w_straight: float = 0.1           # straightness regulariser
    w_recon: float = 0.05             # L1 vs ground truth (auxiliary)

    # --- bookkeeping --------------------------------------------------------------
    out_dir: str = "./manifoldflow_out"
    val_every: int = 500
    log_every: int = 200
    val_scenes: int = 8
    name: str = "manifoldflow"

    def __post_init__(self) -> None:
        assert self.patch % self.scale == 0, "patch must be divisible by scale"

    def resolve(self, source_hints: Sequence[str] = ("cave",),
                target_hints: Sequence[str] = ("harvard",),
                verbose: bool = True) -> "Config":
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