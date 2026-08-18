"""Shared experiment configuration.

`BaseConfig` holds everything the evaluation protocol depends on - the scale
factor, the degradation, the metric settings, the optimiser schedule. Every
proposal subclasses it and adds only its own architectural fields.

That split is the point: because all four proposals inherit the same protocol
fields and the same defaults, a difference in their results is attributable to
the architecture rather than to one of them quietly evaluating at a different
scale factor or normalisation - which is exactly how the ten published
baselines in `existing/` became incomparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from .io_utils import discover_dataset, infer_channels


@dataclass
class BaseConfig:
    # --- data (None => auto-discover) --------------------------------------
    source_root: Optional[str] = None
    target_root: Optional[str] = None
    bands: Optional[int] = None
    msi_bands: Optional[int] = None

    # --- protocol: identical for every proposal ----------------------------
    scale: int = 4
    patch: int = 64
    blur_ksize: int = 9
    eval_sigma: float = 1.2
    sigma_range: Tuple[float, float] = (0.6, 2.4)
    aniso: float = 0.5
    noise_range: Tuple[float, float] = (0.0, 0.03)
    srf_jitter: float = 0.35

    # --- optimisation --------------------------------------------------------
    iters: int = 2000
    batch: int = 16
    lr: float = 2e-4
    min_lr: float = 1e-6
    warmup: int = 200
    grad_clip: float = 1.0
    amp: bool = True
    workers: int = 2
    seed: int = 42
    cache_limit: int = 12

    # --- shared loss weights -------------------------------------------------
    w_char: float = 1.0
    w_sam: float = 0.30
    w_grad: float = 0.20
    w_ssim: float = 0.15
    w_spat: float = 0.50
    w_spec: float = 0.50
    use_physics: bool = True

    # --- bookkeeping ---------------------------------------------------------
    out_dir: str = "./out"
    val_every: int = 1000
    log_every: int = 100
    val_scenes: int = 4
    name: str = "model"

    def __post_init__(self) -> None:
        assert self.patch % self.scale == 0, "patch must be divisible by scale"

    def resolve(self, source_hints: Sequence[str] = ("cave",),
                target_hints: Sequence[str] = ("harvard",),
                verbose: bool = True) -> "BaseConfig":
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
