"""Experiment configuration - GraphDIP (per-scene physics-only prior)."""

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

    # --- graph -----------------------------------------------------------------
    n_seg: int = 64                   # superpixels (nodes)
    graph_k: int = 6                  # kNN neighbours per node
    hidden: int = 32
    n_layers: int = 2
    mix_type: str = "attention"       # linear | nonlinear | attention
    spatial_weight: float = 1.0

    # --- per-scene DIP -----------------------------------------------------------
    dip_steps: int = 1500
    dip_lr: float = 1e-2
    use_tv: bool = False              # total-variation prior on the fused image
    tv_weight: float = 1e-3
    n_restarts: int = 3

    # --- observation model (protocol parity) -------------------------------------
    blur_ksize: int = 9
    sigma_range: Tuple[float, float] = (0.6, 2.4)
    aniso: float = 0.5
    noise_range: Tuple[float, float] = (0.0, 0.03)
    srf_jitter: float = 0.35
    eval_sigma: float = 1.2

    # --- bookkeeping -----------------------------------------------------------
    out_dir: str = "./graphdip_out"
    val_scenes: int = 8
    workers: int = 2
    seed: int = 42
    name: str = "graphdip"

    def __post_init__(self) -> None:
        assert self.patch % self.scale == 0, "patch must be divisible by scale"
        assert self.mix_type in ("linear", "nonlinear", "attention")

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