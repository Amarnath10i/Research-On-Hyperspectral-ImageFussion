"""Experiment configuration - Neural Spectral PDE (NSP).

The fusion problem is solved as the steady state of a diffusion-reaction PDE.
The config carries the shared data pipeline fields (proposal1.daetf modules are
duck-typed) plus the PDE switches that define the ladder.
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

    # --- PDE solver ----------------------------------------------------------
    pde_steps: int = 16
    dt: float = 0.1                    # discretisation step (explicit Euler)
    lam1: float = 1.0                  # LR-HSI penalty weight
    lam2: float = 1.0                  # MSI penalty weight
    learn_dt: bool = True
    learn_lam: bool = True
    use_diffusion: bool = True         # off => pure penalty dynamics (s1)
    use_tensor_net: bool = True        # off => isotropic scalar diffusivity
    scale_adaptive_dt: bool = True     # dt *= 1/scale (continuous-PDE step)

    # --- cross-spectral diffusion tensor -------------------------------------
    tensor_hidden: int = 24
    tensor_layers: int = 2
    iso_diff: float = 0.02             # constant diffusivity when tensor off

    # --- observation model (protocol parity) ---------------------------------
    blur_ksize: int = 9
    sigma_range: Tuple[float, float] = (0.6, 2.4)
    aniso: float = 0.5
    noise_range: Tuple[float, float] = (0.0, 0.03)
    srf_jitter: float = 0.35
    eval_sigma: float = 1.2

    # --- optimisation --------------------------------------------------------
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

    # --- loss weights --------------------------------------------------------
    w_phys: float = 1.0
    w_spec: float = 1.0
    w_recon: float = 0.1
    w_res: float = 0.1

    # --- bookkeeping ---------------------------------------------------------
    out_dir: str = "./nsp_out"
    val_every: int = 500
    log_every: int = 200
    val_scenes: int = 8
    name: str = "nsp"

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