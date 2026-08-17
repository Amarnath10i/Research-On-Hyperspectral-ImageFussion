"""Experiment configuration — DAETF-Net v3.

Every path and channel count defaults to None and is resolved by inspecting the
filesystem, so nothing about a particular machine or Kaggle dataset slug is
baked into the code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from .io_utils import discover_dataset, infer_channels


@dataclass
class Config:
    # --- data (None => auto-discover) --------------------------------------
    source_root: Optional[str] = None    # domain the model trains on
    target_root: Optional[str] = None    # unseen domain used for transfer tests
    bands: Optional[int] = None
    msi_bands: Optional[int] = None
    scale: int = 4                       # super-resolution factor
    patch: int = 96                      # HR training patch (multiple of scale)

    # --- model --------------------------------------------------------------
    width: int = 64                      # main feature width
    equi_width: int = 16                 # per-orientation width in the p4 stem
    equi_depth: int = 2                  # number of p4 -> p4 group convolutions
    rank: int = 16                       # Tucker ranks (R1 = R2 = R3)
    bp_iters: int = 3                    # back-projection refinement steps
    experts: int = 4                     # number of semantic experts (fixed: 4)
    topk: int = 2                        # top-k sparse routing per pixel
    code_dim: int = 128                  # degradation code width
    blur_ksize: int = 9                  # support of the simulated blur kernels

    # --- degradation simulation (domain randomisation) ----------------------
    sigma_range: Tuple[float, float] = (0.6, 2.4)
    aniso: float = 0.5                   # probability of an anisotropic kernel
    noise_range: Tuple[float, float] = (0.0, 0.03)
    srf_jitter: float = 0.35             # probability of a jittered synthetic MSI
    eval_sigma: float = 1.2              # fixed kernel used to build the eval LR

    # --- optimisation --------------------------------------------------------
    iters: int = 90000                   # ~9h on P100 @ 2.8 it/s (batch=12)
    batch: int = 12                      # safe for P100 16GB; T4x2 can use 16
    lr: float = 2e-4
    min_lr: float = 1e-6
    warmup: int = 2000                   # longer warmup for 90k run
    grad_clip: float = 1.0
    amp: bool = True                     # fp16: halves memory, faster on sm_70+
    grad_accum: int = 2                  # effective batch = batch * grad_accum
    ema_decay: float = 0.999             # EMA decay rate for model weights
    n_restarts: int = 3                  # cosine warm restarts over 90k iters
    workers: int = 2
    seed: int = 42
    cache_limit: int = 12                # scenes held in RAM per loader worker

    # --- loss weights --------------------------------------------------------
    w_char: float = 1.0
    w_sam: float = 0.50
    w_grad: float = 0.30
    w_ssim: float = 0.25
    w_spat: float = 0.50                 # || Down(Y) - LR ||
    w_spec: float = 0.50                 # || SRF(Y)  - MSI ||
    w_bal: float = 0.01                  # MoE balance loss
    w_rank: float = 1e-4                 # Tucker nuclear norm
    w_deg: float = 0.05                  # degradation regression
    w_mmd: float = 0.10                  # MMD domain alignment
    w_specgrad: float = 0.15             # spectral gradient loss

    # --- ablation switches (all modules on by default) ----------------------
    use_equivariant: bool = True
    use_tsse: bool = True
    use_moe: bool = True
    use_fdrm: bool = True
    use_backprojection: bool = True
    use_physics: bool = True
    use_degradation_code: bool = True
    use_disagreement: bool = True        # v3: spectral disagreement field
    use_nullspace: bool = True           # v4: range/null decomposition

    # --- projective spectral embedding (v5: the headline idea) ---------------
    # The illumination-invariant, SAM-metric-aligned manifold.  Per-pixel
    # spectra are normalised to the unit sphere (intensity factored out) and
    # mapped through a calibrated spectral MLP so that L2 in the manifold
    # tracks spectral angle.  This is what makes training optimise the metric
    # that actually fails under domain shift.
    embed_dim: int = 16
    embed_hidden: int = 32
    embed_layers: int = 3
    use_projective_embed: bool = True
    w_embed: float = 0.50              # manifold fidelity ||phi(y)-phi(gt)||^2
    w_cal: float = 0.10                # metric calibration of the manifold

    # --- range/null decomposition (v4) --------------------------------------
    # Defaults measured in nullspace.py, not chosen by taste:
    #   cg_steps 1/2/4/8/32 -> consistency 5.8e-1 / 2.0e-1 / 8.5e-3 / 6.2e-6 / 1.9e-6
    #   ridge 0/1e-6/1e-4/1e-2 -> 1.9e-6 / 5.3e-5 / 5.1e-3 / 4.3e-1
    cg_steps: int = 8                    # CG iterations for the pseudo-inverse
    ridge: float = 1e-6                  # conditioning of (D D^T + ridge I)

    # --- bookkeeping ---------------------------------------------------------
    out_dir: str = "./daetf_out"
    val_every: int = 2000
    log_every: int = 200
    val_scenes: int = 8

    def __post_init__(self) -> None:
        assert self.patch % self.scale == 0, "patch must be divisible by scale"
        assert self.topk <= self.experts, "topk cannot exceed the number of experts"

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
                print(f"[config] inferred bands={self.bands} msi_bands={self.msi_bands}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def paper_core(cls, **overrides) -> "Config":
        """Focused configuration for the primary research hypothesis.

        The paper claim should be tested first with a compact model:
        degradation-conditioned HSI--MSI fusion whose learned residual is
        restricted to the null space of the imaging operator.  The optional
        equivariant/Tucker/MoE/wavelet modules remain available as *separate*
        ablations, rather than being presented as inseparable novelty.
"""
        values = dict(
            use_nullspace=True,
            use_backprojection=False,
            use_equivariant=False,
            use_tsse=False,
            use_moe=False,
            use_fdrm=False,
            use_degradation_code=True,
            use_disagreement=False,
            use_physics=True,
            use_projective_embed=True,
        )
        values.update(overrides)
        return cls(**values)
