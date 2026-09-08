"""DACF configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """DACF training / evaluation configuration.

    Works for any band count — set bands per dataset.
    """

    # Data
    dataset: str = "paviau"       # "paviau" or "chikusei"
    bands: int = 103              # auto-detected at runtime
    msi_bands: int = 3
    scale: int = 4
    patch_size: int = 64

    # Degradation encoder
    enc_hidden: int = 32
    code_dim: int = 64

    # Flow network
    flow_hidden: int = 64
    flow_layers: int = 8
    flow_steps: int = 4           # ODE steps at inference

    # Phase 1: flow matching
    p1_iters: int = 20000
    p1_batch: int = 16
    p1_lr: float = 2e-4
    p1_warmup: int = 500
    p1_sigma_range: tuple = (0.5, 3.0)   # blur sigma randomization
    p1_noise_range: tuple = (0.0, 0.05)  # noise randomization

    # Phase 2: self-supervised adaptation
    p2_iters: int = 100
    p2_lr: float = 1e-3
    p2_lambda_cycle: float = 1.0
    p2_lambda_srf: float = 0.5

    # Validation
    val_every: int = 2000
    val_scenes: int = 8
    log_every: int = 100

    # Loss weights (phase 1)
    lambda_flow: float = 1.0
    lambda_consistency: float = 0.5
    lambda_srf: float = 0.3
    lambda_spectral: float = 0.1

    # Paths
    output_dir: str = "dacf_out"
