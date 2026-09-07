"""ASON configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """ASON training / evaluation configuration."""

    # Data
    dataset: str = "chikusei"
    scale: int = 4
    patch_size: int = 64
    bands: int = 128
    msi_bands: int = 3

    # Model
    hidden: int = 24
    n_blocks: int = 3
    code_dim: int = 64
    sample_steps: int = 4
    cg_steps: int = 8
    eval_sigma: float = 1.2

    # Training
    batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-5
    iters: int = 30000
    warmup: int = 1000
    val_every: int = 3000
    log_every: int = 200
    val_scenes: int = 4

    # Loss weights
    lambda_velocity: float = 1.0
    lambda_l1: float = 0.05
    lambda_consistency: float = 0.5
    lambda_spectral: float = 0.1

    # Paths
    output_dir: str = "ason_out"
    best_ckpt: str = "ason_out/best.pth"

    @property
    def device(self) -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
