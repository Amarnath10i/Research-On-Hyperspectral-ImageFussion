"""ASON — Adaptive Spectral Operator Network for HSI-MSI Fusion.

Rectified-flow-on-consistent-set framework with degradation encoding
and FiLM conditioning. Composes verified modules from ManifoldFlow
(velocity flow), SpectralFlow (null-space projection), and DAETF
(degradation encoding concept).
"""

from .config import Config
from .model import ASONNet, VelocityNet, DegradationCode
from .losses import ASONLoss
from .engine import train, evaluate_dataset

__all__ = [
    "Config",
    "ASONNet",
    "VelocityNet",
    "DegradationCode",
    "ASONLoss",
    "train",
    "evaluate_dataset",
]
