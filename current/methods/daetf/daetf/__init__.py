"""DAETF-Net v3: Adaptive Spectral-Causal Routing Network.

Domain-Adaptive Equivariant Tensor Fusion Network — v3 (ASCR).

Hyperspectral-multispectral image fusion built around degradation-conditioned
expert routing rather than a uniform fusion strategy.

    import daetf
    cfg = daetf.Config().resolve()          # finds the datasets, infers bands
    daetf.selfcheck.run_all()               # verifies the mechanisms numerically
    model, hist = daetf.train(cfg)
    daetf.evaluate_dataset(model, cfg.source_root, cfg)
"""

from . import baselines, experiments, selfcheck
from .experiments import PAPER_CORE_ABLATIONS
from .baselines import (BASELINES, evaluate_all_baselines,
                        evaluate_baseline)
from .config import Config
from .data import FusionPatchDataset, SceneCache, estimate_srf
from .degrade import FixedDegradation, blur_downsample, gaussian_kernel2d
from .engine import (cosine_lr, evaluate_dataset, evaluate_with_tta,
                     load_checkpoint, set_seed, test_time_adapt, tiled_inference,
                     train)
from .io_utils import (available_splits, discover_dataset, find_pairs,
                       infer_channels, load_mat, search_roots, to_chw01)
from .losses import (SPCLoss, charbonnier, gradient_loss, mmd_rbf, sam_loss)
from .metrics import (evaluate_arrays, metric_ergas, metric_psnr, metric_sam,
                      metric_ssim, ssim_torch)
from .model import DAETFNet
from .modules import (BackProjectionUpsampler, BicubicUpsampler,
                      ChannelAttention, DegradationConditionedMoE,
                      DegradationEncoder, EquivariantFeatureExtractor, FiLM,
                      FrequencyDomainRefinement, GeometricSelfEnsemble,
                      HaarDWT, P4ConvP4, P4ConvZ2,
                      PlainFeatureExtractor, ResidualDenseBlock,
                      SpectralDisagreementField,
                      TensorSpectralSpatialEncoder)
from .nullspace import RangeNullProjector, decode_degradation_params, kernel_from_params

# Build the SPCLoss for use in TTA outside the engine
from .losses import SPCLoss as _SPCLoss


def build_loss(cfg: Config, srf) -> _SPCLoss:
    """Convenience: build the composite loss from a Config and SRF array."""
    import torch
    import numpy as np
    if isinstance(srf, np.ndarray):
        srf = torch.from_numpy(srf)
    return _SPCLoss(cfg, srf)


__version__ = "3.0.0"
name = "daetf"

__all__ = [
    "Config", "DAETFNet", "SPCLoss", "build_loss",
    "train", "evaluate_dataset", "evaluate_with_tta", "test_time_adapt",
    "tiled_inference", "load_checkpoint", "set_seed", "cosine_lr",
    "FusionPatchDataset", "SceneCache", "estimate_srf",
    "FixedDegradation", "blur_downsample", "gaussian_kernel2d",
    "discover_dataset", "find_pairs", "infer_channels", "available_splits",
    "load_mat", "to_chw01", "search_roots",
    "charbonnier", "sam_loss", "gradient_loss", "mmd_rbf",
    "evaluate_arrays", "metric_psnr", "metric_ssim", "metric_sam",
    "metric_ergas", "ssim_torch",
    "P4ConvZ2", "P4ConvP4", "EquivariantFeatureExtractor",
    "PlainFeatureExtractor", "DegradationEncoder", "FiLM",
    "TensorSpectralSpatialEncoder", "DegradationConditionedMoE",
    "SpectralDisagreementField", "ChannelAttention", "ResidualDenseBlock",
    "HaarDWT", "FrequencyDomainRefinement",
    "BackProjectionUpsampler", "BicubicUpsampler", "GeometricSelfEnsemble",
    "RangeNullProjector", "decode_degradation_params", "kernel_from_params",
    "baselines", "experiments", "selfcheck", "__version__", "name",
    "BASELINES", "evaluate_baseline", "evaluate_all_baselines",
    "PAPER_CORE_ABLATIONS",
]
