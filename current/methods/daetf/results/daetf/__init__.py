"""DAETF-Net: Domain-Adaptive Equivariant Tensor Fusion Network.

Hyperspectral-multispectral image fusion built to survive the transfer from the
dataset it was trained on to one it has never seen.

    import daetf
    cfg = daetf.Config().resolve()          # finds the datasets, infers bands
    daetf.selfcheck.run_all()               # verifies the mechanisms numerically
    model, hist = daetf.train(cfg)
    daetf.evaluate_dataset(model, cfg.source_root, cfg)
"""

from . import baselines, experiments, selfcheck
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
                      DegradationEncoder, EquivariantFeatureExtractor, FiLM,
                      FrequencyDomainRefinement, HaarDWT, P4ConvP4, P4ConvZ2,
                      PlainFeatureExtractor, RegionAwareMoE,
                      TensorSpectralSpatialEncoder)

__version__ = "2.0.0"

__all__ = [
    "Config", "DAETFNet", "SPCLoss",
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
    "TensorSpectralSpatialEncoder", "RegionAwareMoE", "HaarDWT",
    "FrequencyDomainRefinement", "BackProjectionUpsampler", "BicubicUpsampler",
    "baselines", "experiments", "selfcheck", "__version__",
    "BASELINES", "evaluate_baseline", "evaluate_all_baselines",
]
