"""hsifusion - shared infrastructure for the HSI-MSI fusion proposals.

One protocol, one metric implementation, one evaluation harness, four
architectures. Each proposal supplies only its model and its own loss terms:

    from hsifusion import BaseConfig, train, evaluate_dataset, FusionLoss

    model, hist = train(cfg, build_model=my_factory, build_loss=my_loss)
    evaluate_dataset(model, cfg.source_root, cfg)

Any model works here as long as it follows the contract
`forward(lr_hsi, msi) -> {"out": tensor}` and carries a `.cfg`. Because the
protocol fields live in `BaseConfig` and the metrics in `metrics.py`, a
difference between two proposals' numbers is attributable to the architecture
rather than to one of them evaluating at a different scale factor - which is
precisely how the ten published baselines in `existing/` became incomparable.
"""

from . import baselines, experiments
from .baselines import BASELINES, evaluate_all_baselines, evaluate_baseline
from .config import BaseConfig
from .data import FusionPatchDataset, SceneCache, estimate_srf
from .degrade import FixedDegradation, blur_downsample, gaussian_kernel2d
from .engine import (clone_state, cosine_lr, evaluate_dataset, evaluate_with_tta,
                     n_params, set_seed, test_time_adapt, tiled_inference, train)
from .io_utils import (available_splits, discover_dataset, find_dataset_roots,
                       find_pairs, infer_channels, load_mat, search_roots,
                       to_chw01)
from .losses import FusionLoss, charbonnier, gradient_loss, mmd_rbf, sam_loss
from .metrics import (evaluate_arrays, metric_ergas, metric_psnr, metric_sam,
                      metric_ssim, ssim_torch)

__version__ = "1.0.0"

__all__ = [
    "BaseConfig", "FusionLoss",
    "train", "evaluate_dataset", "evaluate_with_tta", "test_time_adapt",
    "tiled_inference", "set_seed", "cosine_lr", "clone_state", "n_params",
    "FusionPatchDataset", "SceneCache", "estimate_srf",
    "FixedDegradation", "blur_downsample", "gaussian_kernel2d",
    "discover_dataset", "find_dataset_roots", "find_pairs", "infer_channels",
    "available_splits", "load_mat", "to_chw01", "search_roots",
    "charbonnier", "sam_loss", "gradient_loss", "mmd_rbf",
    "evaluate_arrays", "metric_psnr", "metric_ssim", "metric_sam",
    "metric_ergas", "ssim_torch",
    "BASELINES", "evaluate_baseline", "evaluate_all_baselines",
    "baselines", "experiments", "__version__",
]
