"""Multi-dataset loaders for Chikusei, Houston, and other HSI-MSI fusion benchmarks.

Chikusei:  128 bands, 2517x2335, Headwall Nano-Hyperspec-VNIR-C (363-1018 nm)
           Ships as a single .mat file with the full HSI cube.
Houston:   48 bands, 1202x4172, ITRES CASI-1500 (380-1050 nm)
           Ships as .h5 (HDF5) with pre-split train/val sets.
"""

from __future__ import annotations

import glob
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .degrade import blur_downsample, gaussian_kernel2d
from .srf import chikusei_srf, houston_srf, nikon_d700_srf


# ---------------------------------------------------------------------------
# Chikusei dataset (single large .mat, HSI-only)
# ---------------------------------------------------------------------------
class ChikuseiDataset(Dataset):
    """Chikusei hyperspectral dataset for HSI-MSI fusion.

    The Kaggle dataset (mingliu123/chikusei) ships a single .mat file:
        HyperspecVNIR_Chikusei_20140729.mat  (2517 x 2335 x 128)

    We split the full scene into non-overlapping 128x128 patches (as used in
    the published literature), with a 70/30 train/test spatial split.

    The MSI is synthesised from the HSI using the sensor-specific SRF.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        bands: int = 128,
        scale: int = 4,
        patch_size: int = 128,
        max_scenes: int = 0,
        srf_mode: str = "sensor",
    ):
        """
        Args:
            root: Path to directory containing the .mat file(s).
            split: 'train' or 'test'.
            bands: Number of spectral bands (128 for Chikusei).
            scale: Downsampling factor.
            patch_size: Spatial patch size for training.
            max_scenes: 0 = use all patches.
            srf_mode: 'sensor' for sensor-specific SRF, 'estimate' to learn from data.
        """
        self.split = split
        self.bands = bands
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = split.lower() == "train"

        # SRF
        if srf_mode == "sensor":
            self.srf = chikusei_srf(bands)
        elif srf_mode == "estimate":
            self.srf = None  # will be estimated from data
        else:
            self.srf = chikusei_srf(bands)

        self.kernel = gaussian_kernel2d(9, 1.2)

        # Load the full HSI cube
        self.cube = self._load_cube(root)
        C, H, W = self.cube.shape
        print(f"[Chikusei] Loaded cube: {C} bands, {H}x{W} pixels")

        # Split into patches
        self.patches = self._make_patches(H, W)
        print(f"[Chikusei] {split}: {len(self.patches)} patches")

    def _load_cube(self, root: str) -> np.ndarray:
        """Load the Chikusei HSI cube from .mat file."""
        # Search for .mat files
        mat_files = glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found under {root}")

        # Find the main HSI file (largest .mat file)
        mat_files.sort(key=lambda f: os.path.getsize(f), reverse=True)
        mat_path = mat_files[0]
        print(f"[Chikusei] Loading from: {mat_path}")

        from scipy.io import loadmat
        data = loadmat(mat_path)

        # Find the HSI array
        for key, val in data.items():
            if not key.startswith("__") and hasattr(val, "shape"):
                arr = np.array(val, dtype=np.float32)
                if arr.ndim == 3 and min(arr.shape) > 10:
                    # Ensure (C, H, W) format
                    if arr.shape[0] > arr.shape[-1]:
                        # Likely (H, W, C) -> (C, H, W)
                        arr = arr.transpose(2, 0, 1)
                    elif arr.shape[1] > arr.shape[-1]:
                        # Likely (H, C, W) -> (C, H, W)
                        arr = arr.transpose(1, 0, 2)
                    # Normalize to [0, 1]
                    if arr.max() > 1.0:
                        arr = arr / arr.max()
                    return arr

        raise ValueError(f"No valid 3D array found in {mat_path}")

    def _make_patches(self, H: int, W: int) -> List[Tuple[int, int]]:
        """Split the scene into non-overlapping patches with train/test split."""
        p = self.patch_size
        stride = p  # non-overlapping

        coords = []
        for y in range(0, H - p + 1, stride):
            for x in range(0, W - p + 1, stride):
                coords.append((y, x))

        # Deterministic 70/30 split (spatially non-overlapping)
        random.seed(42)
        random.shuffle(coords)

        n_train = int(0.7 * len(coords))
        if self.is_train:
            return coords[:n_train]
        else:
            return coords[n_train:]

    def __len__(self) -> int:
        return len(self.patches) * (100 if self.is_train else 1)

    def _sim(self, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate LR-HSI and MSI from HR-HSI."""
        C, H, W = gt.shape
        k = self.kernel.shape[0]

        # Blur + decimate
        from scipy.ndimage import convolve
        blurred = np.empty_like(gt)
        for c in range(C):
            blurred[c] = convolve(gt[c], self.kernel, mode="wrap")

        hr, wr = H // self.scale, W // self.scale
        y0 = (H - hr * self.scale) // 2
        x0 = (W - wr * self.scale) // 2
        lr = blurred[:, y0::self.scale, x0::self.scale].astype(np.float32)

        # Synthesise MSI from HSI using SRF
        srf = self.srf if self.srf is not None else chikusei_srf(self.bands)
        msi = np.einsum("chw,cm->mhw", gt, srf).astype(np.float32)
        msi = np.clip(msi, 0.0, 1.0)

        return lr, msi

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_train:
            patch_idx = idx % len(self.patches)
        else:
            patch_idx = idx % len(self.patches)

        y, x = self.patches[patch_idx]
        p = self.patch_size
        gt = self.cube[:, y:y + p, x:x + p].copy()

        # Random augmentation for training
        if self.is_train:
            if random.random() < 0.5:
                gt = gt[:, :, ::-1].copy()
            if random.random() < 0.5:
                gt = gt[:, ::-1, :].copy()
            if random.random() < 0.5:
                k = random.randint(1, 3)
                gt = np.rot90(gt, k, axes=(1, 2)).copy()

        lr, msi = self._sim(gt)

        return (
            torch.from_numpy(gt.astype(np.float32)),
            torch.from_numpy(lr.astype(np.float32)),
            torch.from_numpy(msi.astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Houston dataset (HDF5 format with train/val splits)
# ---------------------------------------------------------------------------
class HoustonDataset(Dataset):
    """Houston hyperspectral dataset for HSI-MSI fusion.

    The IEEE GRSS Data Fusion Contest 2018 dataset ships as HDF5 (.h5) files:
        train.h5  (keys: LRHSI, HSI_up, RGB, GT)
        val.h5    (keys: LRHSI, HSI_up, RGB, GT)

    For HSI-MSI fusion, we use the full-resolution HSI as ground truth,
    synthesise LR-HSI via blur+decimate, and synthesise MSI via SRF.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        bands: int = 48,
        scale: int = 4,
        patch_size: int = 64,
        max_dim: int = 512,
        srf_mode: str = "sensor",
    ):
        """
        Args:
            root: Path to directory containing the .h5 file(s).
            split: 'train' or 'test'/'val'.
            bands: Number of spectral bands (48 for Houston).
            scale: Downsampling factor.
            patch_size: Spatial patch size for training.
            max_dim: Maximum scene dimension.
            srf_mode: 'sensor' for sensor-specific SRF, 'estimate' to learn from data.
        """
        self.split = split
        self.bands = bands
        self.scale = scale
        self.patch_size = patch_size
        self.max_dim = max_dim
        self.is_train = split.lower() == "train"

        # SRF
        if srf_mode == "sensor":
            self.srf = houston_srf(bands)
        elif srf_mode == "estimate":
            self.srf = None
        else:
            self.srf = houston_srf(bands)

        self.kernel = gaussian_kernel2d(9, 1.2)

        # Load data
        self.scenes = self._load_data(root, split)
        print(f"[Houston] {split}: {len(self.scenes)} scenes")

    def _load_data(self, root: str, split: str) -> List[np.ndarray]:
        """Load Houston data from HDF5 or .mat files."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for Houston dataset. Install with: pip install h5py")

        # Map split names
        split_map = {"train": "train", "test": "val", "val": "val"}
        h5_split = split_map.get(split, split)

        # Search for .h5 files
        h5_files = glob.glob(os.path.join(root, "**", f"*{h5_split}*.h5"), recursive=True)
        if not h5_files:
            h5_files = glob.glob(os.path.join(root, "**", "*.h5"), recursive=True)

        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found under {root}")

        scenes = []
        for h5_path in h5_files:
            print(f"[Houston] Loading from: {h5_path}")
            with h5py.File(h5_path, "r") as f:
                # Try different key patterns
                for key_pattern in [("HSI_up",), ("GT",), ("hsi",), ("data",)]:
                    if key_pattern[0] in f:
                        hsi = np.array(f[key_pattern[0]], dtype=np.float32)
                        # Ensure (C, H, W) format
                        if hsi.ndim == 3:
                            if hsi.shape[-1] == self.bands:
                                hsi = hsi.transpose(2, 0, 1)
                            elif hsi.shape[0] != self.bands and hsi.shape[1] == self.bands:
                                hsi = hsi.transpose(1, 0, 2)
                        # Normalize
                        if hsi.max() > 1.0:
                            hsi = hsi / hsi.max()
                        scenes.append(hsi)
                        break

        if not scenes:
            # Fallback: try .mat format
            mat_files = glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)
            from scipy.io import loadmat
            for mat_path in mat_files:
                data = loadmat(mat_path)
                for key, val in data.items():
                    if not key.startswith("__") and hasattr(val, "shape"):
                        arr = np.array(val, dtype=np.float32)
                        if arr.ndim == 3 and min(arr.shape) > 10:
                            if arr.shape[0] > arr.shape[-1]:
                                arr = arr.transpose(2, 0, 1)
                            if arr.max() > 1.0:
                                arr = arr / arr.max()
                            scenes.append(arr)
                            break

        return scenes

    def __len__(self) -> int:
        return len(self.scenes) * (50 if self.is_train else 1)

    def _sim(self, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate LR-HSI and MSI from HR-HSI."""
        C, H, W = gt.shape

        # Blur + decimate
        from scipy.ndimage import convolve
        blurred = np.empty_like(gt)
        for c in range(C):
            blurred[c] = convolve(gt[c], self.kernel, mode="wrap")

        hr, wr = H // self.scale, W // self.scale
        y0 = (H - hr * self.scale) // 2
        x0 = (W - wr * self.scale) // 2
        lr = blurred[:, y0::self.scale, x0::self.scale].astype(np.float32)

        # Synthesise MSI
        srf = self.srf if self.srf is not None else houston_srf(self.bands)
        msi = np.einsum("chw,cm->mhw", gt, srf).astype(np.float32)
        msi = np.clip(msi, 0.0, 1.0)

        return lr, msi

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scene_idx = idx % len(self.scenes)
        cube = self.scenes[scene_idx]

        C, H, W = cube.shape

        if self.is_train:
            # Random patch
            p = min(self.patch_size, H, W)
            y = random.randrange(0, H - p + 1)
            x = random.randrange(0, W - p + 1)
            gt = cube[:, y:y + p, x:x + p].copy()
            # Augmentation
            if random.random() < 0.5:
                gt = gt[:, :, ::-1].copy()
            if random.random() < 0.5:
                gt = gt[:, ::-1, :].copy()
        else:
            # Centre crop
            p = (min(H, W) // self.scale) * self.scale
            y0 = (H - p) // 2
            x0 = (W - p) // 2
            gt = cube[:, y0:y0 + p, x0:x0 + p].copy()

        lr, msi = self._sim(gt)

        return (
            torch.from_numpy(gt.astype(np.float32)),
            torch.from_numpy(lr.astype(np.float32)),
            torch.from_numpy(msi.astype(np.float32)),
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def get_dataset(
    name: str,
    root: str,
    split: str = "train",
    **kwargs,
) -> Dataset:
    """Get a dataset by name.

    Args:
        name: 'chikusei', 'houston', 'cave', or 'harvard'.
        root: Path to dataset root.
        split: 'train' or 'test'.
        **kwargs: Additional arguments passed to the dataset constructor.
    """
    name = name.lower()
    if name == "chikusei":
        return ChikuseiDataset(root, split, bands=kwargs.get("bands", 128), **kwargs)
    elif name == "houston":
        return HoustonDataset(root, split, bands=kwargs.get("bands", 48), **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'chikusei' or 'houston'.")
