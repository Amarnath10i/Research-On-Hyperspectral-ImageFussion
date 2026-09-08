"""Common CAVE dataset loader for all review benchmark methods.

Loads CAVE PNG images from the liptee Kaggle dataset and simulates Wald's
protocol (Nikon D700 SRF + Gaussian blur + downsampling) consistently
across all methods in the review.

Usage on Kaggle:
    root = "/kaggle/input/datasets/liptee/hyperspectral-image-restoration-based-on-cave"
    ds = CAVEDataset(root, split="train")
    item = ds[0]
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve
from torch.utils.data import Dataset

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Nikon D700 spectral response function (31 bands, 400-700 nm at 10 nm)
# From common/hsifusion/srf.py — the standard used by FeINFN, BDT, DSPNet,
# PSRT, MIMO-SST, DHIF and other published CAVE fusion papers.
# ---------------------------------------------------------------------------

_NIKON_D700_31 = np.array([
    [0.0050, 0.0130, 0.2400], [0.0060, 0.0190, 0.3600],
    [0.0070, 0.0280, 0.5200], [0.0080, 0.0420, 0.7100],
    [0.0090, 0.0620, 0.8800], [0.0100, 0.0890, 0.9800],
    [0.0110, 0.1250, 1.0000], [0.0130, 0.1750, 0.9500],
    [0.0150, 0.2400, 0.8400], [0.0180, 0.3300, 0.6900],
    [0.0230, 0.4500, 0.5300], [0.0310, 0.5900, 0.3900],
    [0.0450, 0.7400, 0.2700], [0.0700, 0.8800, 0.1800],
    [0.1100, 0.9700, 0.1200], [0.1700, 1.0000, 0.0800],
    [0.2600, 0.9800, 0.0550], [0.3800, 0.9100, 0.0400],
    [0.5300, 0.8000, 0.0300], [0.6900, 0.6700, 0.0230],
    [0.8300, 0.5300, 0.0180], [0.9300, 0.4000, 0.0140],
    [0.9900, 0.2900, 0.0110], [1.0000, 0.2100, 0.0090],
    [0.9700, 0.1500, 0.0075], [0.9000, 0.1050, 0.0062],
    [0.8000, 0.0740, 0.0052], [0.6800, 0.0520, 0.0044],
    [0.5500, 0.0370, 0.0037], [0.4300, 0.0260, 0.0031],
    [0.3200, 0.0190, 0.0026],
], dtype=np.float32)


def nike_d700_srf(bands: int = 31, normalise: bool = True) -> np.ndarray:
    """Nikon D700 response as [bands, 3], resampled if bands != 31.

    When *normalise* is True each column sums to 1 so the simulated MSI stays
    in the same radiometric range as the HSI.
    """
    src = _NIKON_D700_31
    if bands != src.shape[0]:
        xs = np.linspace(0.0, 1.0, src.shape[0])
        xd = np.linspace(0.0, 1.0, bands)
        src = np.stack([np.interp(xd, xs, src[:, i]) for i in range(3)], axis=1)
    srf = src.astype(np.float32)
    if normalise:
        srf = srf / np.maximum(srf.sum(axis=0, keepdims=True), 1e-8)
    return srf


# ---------------------------------------------------------------------------
# Gaussian blur kernel
# ---------------------------------------------------------------------------

def make_gaussian_kernel(size: int = 9, sigma: float = 1.2) -> np.ndarray:
    """Isotropic Gaussian blur kernel, normalised to sum to 1.

    Returns a [size, size] float32 array.
    """
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return (k / k.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Wald's protocol simulation
# ---------------------------------------------------------------------------

def simulate_wald(
    hsi: np.ndarray,
    kernel: np.ndarray,
    scale: int,
    srf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Wald's protocol: blur+decimate HSI → LR-HSI, HSI × SRF → HR-MSI.

    Parameters
    ----------
    hsi : [C, H, W] float32 in [0, 1]
        Ground-truth hyperspectral image.
    kernel : [k, k] float32
        Gaussian blur kernel (normalised).
    scale : int
        Downsampling factor.
    srf : [C, 3] float32
        Spectral response function (normalised columns).

    Returns
    -------
    lr_hsi : [C, H//s, W//s] float32
        Low-resolution HSI (blurred + decimated).
    hr_msi : [3, H, W] float32
        High-resolution multispectral image (spectral projection).
    """
    C, H, W = hsi.shape

    # --- LR-HSI: blur each band with mode='wrap', then downsample ---
    k = kernel.shape[0]
    pad = k // 2
    blurred = np.empty_like(hsi)
    for c in range(C):
        blurred[c] = convolve(hsi[c], kernel, mode="wrap")
    hr, wr = H // scale, W // scale
    y0 = (H - hr * scale) // 2
    x0 = (W - wr * scale) // 2
    lr_hsi = blurred[:, y0::scale, x0::scale].astype(np.float32)

    # --- HR-MSI: spectral projection via SRF ---
    # hsi is [C, H, W], srf is [C, 3] → einsum "chw,cm->mhw"
    hr_msi = np.einsum("chw,cm->mhw", hsi, srf).astype(np.float32)
    hr_msi = np.clip(hr_msi, 0.0, 1.0)

    return lr_hsi, hr_msi


# ---------------------------------------------------------------------------
# CAVE Dataset (liptee PNG layout)
# ---------------------------------------------------------------------------

# The liptee Kaggle dataset layout:
#   <root>/Train/<scene_name>/band_01.png  … band_31.png
#   <root>/Test/<scene_name>/band_01.png   … band_31.png
#
# Scenes are discovered by walking the directory tree and locating folders
# that contain band_01.png.

_TRAIN_SCENE_COUNT = 20
_TEST_SCENE_COUNT = 12


def _discover_scenes(root: str, split: str) -> List[Tuple[str, str]]:
    """Walk *root* and return sorted [(scene_name, scene_dir)] for *split*.

    A scene directory is any folder containing ``band_01.png``.
    """
    # Try multiple case variants for the split directory
    split_dir = None
    for name in (split, split.capitalize(), split.upper()):
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            split_dir = candidate
            break
    if split_dir is None:
        raise FileNotFoundError(
            f"split '{split}' not found under {root} "
            f"(looked for {split}, {split.capitalize()}, {split.upper()})"
        )

    scenes: List[Tuple[str, str]] = []
    for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        # Check for band_01.png (case-insensitive)
        for band_name in ("band_01.png", "Band_01.png", "BAND_01.png"):
            if os.path.isfile(os.path.join(entry.path, band_name)):
                scenes.append((entry.name, entry.path))
                break
    if not scenes:
        raise FileNotFoundError(
            f"no scenes containing band_*.png found under {split_dir}"
        )
    return scenes


def _load_scene_bands(
    scene_dir: str,
    bands: int = 31,
    max_dim: Optional[int] = None,
) -> np.ndarray:
    """Load all band PNGs from a scene directory.

    Returns [C, H, W] float32 in [0, 1].
    """
    try:
        from PIL import Image
    except ImportError:
        import cv2
        cv2 = cv2  # keep for type-checkers
        _use_pil = False
    else:
        _use_pil = True

    arrays: List[np.ndarray] = []
    for i in range(1, bands + 1):
        # Try multiple naming conventions
        loaded = False
        for pattern in (f"band_{i:02d}.png", f"Band_{i:02d}.png",
                        f"BAND_{i:02d}.png", f"band_{i}.png"):
            path = os.path.join(scene_dir, pattern)
            if os.path.isfile(path):
                if _use_pil:
                    img = Image.open(path)
                    arr = np.asarray(img, dtype=np.float32)
                else:
                    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                # Normalise to [0, 1]
                if arr.max() > 1.0:
                    arr = arr / 255.0
                arrays.append(arr)
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(
                f"band {i} not found in {scene_dir} (tried band_{i:02d}.png etc.)"
            )

    cube = np.stack(arrays, axis=0)  # [C, H, W]

    # Centre-crop to max_dim if requested
    if max_dim is not None and (cube.shape[1] > max_dim or cube.shape[2] > max_dim):
        y0 = max(0, (cube.shape[1] - max_dim) // 2)
        x0 = max(0, (cube.shape[2] - max_dim) // 2)
        cube = cube[:, y0:y0 + max_dim, x0:x0 + max_dim]

    return cube


class CAVEDataset(Dataset):
    """CAVE dataset loader for review benchmarks (liptee PNG layout).

    Parameters
    ----------
    root : str
        Dataset root (e.g. ``/kaggle/input/datasets/liptee/...``).
    split : str
        ``"train"`` or ``"test"``.
    bands : int
        Number of spectral bands (default 31).
    scale : int
        Downsampling factor (default 4).
    patch_size : int
        Random crop size for training (default 80).
    max_dim : int
        Centre-crop large scenes to this size (default 512).
    srf : np.ndarray or None
        Spectral response function [bands, 3]. Uses Nikon D700 if None.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        bands: int = 31,
        scale: int = 4,
        patch_size: int = 80,
        max_dim: int = 512,
        srf: Optional[np.ndarray] = None,
    ) -> None:
        self.root = root
        self.split = split
        self.bands = bands
        self.scale = scale
        self.patch_size = patch_size
        self.max_dim = max_dim
        self.is_train = split.lower() == "train"
        self.srf = srf if srf is not None else nike_d700_srf(bands)
        self.kernel = make_gaussian_kernel(size=9, sigma=1.2)

        self.scenes = _discover_scenes(root, split)
        # Verify expected counts (warn but don't fail)
        expected = _TRAIN_SCENE_COUNT if self.is_train else _TEST_SCENE_COUNT
        if len(self.scenes) != expected:
            print(
                f"[CAVEDataset] WARNING: expected {expected} {split} scenes, "
                f"found {len(self.scenes)}"
            )

        # Pre-load all scenes into memory (CAVE scenes are small enough)
        self._cache: Dict[str, np.ndarray] = {}
        for name, path in self.scenes:
            self._cache[name] = _load_scene_bands(path, bands, max_dim)

    def __len__(self) -> int:
        if self.is_train:
            # Infinite-style: return a large number for random sampling
            return 10000
        return len(self.scenes)

    def _random_crop(
        self, hsi: np.ndarray, size: int
    ) -> Tuple[np.ndarray, int, int]:
        """Extract a random [size, size] patch from [C, H, W] array."""
        _, H, W = hsi.shape
        if H < size or W < size:
            # Pad if scene is smaller than patch
            pad_h = max(0, size - H)
            pad_w = max(0, size - W)
            hsi = np.pad(
                hsi,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect",
            )
            _, H, W = hsi.shape
        y = np.random.randint(0, H - size + 1)
        x = np.random.randint(0, W - size + 1)
        return hsi[:, y : y + size, x : x + size], y, x

    @staticmethod
    def _augment(
        hsi: np.ndarray, msi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random horizontal flip and 90° rotations."""
        # Random horizontal flip
        if np.random.random() < 0.5:
            hsi = hsi[:, :, ::-1].copy()
            msi = msi[:, :, ::-1].copy()
        # Random vertical flip
        if np.random.random() < 0.5:
            hsi = hsi[:, ::-1, :].copy()
            msi = msi[:, ::-1, :].copy()
        # Random 90° rotation (k ∈ {0,1,2,3})
        k = np.random.randint(0, 4)
        if k:
            hsi = np.rot90(hsi, k, axes=(-2, -1)).copy()
            msi = np.rot90(msi, k, axes=(-2, -1)).copy()
        return hsi, msi

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if self.is_train:
            # Random scene
            name = list(self._cache.keys())[
                np.random.randint(0, len(self._cache))
            ]
            gt = self._cache[name].copy()
            gt, _, _ = self._random_crop(gt, self.patch_size)
            hr_msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
            hr_msi = np.clip(hr_msi, 0.0, 1.0)
            gt, hr_msi = self._augment(gt, hr_msi)
            lr_hsi, _ = simulate_wald(gt, self.kernel, self.scale, self.srf)
        else:
            # Deterministic: full scene, cropped to be divisible by scale
            name, _ = self.scenes[idx % len(self.scenes)]
            gt = self._cache[name].copy()
            H, W = gt.shape[1], gt.shape[2]
            H = (H // self.scale) * self.scale
            W = (W // self.scale) * self.scale
            gt = gt[:, :H, :W]
            hr_msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
            hr_msi = np.clip(hr_msi, 0.0, 1.0)
            lr_hsi, _ = simulate_wald(gt, self.kernel, self.scale, self.srf)

        if torch is not None:
            return {
                "gt": torch.from_numpy(gt.astype(np.float32)),
                "lr_hsi": torch.from_numpy(lr_hsi.astype(np.float32)),
                "hr_msi": torch.from_numpy(hr_msi.astype(np.float32)),
                "scene_name": name,
            }
        return {
            "gt": gt.astype(np.float32),
            "lr_hsi": lr_hsi.astype(np.float32),
            "hr_msi": hr_msi.astype(np.float32),
            "scene_name": name,
        }


# ---------------------------------------------------------------------------
# Main — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    default_root = (
        "/kaggle/input/datasets/liptee/"
        "hyperspectral-image-restoration-based-on-cave"
    )
    root = sys.argv[1] if len(sys.argv) > 1 else default_root

    print(f"Root: {root}")
    print(f"Nikon D700 SRF shape: {nike_d700_srf().shape}")
    print(f"Gaussian kernel shape: {make_gaussian_kernel().shape}")

    for split in ("train", "test"):
        print(f"\n--- {split.upper()} ---")
        ds = CAVEDataset(root, split=split)
        print(f"  Scenes: {len(ds.scenes)}")
        print(f"  Dataset len: {len(ds)}")
        item = ds[0]
        print(f"  gt:         {item['gt'].shape}")
        print(f"  lr_hsi:     {item['lr_hsi'].shape}")
        print(f"  hr_msi:     {item['hr_msi'].shape}")
        print(f"  scene_name: {item['scene_name']}")
