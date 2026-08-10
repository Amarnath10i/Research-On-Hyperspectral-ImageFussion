"""Filesystem discovery and .mat reading.

This module deliberately has no dependency on the rest of the package so that
dataset discovery can never create an import cycle.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover
    sio = None

SPLIT_NAMES = ("Train", "train", "TRAIN")
TEST_NAMES = ("Test", "test", "TEST", "Val", "val")


# --------------------------------------------------------------------------- IO
def load_mat(path: str) -> np.ndarray:
    """Return the first real array stored in a MATLAB file."""
    if sio is None:
        raise RuntimeError("scipy is required to read .mat files")
    mat = sio.loadmat(path)
    for k, v in mat.items():
        if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim >= 2:
            return np.asarray(v)
    raise ValueError(f"no array in {path}")


def to_chw01(arr: np.ndarray, channels: int) -> np.ndarray:
    """Normalise an array to channel-first float32 in [0, 1]."""
    if channels is None:
        raise ValueError("channel count is unresolved - call Config.resolve() first")
    a = np.squeeze(np.asarray(arr)).astype(np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected 3D array, got {a.shape}")
    if a.shape[0] == channels:
        pass
    elif a.shape[-1] == channels:
        a = np.transpose(a, (2, 0, 1))
    else:
        raise ValueError(f"cannot find {channels} channels in {a.shape}")
    mx = float(a.max())
    if mx > 1.0:
        a = a / mx
    return np.clip(a, 0.0, 1.0)


# ------------------------------------------------------------------- discovery
def _looks_like_dataset(path: str) -> bool:
    """A dataset root is any directory holding <split>/HSI."""
    for split in SPLIT_NAMES + TEST_NAMES:
        d = os.path.join(path, split)
        if os.path.isdir(d) and any(
            os.path.isdir(os.path.join(d, h)) for h in ("HSI", "hsi")
        ):
            return True
    return False


def search_roots() -> List[str]:
    """Candidate locations, most specific first.

    DAETF_DATA_ROOTS (os.pathsep separated) always takes priority, so discovery
    can be overridden without editing any code.
    """
    roots: List[str] = []
    env = os.environ.get("DAETF_DATA_ROOTS", "")
    roots += [p for p in env.split(os.pathsep) if p]
    roots += sorted(glob.glob("/kaggle/input/*"))
    roots += sorted(glob.glob(os.path.join(os.getcwd(), "data", "*")))
    roots += [os.path.join(os.getcwd(), "data"), os.getcwd()]
    return [r for r in roots if os.path.isdir(r)]


def discover_dataset(hints: Sequence[str] = (), required: bool = True,
                     verbose: bool = True) -> Optional[str]:
    """Locate a dataset root whose path matches one of `hints`.

    Handles both `<root>/Data/Train/HSI` and `<root>/Train/HSI` packaging.
    """
    found: List[str] = []
    for root in search_roots():
        for cand in (os.path.join(root, "Data"), root):
            if _looks_like_dataset(cand):
                found.append(cand)
                break
    if hints:
        lowered = [h.lower() for h in hints]
        ranked = [f for f in found if any(h in f.lower() for h in lowered)]
        found = ranked or found
    if not found:
        if required:
            raise FileNotFoundError(
                f"no dataset matching {list(hints)} found. Looked under: "
                f"{search_roots()}. Set DAETF_DATA_ROOTS or pass the root "
                f"explicitly via Config(source_root=...)."
            )
        if verbose:
            print(f"[config] optional dataset {list(hints)} not found - skipping")
        return None
    if verbose:
        print(f"[config] using dataset root: {found[0]}")
    return found[0]


def available_splits(root: str) -> Dict[str, str]:
    """Map canonical split name -> the directory name actually present."""
    out = {}
    for canonical, names in (("Train", SPLIT_NAMES), ("Test", TEST_NAMES)):
        for n in names:
            if os.path.isdir(os.path.join(root, n)):
                out[canonical] = n
                break
    return out


def infer_channels(root: str) -> Tuple[int, int]:
    """Read one HSI/RGB pair and report their channel counts."""
    splits = available_splits(root)
    split = splits.get("Train") or splits.get("Test")
    if split is None:
        raise FileNotFoundError(f"no usable split under {root}")
    base = os.path.join(root, split)
    hsi_dir = next(os.path.join(base, d) for d in ("HSI", "hsi")
                   if os.path.isdir(os.path.join(base, d)))
    rgb_dir = next((os.path.join(base, d) for d in ("RGB", "rgb")
                    if os.path.isdir(os.path.join(base, d))), None)
    hsi = np.squeeze(load_mat(sorted(glob.glob(os.path.join(hsi_dir, "*.mat")))[0]))
    bands = int(min(hsi.shape))
    msi_bands = 3
    if rgb_dir:
        rgb = np.squeeze(load_mat(sorted(glob.glob(os.path.join(rgb_dir, "*.mat")))[0]))
        msi_bands = int(min(rgb.shape))
    return bands, msi_bands


def find_pairs(root: str, split: str) -> List[Tuple[str, str, str]]:
    """Matched (stem, hsi_path, rgb_path) triples for a canonical split name."""
    actual = available_splits(root).get(split, split)
    base = os.path.join(root, actual)
    hsi_dir = next((os.path.join(base, d) for d in ("HSI", "hsi")
                    if os.path.isdir(os.path.join(base, d))), None)
    rgb_dir = next((os.path.join(base, d) for d in ("RGB", "rgb")
                    if os.path.isdir(os.path.join(base, d))), None)
    if not hsi_dir or not rgb_dir:
        raise FileNotFoundError(f"no HSI/RGB folders under {base}")
    rgb = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(os.path.join(rgb_dir, "*.mat"))}
    out = []
    for h in sorted(glob.glob(os.path.join(hsi_dir, "*.mat"))):
        stem = os.path.splitext(os.path.basename(h))[0]
        if stem in rgb:
            out.append((stem, h, rgb[stem]))
    if not out:
        raise RuntimeError(f"no matched pairs under {base}")
    return out
