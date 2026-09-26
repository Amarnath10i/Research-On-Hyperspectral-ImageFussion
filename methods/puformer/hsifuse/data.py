"""Dataset loading and the fixed train / val / test splits (Chikusei and Pavia Centre).

Chikusei: central 2048x2048 crop (no black border), normalised by the crop's global max.
    test : rows   0-256   -> 8 tiles of 256x256   (8 test images, as in TIP'26)
    val  : rows 272-400   -> 64 patches of 64x64  (64 val patches, as in TIP'26)
    train: rows 416-2048  -> random 64x64 crops (1632x2048 px ~ 832 non-overlapping 64x64 patches)
Pavia Centre (ROSIS, 1096x715x102): bands 11-102 are kept (92 bands, as in TIP'26 Table II),
normalised by the global max of those bands.
    test : rows 0-512, cols 0-512      -> 4 tiles of 256x256 (as in TIP'26)
    val  : rows 528-656                -> 63 patches of 64x64 (stride 32)
    train: rows 672-1096 (all columns) and rows 0-512, cols 528-715 -> random 64x64 crops
16-pixel gaps keep the regions from sharing any pixel. Every image, including each
training crop, is degraded on its own with reflect padding, so train and test see
the same border handling.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .ops import Degradation

CROP = 2048
TEST_ROWS, VAL_ROWS, TRAIN_ROW0 = (0, 256), (272, 400), 416
PAVIA_FIRST_BAND = 10          # drop ROSIS bands 1-10 (noisy blue end) -> 92 bands


def find_mat(root: str, dataset: str = "chikusei") -> str:
    if dataset == "pavia":
        hits = [p for p in glob.glob(os.path.join(root, "**", "Pavia.mat"), recursive=True)]
    else:
        hits = [p for p in glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)
                if "Ground_Truth" not in p and "chikusei" in p.lower()]
    if not hits:
        raise FileNotFoundError(f"No {dataset} .mat under {root}")
    return sorted(hits, key=len)[0]


def load_chikusei(path: str, cache: str | None = None) -> np.ndarray:
    """(128, 2048, 2048) float32 in [0, 1]."""
    if cache and os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    import h5py  # MATLAB v7.3 file
    with h5py.File(path, "r") as f:
        d = f["chikusei"]                      # stored (C, W, H)
        c, w, h = d.shape
        t, l = (h - CROP) // 2, (w - CROP) // 2
        out = np.empty((c, CROP, CROP), np.float32)
        for b0 in range(0, c, 16):             # chunked: the float64 cube is ~6 GB
            blk = d[b0:b0 + 16, l:l + CROP, t:t + CROP]
            out[b0:b0 + 16] = np.transpose(blk, (0, 2, 1))
    out /= out.max()
    if cache:
        np.save(cache, out)
    return out


def load_pavia(path: str) -> tuple[np.ndarray, float]:
    """(92, 1096, 715) float32 in [0, 1] and the DN value that maps to 1."""
    from scipy.io import loadmat               # MATLAB v5 file
    m = loadmat(path)
    key = [k for k in m if not k.startswith("__")][0]
    cube = np.transpose(m[key].astype(np.float32), (2, 0, 1))[PAVIA_FIRST_BAND:]
    peak = float(cube.max())
    return np.ascontiguousarray(cube / peak), peak


def split(cube: np.ndarray):
    test = [cube[:, TEST_ROWS[0]:TEST_ROWS[1], j:j + 256] for j in range(0, CROP, 256)]
    val = [cube[:, r:r + 64, j:j + 64] for r in range(VAL_ROWS[0], VAL_ROWS[1], 64)
           for j in range(0, CROP, 64)]
    train = cube[:, TRAIN_ROW0:, :]
    return np.ascontiguousarray(train), np.stack(test), np.stack(val)


def split_pavia(cube: np.ndarray):
    """-> (list of training regions, test stack (4,C,256,256), val stack (63,C,64,64))."""
    test = [cube[:, r:r + 256, c:c + 256] for r in (0, 256) for c in (0, 256)]
    w = cube.shape[2]
    val = [cube[:, r:r + 64, c:c + 64] for r in range(528, 656 - 63, 32) for c in range(0, w - 63, 32)]
    train = [np.ascontiguousarray(cube[:, 672:, :]), np.ascontiguousarray(cube[:, 0:512, 528:])]
    return train, np.stack(test), np.stack(val)


def load_dataset(name: str, mat: str, cache: str | None = None):
    """-> train (array or list of arrays), test, val, DN peak."""
    if name == "pavia":
        cube, peak = load_pavia(mat if mat.endswith(".mat") else find_mat(mat, "pavia"))
        tr, te, va = split_pavia(cube)
        return tr, te, va, peak
    path = mat if mat.endswith(".mat") else find_mat(mat)
    tr, te, va = split(np.asarray(load_chikusei(path, cache)))
    return tr, te, va, 15133.0


@torch.no_grad()
def make_pairs(gt: torch.Tensor, deg: Degradation, bs: int = 16):
    """Degrade a stack of standalone images -> (lr_hsi, hr_msi, gt)."""
    lrs, msis = [], []
    for i in range(0, gt.shape[0], bs):
        lr, ms = deg.simulate(gt[i:i + bs])
        lrs.append(lr), msis.append(ms)
    return torch.cat(lrs), torch.cat(msis), gt


class PatchSampler:
    """Random crops (+ dihedral augmentation) from one or several training regions, degraded on-device."""

    def __init__(self, train, deg: Degradation, patch: int = 64, device="cuda"):
        regions = train if isinstance(train, (list, tuple)) else [train]
        self.xs = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in regions]
        areas = torch.tensor([float((x.shape[1] - patch + 1) * (x.shape[2] - patch + 1)) for x in self.xs])
        self.prob = areas / areas.sum()
        self.deg, self.p = deg, patch
        self.dev = device

    @property
    def pixels(self) -> int:
        return int(sum(x.shape[1] * x.shape[2] for x in self.xs))

    @torch.no_grad()
    def __call__(self, bs: int):
        which = torch.multinomial(self.prob, bs, replacement=True).tolist() if len(self.xs) > 1 else [0] * bs
        crops = []
        for r in which:
            _, h, w = self.xs[r].shape
            y = int(torch.randint(0, h - self.p + 1, (1,)))
            x = int(torch.randint(0, w - self.p + 1, (1,)))
            crops.append(self.xs[r][:, y:y + self.p, x:x + self.p])
        g = torch.stack(crops)
        k = int(torch.randint(0, 4, (1,)))
        g = torch.rot90(g, k, (2, 3))
        if torch.rand(1).item() < 0.5:
            g = torch.flip(g, (3,))
        g = g.contiguous()
        lr, ms = self.deg.simulate(g)
        return lr, ms, g
