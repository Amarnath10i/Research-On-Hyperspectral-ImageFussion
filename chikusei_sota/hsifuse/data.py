"""Chikusei loading and the fixed train / val / test split.

Central 2048x2048 crop (no black border), normalised by the crop's global max.
    test : rows   0-256   -> 8 tiles of 256x256   (8 test images, as in TIP'26)
    val  : rows 272-400   -> 64 patches of 64x64  (64 val patches, as in TIP'26)
    train: rows 416-2048  -> random 64x64 crops (1632x2048 px ~ 832 non-overlapping 64x64 patches)
16-row gaps keep the regions from sharing any pixel. Every image, including each
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


def find_mat(root: str) -> str:
    hits = [p for p in glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)
            if "Ground_Truth" not in p and "chikusei" in p.lower()]
    if not hits:
        raise FileNotFoundError(f"No Chikusei .mat under {root}")
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


def split(cube: np.ndarray):
    test = [cube[:, TEST_ROWS[0]:TEST_ROWS[1], j:j + 256] for j in range(0, CROP, 256)]
    val = [cube[:, r:r + 64, j:j + 64] for r in range(VAL_ROWS[0], VAL_ROWS[1], 64)
           for j in range(0, CROP, 64)]
    train = cube[:, TRAIN_ROW0:, :]
    return np.ascontiguousarray(train), np.stack(test), np.stack(val)


@torch.no_grad()
def make_pairs(gt: torch.Tensor, deg: Degradation, bs: int = 16):
    """Degrade a stack of standalone images -> (lr_hsi, hr_msi, gt)."""
    lrs, msis = [], []
    for i in range(0, gt.shape[0], bs):
        lr, ms = deg.simulate(gt[i:i + bs])
        lrs.append(lr), msis.append(ms)
    return torch.cat(lrs), torch.cat(msis), gt


class PatchSampler:
    """Random 64x64 crops + dihedral augmentation, degraded on the fly on-device."""

    def __init__(self, train: np.ndarray, deg: Degradation, patch: int = 64, device="cuda"):
        self.x = torch.from_numpy(train).to(device)
        self.deg, self.p = deg, patch
        self.dev = device

    @torch.no_grad()
    def __call__(self, bs: int):
        _, h, w = self.x.shape
        ys = torch.randint(0, h - self.p + 1, (bs,)).tolist()
        xs = torch.randint(0, w - self.p + 1, (bs,)).tolist()
        g = torch.stack([self.x[:, y:y + self.p, x:x + self.p] for y, x in zip(ys, xs)])
        k = int(torch.randint(0, 4, (1,)))
        g = torch.rot90(g, k, (2, 3))
        if torch.rand(1).item() < 0.5:
            g = torch.flip(g, (3,))
        g = g.contiguous()
        lr, ms = self.deg.simulate(g)
        return lr, ms, g
