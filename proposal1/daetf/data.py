"""Datasets, scene caching and spectral-response estimation.

The v1 dataset returned `torch.randn(...)` with a hardcoded length of 100, so
nothing was ever trained on real data. This module loads the actual .mat scenes
and synthesises the observation pair through the physical forward model.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import Config
from .degrade import blur_downsample, gaussian_kernel2d
from .io_utils import find_pairs, load_mat, to_chw01


class SceneCache:
    """Bounded LRU cache of decoded scenes, held in float16.

    Harvard scenes are 1040x1392x31; caching them all as float32 would need
    ~5.4 GB, so scenes are stored halved and evicted least-recently-used.
    """

    def __init__(self, bands: int, msi_bands: int, limit: int = 12):
        self.bands, self.msi_bands, self.limit = bands, msi_bands, limit
        self.store: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.order: List[str] = []

    def get(self, stem: str, hsi_path: str, rgb_path: str
            ) -> Tuple[np.ndarray, np.ndarray]:
        if stem in self.store:
            self.order.remove(stem)
            self.order.append(stem)
            return self.store[stem]
        hsi = to_chw01(load_mat(hsi_path), self.bands)
        rgb = to_chw01(load_mat(rgb_path), self.msi_bands)
        if rgb.shape[-2:] != hsi.shape[-2:]:
            t = torch.from_numpy(rgb)[None]
            rgb = F.interpolate(t, size=hsi.shape[-2:], mode="bicubic",
                                align_corners=False).clamp(0, 1)[0].numpy()
        item = (hsi.astype(np.float16), rgb.astype(np.float16))
        self.store[stem] = item
        self.order.append(stem)
        while len(self.order) > self.limit:
            self.store.pop(self.order.pop(0), None)
        return item


class FusionPatchDataset(Dataset):
    """Samples HR patches and synthesises (LR-HSI, MSI) through the forward
    model, randomising blur, noise and spectral response.

    The randomisation is the domain-shift defence: a model that has only ever
    seen one fixed bicubic degradation has no reason to work on a real sensor.
    """

    def __init__(self, root: str, split: str, cfg: Config, train: bool = True,
                 srf: Optional[np.ndarray] = None, length: int = 8000):
        self.cfg, self.train, self.length = cfg, train, length
        self.pairs = find_pairs(root, split)
        self.cache = SceneCache(
            cfg.bands, cfg.msi_bands,
            limit=min(len(self.pairs), cfg.cache_limit) if train else 4)
        self.srf = srf

    def __len__(self) -> int:
        return self.length if self.train else len(self.pairs)

    def _sample_kernel(self) -> Tuple[torch.Tensor, List[float]]:
        cfg = self.cfg
        sx = random.uniform(*cfg.sigma_range)
        sy = sx if random.random() > cfg.aniso else random.uniform(*cfg.sigma_range)
        th = random.uniform(0, math.pi)
        return (gaussian_kernel2d(cfg.blur_ksize, sx, sy, th),
                [sx, sy, math.sin(2 * th), math.cos(2 * th)])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        stem, hp, rp = (self.pairs[random.randrange(len(self.pairs))] if self.train
                        else self.pairs[idx % len(self.pairs)])
        hsi, rgb = self.cache.get(stem, hp, rp)

        if self.train:
            p = cfg.patch
            _, h, w = hsi.shape
            top, left = random.randrange(0, h - p + 1), random.randrange(0, w - p + 1)
            gt = torch.from_numpy(hsi[:, top:top + p, left:left + p].astype(np.float32))
            msi = torch.from_numpy(rgb[:, top:top + p, left:left + p].astype(np.float32))
            if random.random() < 0.5:
                gt, msi = torch.flip(gt, [-1]), torch.flip(msi, [-1])
            k = random.randrange(4)          # the p4 stem handles these natively
            if k:
                gt, msi = torch.rot90(gt, k, (-2, -1)), torch.rot90(msi, k, (-2, -1))
        else:
            p = (min(hsi.shape[1], hsi.shape[2]) // cfg.scale) * cfg.scale
            gt = torch.from_numpy(hsi[:, :p, :p].astype(np.float32))
            msi = torch.from_numpy(rgb[:, :p, :p].astype(np.float32))

        es = cfg.eval_sigma
        kernel, deg = (self._sample_kernel() if self.train else
                       (gaussian_kernel2d(cfg.blur_ksize, es, es, 0.0), [es, es, 0.0, 1.0]))

        lr = blur_downsample(gt[None], kernel, cfg.scale)[0]
        noise = random.uniform(*cfg.noise_range) if self.train else 0.0
        if noise > 0:
            lr = (lr + torch.randn_like(lr) * noise).clamp(0, 1)

        # sometimes replace the real RGB with a jittered synthetic MSI, so the
        # model never assumes one fixed spectral response function
        if self.train and self.srf is not None and random.random() < cfg.srf_jitter:
            s = torch.from_numpy(self.srf).float()
            s = (s * (1 + 0.15 * torch.randn_like(s))).clamp_min(0)
            s = s / s.sum(0, keepdim=True).clamp_min(1e-6) * float(self.srf.sum(0).mean())
            msi = torch.einsum("chw,cm->mhw", gt, s).clamp(0, 1)

        return {"lr": lr, "msi": msi, "gt": gt,
                "deg": torch.tensor(deg + [noise], dtype=torch.float32),
                "kernel": kernel, "name": stem}


def estimate_srf(root: str, split: str, cfg: Config, max_scenes: int = 8,
                 samples_per_scene: int = 20000) -> np.ndarray:
    """Least-squares spectral response function: min_S || HSI @ S - RGB ||^2.

    Recovering the SRF from the data makes the spectral-consistency loss a real
    physical constraint instead of a hand-picked approximation, and it adapts
    automatically to a dataset whose RGB was rendered with a different response.
    """
    pairs = find_pairs(root, split)[:max_scenes]
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for stem, hp, rp in pairs:
        hsi = to_chw01(load_mat(hp), cfg.bands)
        rgb = to_chw01(load_mat(rp), cfg.msi_bands)
        if rgb.shape[-2:] != hsi.shape[-2:]:
            rgb = F.interpolate(torch.from_numpy(rgb)[None], size=hsi.shape[-2:],
                                mode="bicubic", align_corners=False)[0].numpy()
        h = hsi.reshape(cfg.bands, -1).T
        r = rgb.reshape(cfg.msi_bands, -1).T
        idx = rng.choice(h.shape[0], size=min(samples_per_scene, h.shape[0]),
                         replace=False)
        xs.append(h[idx])
        ys.append(r[idx])
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    s, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.clip(s, 0.0, None).astype(np.float32)
