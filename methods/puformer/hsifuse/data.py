"""Dataset loading and the fixed train / val / test splits (Chikusei, Pavia Centre, CAVE, Harvard).

Chikusei: central 2048x2048 crop (no black border), normalised by the crop's global max.
    test : rows   0-256   -> 8 tiles of 256x256   (8 test images, as in TIP'26)
    val  : rows 272-400   -> 64 patches of 64x64  (64 val patches, as in TIP'26)
    train: rows 416-2048  -> random 64x64 crops (1632x2048 px ~ 832 non-overlapping 64x64 patches)
Pavia Centre (ROSIS, 1096x715x102): bands 11-102 are kept (92 bands, as in TIP'26 Table II),
normalised by the global max of those bands.
    test : rows 0-512, cols 0-512      -> 4 tiles of 256x256 (as in TIP'26)
    val  : rows 528-656                -> 63 patches of 64x64 (stride 32)
    train: rows 672-1096 (all columns) and rows 0-512, cols 528-715 -> random 64x64 crops
CAVE (32 indoor scenes, 512x512x31, 400-700 nm; the 16-bit images / 65535, as distributed in the Kaggle
dataset liptee/hyperspectral-image-restoration-based-on-cave):
    the first 20 scenes by name train, the last 12 test (full 512x512 images): the 20/12 split of
    MHF-Net, MoG-DCN, 3DT-Net and DCT (the dataset's Train / Test folders).
    val  : the bottom-right 128x128 corner of every training scene -> 4 patches of 64x64 each (80)
    train: the rest of each training scene (rows 0-368, and rows 368-512 x cols 0-368)
Harvard (CZ_hsdb, 1040x1392x31, 420-720 nm; Kaggle nikeshreddypatlolla/harvard-hsi-2, 30 train / 20 test
images), each image divided by its own maximum:
    test : the central 512x512 crop of each test image (20 test images)
    val  : two 64x64 patches from the bottom-right corner of each training image (60)
    train: the rest of each training image (rows 0-896, and rows 896-1040 x cols 0-1248)
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


def _read_hsi(path: str) -> np.ndarray:
    """(31, H, W) float32 from a Kaggle CAVE / Harvard .mat file (key 'hsi', stored H x W x 31)."""
    from scipy.io import loadmat
    m = loadmat(path)
    key = "hsi" if "hsi" in m else [k for k in m if not k.startswith("__")][0]
    return np.ascontiguousarray(np.transpose(m[key].astype(np.float32), (2, 0, 1)))


def _folder(root: str, split_name: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(root, "**", split_name, "HSI", "*.mat"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no {split_name}/HSI/*.mat under {root}")
    return paths


def load_cave(root: str):
    """-> list of training cubes (20) and test cubes (12), each (31, 512, 512) in [0, 1], sorted by name."""
    tr, te = [_read_hsi(p) for p in _folder(root, "Train")], [_read_hsi(p) for p in _folder(root, "Test")]
    assert len(tr) == 20 and len(te) == 12, (len(tr), len(te))
    return tr, te


def split_cave(train_s, test_s):
    val = [c[:, r:r + 64, q:q + 64] for c in train_s for r in (384, 448) for q in (384, 448)]
    train = [np.ascontiguousarray(r) for c in train_s for r in (c[:, :368, :], c[:, 368:, :368])]
    return train, np.stack(test_s), np.stack(val)


def _harvard_cube(path: str) -> np.ndarray:
    c = _read_hsi(path)
    return c / c.max()


def load_harvard(root: str, cache: str | None = None):
    """-> (train regions, test tiles, val patches). With `cache` (a folder) they are saved once as .npy and
    memory-mapped afterwards, so two processes on one machine share a single copy (~5 GB)."""
    names = ("train_a", "train_b", "test", "val")
    if cache and all(os.path.exists(os.path.join(cache, f"{n}.npy")) for n in names):
        a, b, te, va = (np.load(os.path.join(cache, f"{n}.npy"), mmap_mode="r") for n in names)
        return [a[i] for i in range(len(a))] + [b[i] for i in range(len(b))], np.asarray(te), np.asarray(va)
    tr_paths, te_paths = _folder(root, "Train"), _folder(root, "Test")
    assert len(tr_paths) == 30 and len(te_paths) == 20, (len(tr_paths), len(te_paths))
    a = b = None
    val = []
    for i, p in enumerate(tr_paths):
        c = _harvard_cube(p)
        if a is None:
            a = np.empty((len(tr_paths), c.shape[0], 896, c.shape[2]), np.float32)
            b = np.empty((len(tr_paths), c.shape[0], c.shape[1] - 896, 1248), np.float32)
        a[i], b[i] = c[:, :896, :], c[:, 896:, :1248]
        val += [c[:, 976:1040, q:q + 64] for q in (1264, 1328)]
    test = []
    for p in te_paths:
        c = _harvard_cube(p)
        r, q = (c.shape[1] - 512) // 2, (c.shape[2] - 512) // 2
        test.append(c[:, r:r + 512, q:q + 512])           # central 512x512 crop
    te, va = np.stack(test), np.stack(val)
    if cache:
        os.makedirs(cache, exist_ok=True)
        for n, arr in zip(names, (a, b, te, va)):
            np.save(os.path.join(cache, f"{n}.npy"), arr)
    return [a[i] for i in range(len(a))] + [b[i] for i in range(len(b))], te, va


def load_dataset(name: str, mat: str, cache: str | None = None):
    """-> train (array or list of arrays), test, val, DN peak (the value that maps to 1 in RMSE_DN)."""
    if name == "cave":
        tr, te, va = split_cave(*load_cave(mat))
        return tr, te, va, 255.0               # RMSE on the 8-bit scale, as CAVE papers report it
    if name == "harvard":
        tr, te, va = load_harvard(mat, cache)
        return tr, te, va, 255.0
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

    def __init__(self, train, deg: Degradation, patch: int = 64, device="cuda", gpu_limit_gb: float = 3.0):
        regions = train if isinstance(train, (list, tuple)) else [train]
        nbytes = sum(r.size * 4 for r in regions)
        self.host = nbytes > gpu_limit_gb * 2 ** 30   # large training sets stay in host memory
        self.xs = [torch.from_numpy(r) if self.host else torch.from_numpy(np.ascontiguousarray(r)).to(device)
                   for r in regions]
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
        if self.host:
            g = g.to(self.dev, non_blocking=True)
        k = int(torch.randint(0, 4, (1,)))
        g = torch.rot90(g, k, (2, 3))
        if torch.rand(1).item() < 0.5:
            g = torch.flip(g, (3,))
        g = g.contiguous()
        lr, ms = self.deg.simulate(g)
        return lr, ms, g
