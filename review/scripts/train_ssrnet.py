#!/usr/bin/env python3
"""Self-contained SSRNet training script for CAVE x4 fusion task.

Adapts SSRNet (Spatial-Spectral Reconstruction Network) for hyperspectral
image fusion: LR-HSI + HR-MSI -> HR-HSI. The original SSRNet is designed
for single-image SR; here we modify it to take the concatenated
(bicubic_upsampled_LR_HSI, HR_MSI) as input and predict HR-HSI.

Designed for Kaggle P100 16GB GPU. AMP disabled.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

###############################################################################
# Data Loading (consistent with cave_common.py / train_feinfn.py)
###############################################################################

_NIKON_D700_31 = np.array([
    [0.0022, 0.0044, 0.0065], [0.0056, 0.0097, 0.0115],
    [0.0120, 0.0180, 0.0190], [0.0230, 0.0310, 0.0300],
    [0.0420, 0.0500, 0.0430], [0.0680, 0.0740, 0.0570],
    [0.1040, 0.1020, 0.0730], [0.1510, 0.1400, 0.0870],
    [0.2020, 0.1810, 0.1010], [0.2520, 0.2190, 0.1110],
    [0.2940, 0.2520, 0.1180], [0.3250, 0.2800, 0.1220],
    [0.3480, 0.3050, 0.1250], [0.3600, 0.3270, 0.1270],
    [0.3660, 0.3450, 0.1280], [0.3650, 0.3590, 0.1280],
    [0.3570, 0.3700, 0.1280], [0.3420, 0.3760, 0.1280],
    [0.3210, 0.3780, 0.1290], [0.2950, 0.3750, 0.1310],
    [0.2640, 0.3680, 0.1360], [0.2300, 0.3560, 0.1450],
    [0.1940, 0.3390, 0.1610], [0.1590, 0.3160, 0.1860],
    [0.1270, 0.2880, 0.2220], [0.0990, 0.2560, 0.2690],
    [0.0760, 0.2210, 0.3260], [0.0570, 0.1840, 0.3820],
    [0.0420, 0.1480, 0.4320], [0.0300, 0.1150, 0.4660],
    [0.0210, 0.0870, 0.4860],
], dtype=np.float32)
_NIKON_D700_31 /= _NIKON_D700_31.sum(axis=0, keepdims=True)


def nike_d700_srf(bands=31, normalise=True):
    src = _NIKON_D700_31
    if bands != src.shape[0]:
        xs = np.linspace(0.0, 1.0, src.shape[0])
        xd = np.linspace(0.0, 1.0, bands)
        src = np.stack([np.interp(xd, xs, src[:, i]) for i in range(3)], axis=1)
    srf = src.astype(np.float32)
    if normalise:
        srf = srf / np.maximum(srf.sum(axis=0, keepdims=True), 1e-8)
    return srf


def make_gaussian_kernel(size=9, sigma=1.2):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def simulate_wald(hsi, kernel, scale, srf):
    from scipy.ndimage import convolve as scipy_convolve
    C, H, W = hsi.shape
    blurred = np.empty_like(hsi)
    for c in range(C):
        blurred[c] = scipy_convolve(hsi[c], kernel, mode="wrap")
    hr, wr = H // scale, W // scale
    y0 = (H - hr * scale) // 2
    x0 = (W - wr * scale) // 2
    lr_hsi = blurred[:, y0::scale, x0::scale].astype(np.float32)
    hr_msi = np.einsum("chw,cm->mhw", hsi, srf).astype(np.float32)
    hr_msi = np.clip(hr_msi, 0.0, 1.0)
    return lr_hsi, hr_msi


def _load_mat(mat_path):
    """Load .mat file, return first real array as float32."""
    from scipy.io import loadmat
    data = loadmat(mat_path)
    for key, val in data.items():
        if not key.startswith('_') and hasattr(val, 'shape'):
            arr = np.array(val, dtype=np.float32)
            if arr.ndim >= 2 and min(arr.shape) > 1:
                if arr.ndim == 3 and arr.shape[2] > arr.shape[0]:
                    arr = arr.transpose(2, 0, 1)
                return arr
    raise ValueError(f"No valid array in {mat_path}")


def _discover_scenes(root, split):
    split_dir = None
    for name in (split, split.capitalize(), split.upper()):
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            split_dir = candidate
            break
    if split_dir is None:
        raise FileNotFoundError(f"split '{split}' not found under {root}")
    scenes = []
    for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        mat_files = [f for f in os.listdir(entry.path) if f.endswith('.mat')]
        if mat_files:
            scenes.append((entry.name, entry.path))
            continue
        for band_name in ("band_01.png", "Band_01.png", "BAND_01.png"):
            if os.path.isfile(os.path.join(entry.path, band_name)):
                scenes.append((entry.name, entry.path))
                break
    if not scenes:
        raise FileNotFoundError(f"no scenes found under {split_dir}")
    return scenes


def _load_scene_bands(scene_dir, bands=31, max_dim=None):
    mat_files = [f for f in os.listdir(scene_dir) if f.endswith('.mat')]
    if mat_files:
        cube = _load_mat(os.path.join(scene_dir, mat_files[0]))
        if max_dim is not None and (cube.shape[1] > max_dim or cube.shape[2] > max_dim):
            y0 = max(0, (cube.shape[1] - max_dim) // 2)
            x0 = max(0, (cube.shape[2] - max_dim) // 2)
            cube = cube[:, y0:y0 + max_dim, x0:x0 + max_dim]
        return cube

    try:
        from PIL import Image
        _use_pil = True
    except ImportError:
        import cv2
        _use_pil = False
    arrays = []
    for i in range(1, bands + 1):
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
                if arr.max() > 1.0:
                    arr = arr / 255.0
                arrays.append(arr)
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(f"band {i} not found in {scene_dir}")
    cube = np.stack(arrays, axis=0)
    if max_dim is not None and (cube.shape[1] > max_dim or cube.shape[2] > max_dim):
        y0 = max(0, (cube.shape[1] - max_dim) // 2)
        x0 = max(0, (cube.shape[2] - max_dim) // 2)
        cube = cube[:, y0:y0 + max_dim, x0:x0 + max_dim]
    return cube


class CAVEDataset:
    def __init__(self, root, split="train", bands=31, scale=4,
                 patch_size=80, max_dim=512, srf=None):
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
        self._cache = {}
        for name, path in self.scenes:
            self._cache[name] = _load_scene_bands(path, bands, max_dim)

    def __len__(self):
        return 10000 if self.is_train else len(self.scenes)

    def _random_crop(self, hsi, size):
        _, H, W = hsi.shape
        if H < size or W < size:
            pad_h = max(0, size - H)
            pad_w = max(0, size - W)
            hsi = np.pad(hsi, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            _, H, W = hsi.shape
        y = np.random.randint(0, H - size + 1)
        x = np.random.randint(0, W - size + 1)
        return hsi[:, y:y + size, x:x + size], y, x

    @staticmethod
    def _augment(hsi, msi):
        if np.random.random() < 0.5:
            hsi = hsi[:, :, ::-1].copy()
            msi = msi[:, :, ::-1].copy()
        if np.random.random() < 0.5:
            hsi = hsi[:, ::-1, :].copy()
            msi = msi[:, ::-1, :].copy()
        k = np.random.randint(0, 4)
        if k:
            hsi = np.rot90(hsi, k, axes=(-2, -1)).copy()
            msi = np.rot90(msi, k, axes=(-2, -1)).copy()
        return hsi, msi

    def __getitem__(self, idx):
        if self.is_train:
            name = list(self._cache.keys())[
                np.random.randint(0, len(self._cache))]
            gt = self._cache[name].copy()
            gt, _, _ = self._random_crop(gt, self.patch_size)
            hr_msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
            hr_msi = np.clip(hr_msi, 0.0, 1.0)
            gt, hr_msi = self._augment(gt, hr_msi)
            lr_hsi, _ = simulate_wald(gt, self.kernel, self.scale, self.srf)
        else:
            name, _ = self.scenes[idx % len(self.scenes)]
            gt = self._cache[name].copy()
            H, W = gt.shape[1], gt.shape[2]
            H = (H // self.scale) * self.scale
            W = (W // self.scale) * self.scale
            gt = gt[:, :H, :W]
            hr_msi = np.einsum("chw,cm->mhw", gt, self.srf).astype(np.float32)
            hr_msi = np.clip(hr_msi, 0.0, 1.0)
            lr_hsi, _ = simulate_wald(gt, self.kernel, self.scale, self.srf)
        return {
            "gt": torch.from_numpy(gt.astype(np.float32)),
            "lr_hsi": torch.from_numpy(lr_hsi.astype(np.float32)),
            "hr_msi": torch.from_numpy(hr_msi.astype(np.float32)),
            "scene_name": name,
        }


###############################################################################
# SSRNet Model (Adapted for Fusion)
###############################################################################

class SSRNetFusion(nn.Module):
    """SSRNet adapted for hyperspectral image fusion.

    Architecture keeps the spirit of the original SSRNet (spatial + spectral
    branches) but is redesigned for the fusion setting:

    Input: concat(bicubic_upsampled_LR_HSI, HR_MSI) -> [34, H, W]
           (31 bands upsampled + 3 MSI bands)
    Output: HR-HSI [31, H, W]

    The model has three branches:
    1. Fusion conv: initial feature extraction from concatenated input
    2. Spatial branch: refines spatial details
    3. Spectral branch: refines spectral information

    Each branch is a stack of Conv+ReLU layers with residual connections.
    """

    def __init__(self, n_bands=31, n_msi=3, n_feats=64, n_layers=4):
        super().__init__()
        self.n_bands = n_bands
        in_ch = n_bands + n_msi  # 31 + 3 = 34

        # Fusion branch: process concatenated (upsampled LR-HSI, HR-MSI)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Spatial branch: spatial detail refinement
        spat_layers = [nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True)]
        for _ in range(n_layers - 1):
            spat_layers.extend([
                nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
        self.spat_branch = nn.Sequential(*spat_layers)

        # Spectral branch: spectral detail refinement
        spec_layers = [nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True)]
        for _ in range(n_layers - 1):
            spec_layers.extend([
                nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])
        self.spec_branch = nn.Sequential(*spec_layers)

        # Combination of spatial + spectral features
        self.combo = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Output projection
        self.tail = nn.Conv2d(n_feats, n_bands, kernel_size=3, padding=1)

    def forward(self, lr_hsi, hr_msi):
        """
        Args:
            lr_hsi: [B, 31, H//4, W//4]  low-resolution HSI
            hr_msi: [B, 3, H, W]         high-resolution MSI
        Returns:
            out: [B, 31, H, W]  predicted HR-HSI
        """
        B, C, H, W = hr_msi.shape

        # Bicubic upsample LR-HSI to HR spatial dimensions
        lr_up = F.interpolate(lr_hsi, size=(H, W), mode='bicubic',
                              align_corners=False)

        # Concatenate upsampled LR-HSI and HR-MSI
        x = torch.cat([lr_up, hr_msi], dim=1)  # [B, 34, H, W]

        # Fusion head
        feat = self.head(x)  # [B, n_feats, H, W]

        # Spatial branch with residual
        spat = feat + self.spat_branch(feat)

        # Spectral branch with residual
        spec = feat + self.spec_branch(feat)

        # Combine spatial and spectral features
        combo = self.combo(torch.cat([spat, spec], dim=1))

        # Residual connection to upsampled LR-HSI
        out = self.tail(combo) + lr_up

        return out


###############################################################################
# Evaluation Metrics
###############################################################################

def calc_psnr(pred, gt):
    return -10.0 * torch.log10(
        torch.mean((pred - gt) ** 2) + 1e-8).item()


def calc_ssim(pred, gt, channels=31, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ax = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    _2d = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2)).unsqueeze(0).unsqueeze(0)
    _2d = _2d / _2d.sum()
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(gt, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    s2 = F.conv2d(gt * gt, window, padding=pad, groups=channels) - mu2_sq
    s12 = F.conv2d(pred * gt, window, padding=pad, groups=channels) - mu1_mu2
    cs = (2 * s12 + C2) / (s1 + s2 + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs
    return ssim_map.mean().item()


def calc_sam(pred, gt):
    B, C, H, W = pred.shape
    p = pred.view(B, C, -1)
    g = gt.view(B, C, -1)
    p = F.normalize(p, dim=1)
    g = F.normalize(g, dim=1)
    cos_sim = torch.clamp((p * g).sum(dim=1), -1.0, 1.0)
    sam = torch.acos(cos_sim)
    return sam.mean().item() * (180.0 / math.pi)


def calc_ergas(pred, gt, scale=4):
    B, C, H, W = pred.shape
    err = (pred - gt) ** 2
    ergas = 0.0
    for c in range(C):
        mg = gt[:, c].mean()
        if mg > 0:
            ergas += err[:, c].mean() / (mg ** 2)
    return math.sqrt(ergas / C) * 100.0 * scale


###############################################################################
# Training
###############################################################################

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"SM: {torch.cuda.get_device_capability(0)}")

    print("\nLoading datasets...")
    train_ds = CAVEDataset(
        args.root, split="train", bands=args.bands, scale=args.scale,
        patch_size=args.patch_size, max_dim=args.max_dim)
    test_ds = CAVEDataset(
        args.root, split="test", bands=args.bands, scale=args.scale,
        patch_size=0, max_dim=args.max_dim)
    print(f"Train scenes: {len(train_ds.scenes)}, "
          f"Test scenes: {len(test_ds.scenes)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    iter_loader = iter(train_loader)

    print("\nBuilding SSRNet Fusion model...")
    model = SSRNetFusion(
        n_bands=args.bands, n_msi=3, n_feats=64, n_layers=4).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = nn.L1Loss()

    best_psnr = 0.0
    best_epoch = 0
    save_dir = os.path.join(args.output_dir, "ssrnet_checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch: {args.batch_size}, Patch: {args.patch_size}, LR: {args.lr}")
    print(f"Eval every {args.eval_every} epochs on "
          f"{len(test_ds.scenes)} test scenes")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(args.steps_per_epoch):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            gt = batch["gt"].to(device)
            lr_hsi = batch["lr_hsi"].to(device)
            hr_msi = batch["hr_msi"].to(device)

            output = model(lr_hsi, hr_msi)
            output = output.clip(0, 1)

            loss = criterion(output, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / args.steps_per_epoch
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} | LR: {lr_now:.2e} | "
                  f"Time: {elapsed:.1f}s")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            psnr_l, ssim_l, sam_l, ergas_l = [], [], [], []

            with torch.no_grad():
                for i in range(len(test_ds.scenes)):
                    item = test_ds[i]
                    gt_e = item["gt"].unsqueeze(0).to(device)
                    lr_e = item["lr_hsi"].unsqueeze(0).to(device)
                    hr_e = item["hr_msi"].unsqueeze(0).to(device)
                    pred = model(lr_e, hr_e).clip(0, 1)
                    psnr_l.append(calc_psnr(pred, gt_e))
                    ssim_l.append(calc_ssim(pred, gt_e))
                    sam_l.append(calc_sam(pred, gt_e))
                    ergas_l.append(calc_ergas(pred, gt_e, args.scale))

            avg_psnr = np.mean(psnr_l)
            avg_ssim = np.mean(ssim_l)
            avg_sam = np.mean(sam_l)
            avg_ergas = np.mean(ergas_l)
            print(f"  [Eval] PSNR: {avg_psnr:.4f} dB | "
                  f"SSIM: {avg_ssim:.4f} | "
                  f"SAM: {avg_sam:.4f} deg | "
                  f"ERGAS: {avg_ergas:.4f}")

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                ckpt = os.path.join(save_dir, "ssrnet_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "psnr": avg_psnr, "ssim": avg_ssim,
                    "sam": avg_sam, "ergas": avg_ergas,
                }, ckpt)
                print(f"  >> New best! Saved to {ckpt}")

    print("\n" + "=" * 70)
    print(f"Training complete! Best PSNR: {best_psnr:.4f} dB "
          f"at epoch {best_epoch}")
    print("=" * 70)


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SSRNet Fusion Training for CAVE x4")
    parser.add_argument(
        "--root", type=str,
        default="/kaggle/input/datasets/liptee/"
                "hyperspectral-image-restoration-based-on-cave",
        help="Dataset root directory")
    parser.add_argument("--bands", type=int, default=31)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=80)
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--output_dir", type=str,
        default="/kaggle/working",
        help="Directory to save checkpoints")
    args = parser.parse_args()
    train(args)
