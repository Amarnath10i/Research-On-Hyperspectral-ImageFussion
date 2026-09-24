"""
BDT (Bidirectional Dilation Transformer) — CAVE x4 fusion training script.
IJCAI 2023, reported 52.30 dB.

Kaggle self-contained: model architecture inlined, dataset loaded from
/kaggle/input/datasets/liptee/hyperspectral-image-restoration-based-on-cave

Usage on Kaggle:
    pip install timm einops
    python train_bdt.py
"""

import os, sys, glob, time, math, random, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    os.system("pip install timm -q")
    from timm.models.layers import DropPath, trunc_normal_

try:
    from einops import rearrange, repeat
except ImportError:
    os.system("pip install einops -q")
    from einops import rearrange, repeat

try:
    from PIL import Image
except ImportError:
    os.system("pip install pillow -q")
    from PIL import Image

# ============================================================================
# Configuration
# ============================================================================
DATA_ROOT   = "/kaggle/input/datasets/liptee/hyperspectral-image-restoration-based-on-cave"
OUTPUT_DIR  = "./bdt_cave_x4_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE    = 128
SCALE       = 4
LR_IMG_SIZE = IMG_SIZE // SCALE
NUM_BANDS   = 31
PATCH_SIZE  = 128
BATCH_SIZE  = 8
NUM_EPOCHS  = 2000
LR          = 1e-4
WEIGHT_DECAY = 1e-4
EVAL_EVERY  = 50
SEED        = 42

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# Nikon D700 SRF — HR-HSI -> HR-MSI
# ============================================================================
NIKON_D700_SRF = np.array([
    [0.0022, 0.0044, 0.0065], [0.0056, 0.0097, 0.0115], [0.0120, 0.0180, 0.0190],
    [0.0230, 0.0310, 0.0300], [0.0420, 0.0500, 0.0430], [0.0680, 0.0740, 0.0570],
    [0.1040, 0.1020, 0.0730], [0.1510, 0.1400, 0.0870], [0.2020, 0.1810, 0.1010],
    [0.2520, 0.2190, 0.1110], [0.2940, 0.2520, 0.1180], [0.3250, 0.2800, 0.1220],
    [0.3480, 0.3050, 0.1250], [0.3600, 0.3270, 0.1270], [0.3660, 0.3450, 0.1280],
    [0.3650, 0.3590, 0.1280], [0.3570, 0.3700, 0.1280], [0.3420, 0.3760, 0.1280],
    [0.3210, 0.3780, 0.1290], [0.2950, 0.3750, 0.1310], [0.2640, 0.3680, 0.1360],
    [0.2300, 0.3560, 0.1450], [0.1940, 0.3390, 0.1610], [0.1590, 0.3160, 0.1860],
    [0.1270, 0.2880, 0.2220], [0.0990, 0.2560, 0.2690], [0.0760, 0.2210, 0.3260],
    [0.0570, 0.1840, 0.3820], [0.0420, 0.1480, 0.4320], [0.0300, 0.1150, 0.4660],
    [0.0210, 0.0870, 0.4860],
], dtype=np.float32)
NIKON_D700_SRF /= NIKON_D700_SRF.sum(axis=0, keepdims=True)
SRF_TORCH = torch.from_numpy(NIKON_D700_SRF).to(DEVICE)  # (31, 3)

# ============================================================================
# Dataset
# ============================================================================
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


def find_scenes(root):
    """Walk root to find scene folders containing band_*.png OR .mat files."""
    scenes = []
    for dirpath, dirnames, filenames in os.walk(root):
        mat_files = [f for f in filenames if f.endswith('.mat')]
        if mat_files:
            scene_name = os.path.basename(dirpath)
            mat_paths = [os.path.join(dirpath, f) for f in mat_files]
            scenes.append((scene_name, dirpath, mat_paths))
            # don't recurse deeper into this scene directory
            dirnames[:] = []
            continue
        band_patterns = ["band_01.png", "band_01.tif", "band_01.bmp",
                         "band_01.jpg", "01.png", "01.tif"]
        for pat in band_patterns:
            if pat in filenames:
                scene_name = os.path.basename(dirpath)
                bands = []
                for b in range(1, NUM_BANDS + 1):
                    for ext in ["png", "tif", "bmp", "jpg"]:
                        bp = os.path.join(dirpath, f"band_{b:02d}.{ext}")
                        if os.path.exists(bp):
                            bands.append(bp)
                            break
                if len(bands) == NUM_BANDS:
                    scenes.append((scene_name, dirpath, sorted(bands)))
                break
    scenes.sort(key=lambda x: x[0])
    return scenes


def load_scene_bands(band_paths):
    """Load scene: .mat file -> (31, H, W) or band_*.png -> (31, H, W) float32 [0,1]."""
    if len(band_paths) == 1 and band_paths[0].endswith('.mat'):
        return _load_mat(band_paths[0])
    bands = []
    for p in band_paths:
        img = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        bands.append(img)
    return np.stack(bands, axis=0)


def make_gaussian_kernel(ksize, sigma):
    ax = np.arange(ksize) - ksize // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def degrade_hsi(hr_hsi, scale, sigma):
    """Wald's protocol: Gaussian blur + downsample."""
    C, H, W = hr_hsi.shape
    ksize = 9
    kern = make_gaussian_kernel(ksize, sigma)
    pad = ksize // 2
    hr_padded = np.pad(hr_hsi, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    lr = np.zeros((C, H // scale, W // scale), dtype=np.float32)
    for c in range(C):
        for i in range(H // scale):
            for j in range(W // scale):
                patch = hr_padded[c, i * scale:i * scale + ksize, j * scale:j * scale + ksize]
                lr[c, i, j] = np.sum(patch * kern)
    return lr


def hsi_to_rgb(hsi, srf):
    """HSI (31, H, W) -> RGB (3, H, W) via SRF einsum."""
    return np.einsum("chw,cd->dhw", hsi, srf).astype(np.float32)


class CAVEDataset(Dataset):
    def __init__(self, scenes, patch_size, scale, sigma=1.2, augment=True):
        self.scenes = scenes
        self.patch_size = patch_size
        self.scale = scale
        self.sigma = sigma
        self.augment = augment
        self.srf = NIKON_D700_SRF
        self.hr_cache = {}

    def __len__(self):
        return len(self.scenes) * 100  # ~100 crops per scene

    def _load_hr(self, idx):
        if idx not in self.hr_cache:
            self.hr_cache[idx] = load_scene_bands(self.scenes[idx][2])
        return self.hr_cache[idx].copy()

    def __getitem__(self, index):
        scene_idx = index % len(self.scenes)
        hr_hsi = self._load_hr(scene_idx)
        C, H, W = hr_hsi.shape
        ps = self.patch_size

        if self.augment:
            ry = random.randint(0, max(0, H - ps))
            rx = random.randint(0, max(0, W - ps))
            hr_hsi = hr_hsi[:, ry:ry + ps, rx:rx + ps]
            if random.random() > 0.5:
                hr_hsi = hr_hsi[:, :, ::-1].copy()
            if random.random() > 0.5:
                hr_hsi = hr_hsi[:, ::-1, :].copy()
        else:
            ry = (H - ps) // 2
            rx = (W - ps) // 2
            hr_hsi = hr_hsi[:, ry:ry + ps, rx:rx + ps]

        lr_hsi = degrade_hsi(hr_hsi, self.scale, self.sigma)
        rgb = hsi_to_rgb(hr_hsi, self.srf)

        # Bicubic upsample LR-HSI to HR resolution
        lr_t = torch.from_numpy(lr_hsi).unsqueeze(0)  # (1, C, lr_h, lr_w)
        lr_up = F.interpolate(lr_t, size=(ps, ps), mode="bicubic", align_corners=False)
        lr_up = lr_up.squeeze(0).numpy()

        hr_t = torch.from_numpy(hr_hsi).float()
        rgb_t = torch.from_numpy(rgb).float()
        lr_up_t = torch.from_numpy(lr_up).float()
        lr_t2 = torch.from_numpy(lr_hsi).float()

        return {
            "gt": hr_t,          # (31, H, W)
            "rgb": rgb_t,        # (3, H, W)
            "lr_up": lr_up_t,    # (31, H, W)
            "lr": lr_t2,         # (31, H/4, W/4)
            "name": self.scenes[scene_idx][0],
        }


# ============================================================================
# BDT Model — Bidirection (from bidirection.py)
# ============================================================================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def Win_Dila(x, win_size):
    n_win = win_size * 2
    B, H, W, C = x.shape
    x = x.reshape(-1, (H // n_win), n_win, (W // n_win), n_win, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    xt = torch.zeros_like(x)
    x0 = x[:, :, :, 0::2, 0::2, :]
    x1 = x[:, :, :, 0::2, 1::2, :]
    x2 = x[:, :, :, 1::2, 0::2, :]
    x3 = x[:, :, :, 1::2, 1::2, :]
    half = n_win // 2
    xt[:, :, :, 0:half, 0:half, :] = x0
    xt[:, :, :, 0:half, half:n_win, :] = x1
    xt[:, :, :, half:n_win, 0:half, :] = x2
    xt[:, :, :, half:n_win, half:n_win, :] = x3
    xt = xt.permute(0, 1, 3, 2, 4, 5).reshape(-1, H, W, C)
    return xt


def Win_ReDila(x, win_size):
    n_win = win_size * 2
    B, H, W, C = x.shape
    x = x.reshape(-1, (H // n_win), n_win, (W // n_win), n_win, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    xt = torch.zeros_like(x)
    half = n_win // 2
    xt[:, :, :, 0::2, 0::2, :] = x[:, :, :, 0:half, 0:half, :]
    xt[:, :, :, 0::2, 1::2, :] = x[:, :, :, 0:half, half:n_win, :]
    xt[:, :, :, 1::2, 0::2, :] = x[:, :, :, half:n_win, 0:half, :]
    xt[:, :, :, 1::2, 1::2, :] = x[:, :, :, half:n_win, half:n_win, :]
    xt = xt.permute(0, 1, 3, 2, 4, 5).reshape(-1, H, W, C)
    return xt


class BDT_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BDT_Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BDT_Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class BDT_Stage(nn.Module):
    def __init__(self, dim=32, input_resolution=(16, 16), num_heads=8,
                 window_size=4, mlp_ratio=4.0, qkv_bias=True, qk_scale=4,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        H, W = input_resolution
        self.attn1 = BDT_Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.attn2 = BDT_Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.attn3 = BDT_Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp1 = BDT_Mlp(dim, mlp_hidden, drop=drop)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = BDT_Mlp(dim, mlp_hidden, drop=drop)
        self.norm5 = nn.LayerNorm(dim)
        self.norm6 = nn.LayerNorm(dim)
        self.mlp3 = BDT_Mlp(dim, mlp_hidden, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        x_w = window_partition(x, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        x_w = self.attn1(x_w)
        x = window_reverse(x_w, self.window_size, H, W).view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp1(self.norm2(x)))

        shortcut = x
        x = self.norm3(x).view(B, H, W, C)
        shifted = Win_Dila(x, self.window_size)
        x_w = window_partition(shifted, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        x_w = self.attn2(x_w)
        shifted = window_reverse(x_w, self.window_size, H, W)
        x = Win_ReDila(shifted, self.window_size).view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp2(self.norm4(x)))

        shortcut = x
        x = self.norm5(x).view(B, H, W, C)
        x_w = window_partition(x, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        x_w = self.attn3(x_w)
        x = window_reverse(x_w, self.window_size, H, W).view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp3(self.norm6(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


def _to3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def _to4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        ns = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(ns))
        self.bias = nn.Parameter(torch.zeros(ns))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class BDT_LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        self.body = (BiasFree_LayerNorm(dim) if LayerNorm_type == "BiasFree"
                     else WithBias_LayerNorm(dim))

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to4d(self.body(_to3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class XCAttention(nn.Module):
    def __init__(self, dim, num_heads, win, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.win = win
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        ws = self.win
        q = q.view(B, C, H // ws, ws, W // ws, ws).permute(0, 2, 4, 1, 3, 5
            ).contiguous().view(-1, C, ws * ws)
        k = k.view(B, C, H // ws, ws, W // ws, ws).permute(0, 2, 4, 1, 3, 5
            ).contiguous().view(-1, C, ws * ws)
        v = v.view(B, C, H // ws, ws, W // ws, ws).permute(0, 2, 4, 1, 3, 5
            ).contiguous().view(-1, C, ws * ws)
        q = rearrange(q, "n (head c) win -> n head c win", head=self.num_heads)
        k = rearrange(k, "n (head c) win -> n head c win", head=self.num_heads)
        v = rearrange(v, "n (head c) win -> n head c win", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, "n head c (h w) -> n (head c) h w", w=ws)
        out = out.view(B, H // ws, W // ws, C, ws, ws
            ).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        return self.project_out(out)


class XCABlock(nn.Module):
    def __init__(self, dim, num_heads, win, ffn_expansion_factor, bias,
                 LayerNorm_type):
        super().__init__()
        self.norm1 = BDT_LayerNorm(dim, LayerNorm_type)
        self.attn = XCAttention(dim, num_heads, win, bias)
        self.norm2 = BDT_LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# --- Direction1: W-MSA encoder (processes cat(lms, rgb)) ---
class Direction1(nn.Module):
    def __init__(self, img_size=64, in_chans=34, embed_dim=96,
                 num_heads=[8, 8, 8], window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.conv = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.downsample1 = PatchMerging((img_size, img_size), embed_dim)
        self.downsample2 = PatchMerging((img_size // 2, img_size // 2), embed_dim * 2)
        self.stage1 = BDT_Stage(embed_dim, (img_size, img_size),
                                num_heads[0], window_size)
        self.stage2 = BDT_Stage(embed_dim * 2, (img_size // 2, img_size // 2),
                                num_heads[1], window_size)
        self.stage3 = BDT_Stage(embed_dim * 4, (img_size // 4, img_size // 4),
                                num_heads[2], window_size)

    def forward(self, x):
        B, _, H, _ = x.shape
        x = self.conv(x)
        x = rearrange(x, "B C H W -> B (H W) C")
        s1 = self.stage1(x)
        s1_4d = rearrange(s1, "B (H W) C -> B C H W", H=H)
        s1 = self.downsample1(s1)
        H2 = H // 2
        s2 = self.stage2(s1)
        s2_4d = rearrange(s2, "B (H W) C -> B C H W", H=H2)
        s2 = self.downsample2(s2)
        H4 = H2 // 2
        s3 = self.stage3(s2)
        s3_4d = rearrange(s3, "B (H W) C -> B C H W", H=H4)
        return s1_4d, s2_4d, s3_4d


# --- Direction2: XCA decoder (processes LR-HSI) ---
class Direction2(nn.Module):
    def __init__(self, inp_channels=31, dim=32, num_heads=[8, 8, 8], group=8,
                 ffn_expansion_factor=2.66, LayerNorm_type="WithBias", bias=False):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.stage1 = XCABlock(dim, num_heads[0], group, ffn_expansion_factor,
                               bias, LayerNorm_type)
        self.up1_2 = BDT_Upsample(dim)
        self.stage2 = XCABlock(dim, num_heads[1], group, ffn_expansion_factor,
                               bias, LayerNorm_type)
        self.up2_3 = BDT_Upsample(dim)
        self.stage3 = XCABlock(dim, num_heads[2], group, ffn_expansion_factor,
                               bias, LayerNorm_type)

    def forward(self, x):
        e = self.patch_embed(x)
        o1 = self.stage1(e)
        o1_up = self.up1_2(o1)
        o2 = self.stage2(o1_up)
        o2_up = self.up2_3(o2)
        o3 = self.stage3(o2_up)
        return o3, o2, o1


# --- Merge module ---
class BDT_Merge(nn.Module):
    def __init__(self, img_size=64, in_chans1=34, in_chans2=31, embed_dim=48,
                 num_heads1=[8, 8, 8], window_size=8, group=8, mlp_ratio=4.0,
                 dim=32, num_heads2=[8, 8, 8], ffn_expansion_factor=2.66,
                 LayerNorm_type="WithBias", bias=False):
        super().__init__()
        self.direction1 = Direction1(img_size, in_chans1, embed_dim,
                                     num_heads1, window_size, mlp_ratio)
        self.direction2 = Direction2(in_chans2, dim, num_heads2, group,
                                     ffn_expansion_factor, LayerNorm_type, bias)
        self.merge1 = nn.Sequential(
            nn.Conv2d(embed_dim * 4 + dim, dim, 3, 1, 1, bias=bias),
            nn.Conv2d(dim, dim, 5, 2, 2, bias=bias),
            nn.LeakyReLU(0.2, inplace=True))
        self.up1 = BDT_Upsample(dim)
        self.merge2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2 + dim + dim, dim, 3, 1, 1, bias=bias),
            nn.Conv2d(dim, dim, 5, 2, 2, bias=bias),
            nn.LeakyReLU(0.2, inplace=True))
        self.up2 = BDT_Upsample(dim)
        self.merge3 = nn.Sequential(
            nn.Conv2d(embed_dim + dim + dim, in_chans2, 3, 1, 1, bias=bias),
            nn.Conv2d(in_chans2, in_chans2, 5, 2, 2, bias=bias))

    def forward(self, x, y):
        d1s1, d1s2, d1s3 = self.direction1(x)
        d2s3, d2s2, d2s1 = self.direction2(y)
        m1 = self.merge1(torch.cat((d1s3, d2s1), 1))
        m2 = self.merge2(torch.cat((d1s2, d2s2, self.up1(m1)), 1))
        return self.merge3(torch.cat((d1s1, d2s3, self.up2(m2)), 1))


# ============================================================================
# BDT Model — Bidinet wrapper (from model_SR_x4.py)
# ============================================================================
class BDT_Bidinet(nn.Module):
    """
    BDT for hyperspectral image fusion.

    forward(rgb, lms, ms):
        rgb : HR-MSI      (B, 3,  H,  W)
        lms : up-LR-HSI   (B, 31, H,  W)
        ms  : LR-HSI      (B, 31, H/4, W/4)
    -> output: reconstructed HR-HSI (B, 31, H, W)
    """

    def __init__(self, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.merge = BDT_Merge(
            img_size=img_size, in_chans1=34, in_chans2=31,
            embed_dim=48, num_heads1=[8, 8, 8], window_size=8, group=8,
            mlp_ratio=4.0, dim=48, num_heads2=[8, 8, 8],
            ffn_expansion_factor=2.66, LayerNorm_type="WithBias", bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rgb, lms, ms):
        cat_input = torch.cat((lms, rgb), 1)  # (B, 34, H, W)
        out = self.merge(cat_input, ms)        # (B, 31, H, W)
        return out + lms                       # residual connection


# ============================================================================
# Metrics
# ============================================================================
def calc_psnr(pred, ref):
    """PSNR per band, then mean."""
    err = 0.0
    n = pred.shape[0]
    for i in range(n):
        mse = np.mean((pred[i] - ref[i]) ** 2)
        if mse < 1e-10:
            err += 100.0
        else:
            err += 10.0 * math.log10(1.0 / mse)
    return err / n


def calc_ssim(pred, ref, win_size=11):
    """SSIM per band, then mean."""
    def _ssim_band(p, r):
        p = p.astype(np.float64)
        r = r.astype(np.float64)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        h, w = p.shape
        ssim_sum, cnt = 0.0, 0
        for i in range(0, h - win_size + 1, 3):
            for j in range(0, w - win_size + 1, 3):
                pw = p[i:i + win_size, j:j + win_size]
                rw = r[i:i + win_size, j:j + win_size]
                mu_p, mu_r = pw.mean(), rw.mean()
                sig_p, sig_r = pw.var(), rw.var()
                sig_pr = ((pw - mu_p) * (rw - mu_r)).mean()
                num = (2 * mu_p * mu_r + c1) * (2 * sig_pr + c2)
                den = (mu_p ** 2 + mu_r ** 2 + c1) * (sig_p + sig_r + c2)
                ssim_sum += num / den
                cnt += 1
        return ssim_sum / max(cnt, 1)

    n = pred.shape[0]
    return np.mean([_ssim_band(pred[i], ref[i]) for i in range(n)])


def calc_sam(pred, ref):
    """Spectral Angle Mapper (degrees)."""
    n_bands, h, w = pred.shape
    pred_r = pred.reshape(n_bands, -1).astype(np.float64)
    ref_r = ref.reshape(n_bands, -1).astype(np.float64)
    dot = np.sum(pred_r * ref_r, axis=0)
    norm_p = np.sqrt(np.sum(pred_r ** 2, axis=0)) + 1e-12
    norm_r = np.sqrt(np.sum(ref_r ** 2, axis=0)) + 1e-12
    cos_angle = np.clip(dot / (norm_p * norm_r), -1.0, 1.0)
    return float(np.mean(np.arccos(cos_angle)) * 180.0 / math.pi)


def calc_ergas(pred, ref, scale=4):
    """ERGAS (lower is better)."""
    n_bands = pred.shape[0]
    err = 0.0
    for b in range(n_bands):
        mse = np.mean((pred[b] - ref[b]) ** 2)
        mu = ref[b].mean()
        if mu < 1e-10:
            continue
        err += mse / (mu ** 2)
    return 100.0 * scale * math.sqrt(err / n_bands)


# ============================================================================
# Training
# ============================================================================
def build_model():
    model = BDT_Bidinet(img_size=IMG_SIZE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[BDT] Total parameters: {n_params / 1e6:.3f} M")
    return model.to(DEVICE)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        gt   = batch["gt"].to(DEVICE)
        rgb  = batch["rgb"].to(DEVICE)
        lr_up = batch["lr_up"].to(DEVICE)
        lr   = batch["lr"].to(DEVICE)

        optimizer.zero_grad()
        out = model(rgb, lr_up, lr)
        loss = criterion(out, gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, test_scenes, scale=4, sigma=1.2):
    """Evaluate on all test scenes. Returns dict of metric arrays."""
    model.eval()
    all_psnr, all_ssim, all_sam, all_ergas = [], [], [], []

    for name, _, band_paths in test_scenes:
        hr_hsi = load_scene_bands(band_paths)  # (31, H, W)
        C, H, W = hr_hsi.shape

        # Pad to multiple of IMG_SIZE for tiled processing
        pad_h = (IMG_SIZE - H % IMG_SIZE) % IMG_SIZE
        pad_w = (IMG_SIZE - W % IMG_SIZE) % IMG_SIZE
        if pad_h > 0 or pad_w > 0:
            hr_hsi = np.pad(hr_hsi, ((0, 0), (0, pad_h), (0, pad_w)),
                            mode="reflect")

        _, Hp, Wp = hr_hsi.shape
        out_buf = np.zeros_like(hr_hsi)

        for y in range(0, Hp, IMG_SIZE):
            for x in range(0, Wp, IMG_SIZE):
                patch_hr = hr_hsi[:, y:y + IMG_SIZE, x:x + IMG_SIZE]
                lr_patch = degrade_hsi(patch_hr, scale, sigma)
                rgb_patch = hsi_to_rgb(patch_hr, NIKON_D700_SRF)

                lr_t = torch.from_numpy(lr_patch).unsqueeze(0).to(DEVICE)
                lr_up_t = F.interpolate(lr_t, size=(IMG_SIZE, IMG_SIZE),
                                        mode="bicubic", align_corners=False)
                rgb_t = torch.from_numpy(rgb_patch).unsqueeze(0).to(DEVICE)

                pred = model(rgb_t, lr_up_t, lr_t)  # (1, 31, IMG_SIZE, IMG_SIZE)
                out_buf[:, y:y + IMG_SIZE, x:x + IMG_SIZE] = \
                    pred.squeeze(0).cpu().numpy()

        out_buf = out_buf[:, :H, :W]
        hr_ref = hr_hsi[:, :H, :W]

        all_psnr.append(calc_psnr(out_buf, hr_ref))
        all_ssim.append(calc_ssim(out_buf, hr_ref))
        all_sam.append(calc_sam(out_buf, hr_ref))
        all_ergas.append(calc_ergas(out_buf, hr_ref, scale))

    return {
        "psnr": np.array(all_psnr),
        "ssim": np.array(all_ssim),
        "sam": np.array(all_sam),
        "ergas": np.array(all_ergas),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("BDT — CAVE x4 Fusion Training")
    print("=" * 60)

    # --- Discover dataset ---
    all_scenes = find_scenes(DATA_ROOT)
    print(f"Found {len(all_scenes)} scenes with {NUM_BANDS} bands each.")
    if len(all_scenes) == 0:
        print("ERROR: No scenes found. Check DATA_ROOT path.")
        print(f"  DATA_ROOT = {DATA_ROOT}")
        sys.exit(1)

    # --- Split train / test ---
    n_test = min(8, len(all_scenes) // 4)
    n_test = max(n_test, 1)
    # Use last n_test scenes as test
    test_scenes = all_scenes[-n_test:]
    train_scenes = all_scenes[:-n_test]
    print(f"Train scenes: {len(train_scenes)}, Test scenes: {len(test_scenes)}")
    print("Test:", [s[0] for s in test_scenes])

    # --- Data loaders ---
    train_ds = CAVEDataset(train_scenes, PATCH_SIZE, SCALE, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)

    # --- Model ---
    model = build_model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- Baseline metrics (bicubic upsample) ---
    print("\n--- Baseline (bicubic upsample only) ---")
    base_psnr, base_ssim, base_sam, base_ergas = [], [], [], []
    for name, _, band_paths in test_scenes:
        hr = load_scene_bands(band_paths)
        lr = degrade_hsi(hr, SCALE, 1.2)
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        up = F.interpolate(lr_t, size=(hr.shape[1], hr.shape[2]),
                           mode="bicubic", align_corners=False).squeeze(0).numpy()
        base_psnr.append(calc_psnr(up, hr))
        base_ssim.append(calc_ssim(up, hr))
        base_sam.append(calc_sam(up, hr))
        base_ergas.append(calc_ergas(up, hr, SCALE))
    print(f"  PSNR={np.mean(base_psnr):.4f}  SSIM={np.mean(base_ssim):.4f}  "
          f"SAM={np.mean(base_sam):.4f}  ERGAS={np.mean(base_ergas):.4f}")

    # --- Training loop ---
    best_psnr = 0.0
    best_epoch = 0
    best_path = os.path.join(OUTPUT_DIR, "best_bdt.pth")
    print("\n--- Training ---")
    print(f"Epochs={NUM_EPOCHS}, BatchSize={BATCH_SIZE}, LR={LR}, "
          f"EvalEvery={EVAL_EVERY} epochs")

    for epoch in range(1, NUM_EPOCHS + 1):
        t_ep = time.time()
        avg_loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        elapsed = time.time() - t_ep
        if epoch == 1 or epoch % 20 == 0 or epoch == NUM_EPOCHS:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>5d}/{NUM_EPOCHS}  loss={avg_loss:.6f}  "
                  f"lr={lr_now:.2e}  {elapsed:.1f}s")

        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate(model, test_scenes, SCALE, 1.2)
            m_psnr = metrics["psnr"].mean()
            m_ssim = metrics["ssim"].mean()
            m_sam  = metrics["sam"].mean()
            m_ergas = metrics["ergas"].mean()

            improved = ""
            if m_psnr > best_psnr:
                best_psnr = m_psnr
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "psnr": m_psnr,
                    "ssim": m_ssim,
                    "sam": m_sam,
                    "ergas": m_ergas,
                }, best_path)
                improved = " [BEST]"

            print(f"  >>> Test @ epoch {epoch}: PSNR={m_psnr:.4f}  "
                  f"SSIM={m_ssim:.4f}  SAM={m_sam:.4f}  ERGAS={m_ergas:.4f}"
                  f"  (best={best_psnr:.4f}@{best_epoch}){improved}")

    # --- Final evaluation from best checkpoint ---
    print("\n" + "=" * 60)
    print("Final Results from Best Checkpoint")
    print("=" * 60)
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    final = evaluate(model, test_scenes, SCALE, 1.2)

    print(f"\nBest epoch : {ckpt['epoch']}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    print(f"\n{'Scene':<20s} {'PSNR':>8s} {'SSIM':>8s} {'SAM':>8s} {'ERGAS':>8s}")
    print("-" * 52)
    for i, (name, _, _) in enumerate(test_scenes):
        print(f"{name:<20s} {final['psnr'][i]:>8.4f} {final['ssim'][i]:>8.4f} "
              f"{final['sam'][i]:>8.4f} {final['ergas'][i]:>8.4f}")
    print("-" * 52)
    print(f"{'Mean':<20s} {final['psnr'].mean():>8.4f} {final['ssim'].mean():>8.4f} "
          f"{final['sam'].mean():>8.4f} {final['ergas'].mean():>8.4f}")

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time / 3600:.2f} hours")
    print(f"Best checkpoint: {best_path}")
    print("Done.")


if __name__ == "__main__":
    main()
