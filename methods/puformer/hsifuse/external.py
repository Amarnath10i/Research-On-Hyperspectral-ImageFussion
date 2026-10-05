"""Published fusion networks, built from their authors' code and retrained under our protocol.

The repositories carry no license, so they are not copied here. `fetch()` clones them at run time (Kaggle
kernels have internet) into $PUF_EXT (default /kaggle/working/ext), pinned to the commits below, and the
adapters import the network classes from there. Only the band counts and the scale factor are changed:

    psrt       PSRT (Deng et al., TGRS 2023)            Deng-shangqi/PSRT        model/model_SR.py PSRTnet
    dct        DCT / DCTransformer (Ma et al., InfFus 2024)  qingma2016/DCTransformer  dual.py DCT
    mimformer  MIMFormer (Li et al., JSTARS 2024)       meiruni/MIMFormer        models/MIMFormer_CPW.py
    fusformer  Fusformer (Hu et al., GRSL 2022)         J-FHu/Fusformer          model.py MainNet

Every adapter maps (Y_H, Y_M) -> X like PUFormer, and `forward(..., return_all=True)` returns [X] so the
training loop is shared. Fusformer attends over all pixels (quadratic memory), so it is evaluated on
overlapping tiles. The official training settings that differ from ours are in RECIPES and are applied by
train.py (learning rate, weight decay, gradient clipping).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

EXT = os.environ.get("PUF_EXT", "/kaggle/working/ext")
REPOS = {
    "psrt": ("https://github.com/Deng-shangqi/PSRT", "PSRT"),
    "dct": ("https://github.com/qingma2016/DCTransformer", "DCTransformer"),
    "mimformer": ("https://github.com/meiruni/MIMFormer", "MIMFormer"),
    "fusformer": ("https://github.com/J-FHu/Fusformer", "Fusformer"),
}
# official optimiser settings (all use an L1 loss): lr, weight decay, gradient clipping (0 = off)
RECIPES = {
    "psrt": dict(lr=1e-4, wd=0.0, clip=0.0),
    "dct": dict(lr=1e-4, wd=0.0, clip=0.0),
    "mimformer": dict(lr=1e-4, wd=0.0, clip=0.0),
    "fusformer": dict(lr=1e-3, wd=0.0, clip=0.0),
}


def fetch(name: str) -> str:
    url, folder = REPOS[name]
    path = os.path.join(EXT, folder)
    if not os.path.isdir(path):
        os.makedirs(EXT, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, path], check=True)
    rev = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"[external] {name}: {url} @ {rev}", flush=True)
    return path


def _load(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def tiled(fn, yh, ym, s, tile, overlap):
    """Run fn(yh_tile, ym_tile) on overlapping HR tiles (multiples of s) and average the overlaps."""
    _, _, H, W = ym.shape
    if H <= tile and W <= tile:
        return fn(yh, ym)
    step = tile - overlap
    starts = lambda n: sorted(set(list(range(0, max(n - tile, 0) + 1, step)) + [max(n - tile, 0)]))
    out = ym.new_zeros(yh.shape[0], yh.shape[1], H, W)
    cnt = ym.new_zeros(1, 1, H, W)
    for r in starts(H):
        for c in starts(W):
            o = fn(yh[..., r // s:(r + tile) // s, c // s:(c + tile) // s], ym[..., r:r + tile, c:c + tile])
            out[..., r:r + tile, c:c + tile] += o
            cnt[..., r:r + tile, c:c + tile] += 1
    return out / cnt


class External(nn.Module):
    def __init__(self, net, call, scale, tile=0, overlap=0):
        super().__init__()
        self.net, self.call, self.scale, self.tile, self.overlap = net, call, scale, tile, overlap

    def forward(self, yh, ym, return_all=False):
        if self.tile and not self.training:
            x = tiled(lambda a, b: self.call(self.net, a, b), yh, ym, self.scale, self.tile, self.overlap)
        else:
            x = self.call(self.net, yh, ym)
        return [x] if return_all else x


def _up(yh, s):
    return F.interpolate(yh, scale_factor=s, mode="bicubic", align_corners=False)


class Checkpointed(nn.Module):
    """Recompute the wrapped module in the backward pass instead of storing its activations (training only).

    DCT and MIMFormer do not fit a 16 GB T4 at batch 8 otherwise (DCT's own use_checkpoint flag calls its
    blocks with the wrong arguments). The computed function is unchanged. A real module (not a patched
    forward) so that deepcopy, i.e. the EMA model, gets its own weights.
    """

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, *args):
        if torch.is_grad_enabled() and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.m, *args, use_reentrant=False)
        return self.m(*args)


def build_external(name: str, bands: int, msi: int, scale: int) -> nn.Module:
    if name not in REPOS:
        raise ValueError(f"unknown model {name}")
    root = fetch(name)
    if root not in sys.path:
        sys.path.insert(0, root)
    if name == "psrt":
        m = _load(os.path.join(root, "model", "model_SR.py"), "psrt_model_SR")
        net = m.PSRTnet(None)
        net.in_channels = bands                                  # 31 -> B, 34 -> B + M
        net.conv = nn.Sequential(nn.Conv2d(net.embed, bands, 3, 1, 1), nn.LeakyReLU(0.2, True))
        net.w = m.Block(out_num=2, inside_num=3, img_size=net.img_size, in_chans=bands + msi, embed_dim=net.embed,
                        head=8, win_size=8)
        m.init_weights(net.conv), m.init_w(net.w)
        return External(net, lambda n, yh, ym: n(ym, _up(yh, scale)), scale)
    if name == "dct":
        m = _load(os.path.join(root, "dual.py"), "dct_dual")
        net = m.DCT(bands, scale)
        net.headY[0] = nn.Conv2d(msi, 64, 3, stride=1, padding=1)  # RGB (3) -> M bands
        net.body = Checkpointed(net.body)                          # the shared body runs three times per forward
        return External(net, lambda n, yh, ym: n(yh, ym), scale)
    if name == "mimformer":
        m = _load(os.path.join(root, "models", "MIMFormer_CPW.py"), "mimformer_cpw")
        net = m.MIMFormer(hsi_chans=bands, msi_chans=msi, scale_factor=scale)
        net.MIMFormer = Checkpointed(net.MIMFormer)
        return External(net, lambda n, yh, ym: n(yh, ym), scale)
    m = _load(os.path.join(root, "model.py"), "fusformer_model")
    net = m.MainNet()
    feat = net.refine[0].in_channels
    net.Embedding = nn.Sequential(nn.Linear(bands + msi, feat))
    net.refine = nn.Sequential(nn.Conv2d(feat, feat, 3, 1, 1), nn.LeakyReLU(), nn.Conv2d(feat, bands, 3, 1, 1))
    m.init_weights(net.Embedding, net.refine)

    def call(n, yh, ym):  # MainNet hard-codes x4 bicubic upsampling; same steps with our scale
        up = _up(yh, scale).clamp(0, 1)
        sz = up.shape[2]
        e = n.Embedding(torch.cat((up, ym), 1).flatten(2).transpose(1, 2))
        hp = n.T_D(n.T_E(e)).transpose(1, 2).reshape(up.shape[0], -1, sz, up.shape[3])
        return (n.refine(hp) + up).clamp(0, 1)
    return External(net, call, scale, tile=128, overlap=32)
