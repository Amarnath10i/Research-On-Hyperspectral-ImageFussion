"""Published fusion networks, built from their authors' code and retrained under our protocol.

The repositories carry no license, so they are not copied here. `fetch()` clones them at run time (Kaggle
kernels have internet) into $PUF_EXT (default /kaggle/working/ext), pinned to the commits below, and the
adapters import the network classes from there. Only the band counts and the scale factor are changed:

    psrt       PSRT (Deng et al., TGRS 2023)            Deng-shangqi/PSRT        model/model_SR.py PSRTnet
    dct        DCT / DCTransformer (Ma et al., InfFus 2024)  qingma2016/DCTransformer  dual.py DCT
    mimformer  MIMFormer (Li et al., JSTARS 2024)       meiruni/MIMFormer        models/MIMFormer_CPW.py
    fusformer  Fusformer (Hu et al., GRSL 2022)         J-FHu/Fusformer          model.py MainNet
    smgu       SMGU-Net (Yan et al., PR 2025)           yansql/SMGU-Net          model.py Net (x4 only)
    clsnet     CLSNet (Yang et al., InfFus 2026)        HengYang01/CLSNet        model/CLSNet.py CLSNet
    otpnet     OTPNet (AAAI 2025)                       tsharry2201/OTPNet       otpnet/model.py OTPNet
    u2net      U2Net (Peng et al., ACM MM 2023)         PSRben/U2Net             model/u2net.py U2Net
    mimosst    MIMO-SST (Fang et al., TGRS 2024)        Freelancefangjian/MIMO-SST  Model.py Net

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
    "dhif": ("https://github.com/TaoHuang95/DHIF-Net", "DHIF-Net"),
    "smgu": ("https://github.com/yansql/SMGU-Net", "SMGU-Net"),
    "clsnet": ("https://github.com/HengYang01/CLSNet", "CLSNet"),
    "otpnet": ("https://github.com/tsharry2201/OTPNet", "OTPNet"),
    "u2net": ("https://github.com/PSRben/U2Net", "U2Net"),
    "mimosst": ("https://github.com/Freelancefangjian/MIMO-SST", "MIMO-SST"),
}
# pipeline settings of the published networks (all with an L1 loss): their published learning rate, no weight
# decay, no clipping. PSRT's own code adds AdamW decay 1e-4 and clipping at 10, and MIMFormer's uses batch 16 and
# a step schedule; the off-* jobs of kaggle/jobs.py train both with their authors' recipes.
RECIPES = {
    "psrt": dict(lr=1e-4, wd=0.0, clip=0.0),
    "dct": dict(lr=1e-4, wd=0.0, clip=0.0),
    "mimformer": dict(lr=1e-4, wd=0.0, clip=0.0),
    "fusformer": dict(lr=1e-3, wd=0.0, clip=0.0),
    "dhif": dict(lr=3e-4, wd=0.0, clip=0.0),
    "smgu": dict(lr=1e-4, wd=0.0, clip=0.0),          # train.py: Adam 1e-4
    "clsnet": dict(lr=2e-4, wd=0.0, clip=0.0),        # train_*.py: Adam 2e-4 (decay 1e-8), cosine
    "otpnet": dict(lr=1e-4, wd=0.0, clip=0.0),        # train.py: AdamW 1e-4 (decay 1e-4), step
    "u2net": dict(lr=1e-3, wd=0.0, clip=0.0),         # train.py: Adam 1e-3, step
    "mimosst": dict(lr=1e-4, wd=0.0, clip=0.0),       # train.py: Adam 1e-4; three outputs + FFT loss there
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


def _load_src(path: str, modname: str, cut: str):
    """Like _load, but without the module's trailing demo code (everything from the line `cut` on)."""
    src = open(path, encoding="utf-8").read()
    if cut in src:
        src = src[:src.index(cut)]
    mod = type(sys)(modname)
    mod.__file__ = path
    sys.modules[modname] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
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
    if name == "dhif":
        # model-guided unfolding with learned operators: R, R^T as 3x3 convolutions, B, B^T as strided
        # (transposed) convolutions; the official code covers x8 and x16, x4 uses one stage of its x16 pair
        m = _load(os.path.join(root, "CAVE", "Model.py"), "dhif_model")
        net = m.HSI_Fusion(Ch=bands, stages=4, sf=scale)
        net.RT = nn.Sequential(nn.Conv2d(msi, bands, 3, 1, 1), nn.LeakyReLU())
        net.R = nn.Sequential(nn.Conv2d(bands, msi, 3, 1, 1), nn.LeakyReLU())
        net.conv = nn.Conv2d(bands + msi, 64, 3, 1, 1)
        if scale == 4:
            net.B = nn.Sequential(nn.Conv2d(bands, bands, 6, 4, 1), nn.LeakyReLU())
            net.BT = nn.Sequential(nn.ConvTranspose2d(bands, bands, 6, 4, 1), nn.LeakyReLU())
        net._initialize_weights()
        return External(net, lambda n, yh, ym: n(ym, yh), scale)
    if name == "mimformer":
        m = _load(os.path.join(root, "models", "MIMFormer_CPW.py"), "mimformer_cpw")
        net = m.MIMFormer(hsi_chans=bands, msi_chans=msi, scale_factor=scale)
        net.MIMFormer = Checkpointed(net.MIMFormer)
        return External(net, lambda n, yh, ym: n(yh, ym), scale)
    if name == "smgu":
        # spatial / spectral unfolding with learned up- and down-sampling; the stride-4 layers fix the scale
        if scale != 4:
            raise ValueError("SMGU-Net's code is written for x4 only")
        m = _load(os.path.join(root, "model.py"), "smgu_model")
        net = m.Net(niter=2, hsi_channels=bands, msi_channels=msi)
        return External(net, lambda n, yh, ym: n(yh, ym), scale)
    if name == "clsnet":
        m = _load_src(os.path.join(root, "model", "CLSNet.py"), "clsnet_model", "from thop import profile")
        net = m.CLSNet(in_channels=msi, out_channels=bands, n_feat=62, stage=3)
        net.upsamp = nn.Upsample(scale_factor=scale, mode="nearest")     # written for x8
        return External(net, lambda n, yh, ym: n(yh, ym), scale)
    if name == "otpnet":
        # written for pansharpening: the HR-MSI takes the place of the PAN image, the LR-HSI that of the LR-MS
        import importlib
        net = importlib.import_module("otpnet.model").OTPNet(pan_channels=msi, ms_channels=bands, hidden_channels=64,
                                                              num_stages=4, proximal_layers=3)
        return External(net, lambda n, yh, ym: n(ym, yh), scale)
    if name == "u2net":
        # written for pansharpening with 8 MS bands, 1 PAN band and x4: only these sizes are changed
        m = _load(os.path.join(root, "model", "u2net.py"), "u2net_model")
        net = m.U2Net(32)
        dim = net.raise_ms_dim[0].out_channels
        net.upsample = nn.Sequential(nn.Conv2d(bands, bands * scale * scale, 3, 1, 1, bias=False), nn.PixelShuffle(scale))
        net.raise_ms_dim = nn.Sequential(nn.Conv2d(bands, dim, 3, 1, 1), nn.LeakyReLU())
        net.raise_pan_dim = nn.Sequential(nn.Conv2d(msi, dim, 3, 1, 1), nn.LeakyReLU())
        net.to_hrms = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(), nn.Conv2d(dim, bands, 3, 1, 1))
        m.init_weights(net.upsample, net.raise_ms_dim, net.raise_pan_dim, net.to_hrms)
        return External(net, lambda n, yh, ym: n(yh, ym), scale)
    if name == "mimosst":
        # Model.py imports a PixelUnshuffle module that is not in the repository; torch has the same function
        shim = type(sys)("PixelUnshuffle")
        shim.pixel_unshuffle = F.pixel_unshuffle
        sys.modules["PixelUnshuffle"] = shim
        m = _load(os.path.join(root, "Model.py"), "mimosst_model")
        net = m.Net()                                                     # written for 31 + 3 bands at x8
        dim = net.Conv3_64.out_channels
        net.patch_embed = m.OverlapPatchEmbed(2 * dim + bands + msi, net.patch_embed.proj.out_channels)
        net.Conv3_64 = nn.Conv2d(msi, dim, 3, padding=1)
        net.Conv31_64 = nn.Conv2d(bands, dim, 3, padding=1)
        net.Conv96_31 = nn.Conv2d(net.Conv96_31.in_channels, bands, 3, padding=1)
        net.Conv192_31 = nn.Conv2d(net.Conv192_31.in_channels, bands, 3, padding=1)
        net.upSample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        net.re = nn.ReLU(inplace=True)
        return External(net, lambda n, yh, ym: n(ym, yh)[-1], scale)     # outputs at 1/4, 1/2 and full size
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
