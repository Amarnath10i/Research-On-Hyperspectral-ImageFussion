"""NullFusion v4 training for Chikusei (128 bands).

Adapted from train_nullfusion_v4.py for CAVE (31 bands).
Key changes:
  - bands=128, srf=chikusei_srf(128)
  - Larger dictionary atoms to handle 128-band spectral diversity
  - Reduced patch size for memory efficiency
  - Uses the ChikuseiDataset from common.hsifusion.datasets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

# Add common to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))

from hsifusion.srf import chikusei_srf, gaussian_srf, conditioning


# ===========================================================================
# 1. Operator (inline): A = [D; R], joint range/null decomposition
# ===========================================================================
def gaussian_kernel2d(size, sigma):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    k = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    return (k / k.sum()).astype(np.float32)


class DegradationOperator(nn.Module):
    """Per-band blur + decimate (forward) / upsample + blur (adjoint)."""

    def __init__(self, scale, ksize, sigma):
        super().__init__()
        self.scale = scale
        k = gaussian_kernel2d(ksize, sigma)
        self.register_buffer("kernel", torch.from_numpy(k).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel.repeat(C, 1, 1, 1)
        xb = F.conv2d(x, k, padding=self.kernel.shape[-1] // 2, groups=C)
        return xb[:, :, :: self.scale, :: self.scale]

    def transpose(self, y, out_hw):
        B, C, h, w = y.shape
        yu = F.interpolate(y, size=out_hw, mode="bicubic", align_corners=False)
        k = self.kernel.repeat(C, 1, 1, 1)
        return F.conv2d(yu, k, padding=self.kernel.shape[-1] // 2, groups=C)


def block_cg(applyA, rhs, steps, tol=1e-10):
    z = tuple(torch.zeros_like(r) for r in rhs)
    ap0 = applyA(z)
    r = tuple(rh - a for rh, a in zip(rhs, ap0))
    p = tuple(ri.clone() for ri in r)
    rs = sum((ri * ri).flatten(1).sum(1) for ri in r)
    shape = (rhs[0].shape[0],) + (1,) * (rhs[0].dim() - 1)
    for _ in range(steps):
        ap = applyA(p)
        denom = sum((pi * ai).flatten(1).sum(1) for pi, ai in zip(p, ap))
        alpha = (rs / denom.clamp_min(tol)).reshape(*shape)
        z = tuple(zi + alpha * pi for zi, pi in zip(z, p))
        r = tuple(ri - alpha * ai for ri, ai in zip(r, ap))
        rs_new = sum((ri * ri).flatten(1).sum(1) for ri in r)
        beta = (rs_new / rs.clamp_min(tol)).reshape(*shape)
        p = tuple(ri + beta * pi for ri, pi in zip(r, p))
        rs = rs_new
    return z


class CombinedOperator(nn.Module):
    def __init__(self, scale, bands, msi, ksize, sigma, srf, cg_steps, ridge):
        super().__init__()
        self.D = DegradationOperator(scale, ksize, sigma)
        self.scale = scale
        self.bands = bands
        self.msi_bands = msi
        self.cg_steps = cg_steps
        self.ridge = ridge
        self.register_buffer("srf", srf.float())

    def R(self, x):
        return torch.einsum("nbhw,bm->nmhw", x, self.srf)

    def Rt(self, m):
        return torch.einsum("nmhw,bm->nbhw", m, self.srf)

    def forward(self, x):
        return self.D(x), self.R(x)

    def adjoint(self, yH, yM, out_hw):
        return self.D.transpose(yH, out_hw) + self.Rt(yM)

    def apply_gram(self, zH, zM, out_hw):
        Dt = self.D.transpose(zH, out_hw)
        RtM = self.Rt(zM)
        wH = self.D(Dt) + self.D(RtM) + self.ridge * zH
        wM = self.R(Dt) + self.R(RtM) + self.ridge * zM
        return wH, wM

    def pinv(self, yH, yM, out_hw):
        zH, zM = block_cg(
            lambda p: self.apply_gram(p[0], p[1], out_hw), (yH, yM), self.cg_steps
        )
        return self.adjoint(zH, zM, out_hw)

    def project_null(self, v):
        out_hw = (v.shape[-2], v.shape[-1])
        yH, yM = self.forward(v)
        return v - self.pinv(yH, yM, out_hw)


# ===========================================================================
# 2. NullFusionNetV4 -- adapted for 128-band Chikusei
# ===========================================================================
class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class _CrossAttn(nn.Module):
    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_ch = ch // n_heads
        self.q = nn.Conv2d(ch, ch, 1)
        self.kv = nn.Conv2d(ch, ch * 2, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = max(self.head_ch, 1) ** -0.5

    def forward(self, query, context):
        B, C, Hq, Wq = query.shape
        if context.shape[-2:] != (Hq, Wq):
            context = F.adaptive_avg_pool2d(context, (Hq, Wq))
        q = self.q(query)
        kv = self.kv(context)
        k, v = kv.chunk(2, dim=1)
        nH, d = self.n_heads, self.head_ch
        q = q.reshape(B, nH, d, Hq * Wq).transpose(2, 3)
        k = k.reshape(B, nH, d, Hq * Wq).transpose(2, 3)
        v = v.reshape(B, nH, d, Hq * Wq).transpose(2, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(2, 3).reshape(B, C, Hq, Wq)
        return query + self.proj(out)


class _SpectralGate(nn.Module):
    """Per-pixel gating over K dictionary atoms."""
    def __init__(self, K, ctx_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ctx_ch + K, K, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, K, 1),
            nn.Sigmoid(),
        )

    def forward(self, alpha, ctx):
        x = torch.cat([alpha, ctx], dim=1)
        gate = self.net(x)
        return alpha * gate


def haar_dwt2d(x):
    """2D Haar DWT: x (B,C,H,W) -> (LL, LH, HL, HH) each (B,C,H/2,W/2)."""
    B, C, H, W = x.shape
    if H % 2 == 1:
        x = F.pad(x, (0, 0, 0, 1), mode='reflect')
    if W % 2 == 1:
        x = F.pad(x, (0, 1, 0, 0), mode='reflect')
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//2, 2, W//2, 2)
    LL = (x[:, :, :, 0, :, 0] + x[:, :, :, 1, :, 0] +
          x[:, :, :, 0, :, 1] + x[:, :, :, 1, :, 1]) / 4
    LH = (x[:, :, :, 0, :, 0] - x[:, :, :, 1, :, 0] +
          x[:, :, :, 0, :, 1] - x[:, :, :, 1, :, 1]) / 4
    HL = (x[:, :, :, 0, :, 0] + x[:, :, :, 1, :, 0] -
          x[:, :, :, 0, :, 1] - x[:, :, :, 1, :, 1]) / 4
    HH = (x[:, :, :, 0, :, 0] - x[:, :, :, 1, :, 0] -
          x[:, :, :, 0, :, 1] + x[:, :, :, 1, :, 1]) / 4
    return LL, LH, HL, HH


def haar_idwt2d(LL, LH, HL, HH):
    """Inverse 2D Haar DWT."""
    B, C, h, w = LL.shape
    H, W = h * 2, w * 2
    x = torch.zeros(B, C, H, W, device=LL.device, dtype=LL.dtype)
    x[:, :, 0::2, 0::2] = LL + LH + HL + HH
    x[:, :, 1::2, 0::2] = LL - LH + HL - HH
    x[:, :, 0::2, 1::2] = LL + LH - HL - HH
    x[:, :, 1::2, 1::2] = LL - LH - HL + HH
    return x


class _WaveletDetailBranch(nn.Module):
    """Wavelet high-frequency branch for null-space detail."""
    def __init__(self, bands, width, depth=2):
        super().__init__()
        self.bands = bands
        self.enc = nn.Sequential(
            nn.Conv2d(bands * 3, width, 3, 1, 1),
            nn.ReLU(inplace=True),
            *[_ResBlock(width) for _ in range(depth)],
            nn.Conv2d(width, bands * 3, 3, 1, 1)
        )

    def forward(self, null_comp):
        LL, LH, HL, HH = haar_dwt2d(null_comp)
        hf = torch.cat([LH, HL, HH], dim=1)
        hf_corr = self.enc(hf)
        LH_c, HL_c, HH_c = hf_corr.chunk(3, dim=1)
        LL_zero = torch.zeros_like(LL)
        corr = haar_idwt2d(LL_zero, LH_c, HL_c, HH_c)
        return corr[:, :, :null_comp.shape[-2], :null_comp.shape[-1]]


class NullFusionNetV4Chikusei(nn.Module):
    """NullFusion v4 adapted for Chikusei (128 bands).

    Key changes from CAVE version:
      - Larger dictionary atoms (global=96, mid=64, fine=48) for 128-band diversity
      - Reduced width (96->80) to fit in memory with 128 bands
      - Wavelet branch handles 128-band subbands
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        Bn, M, W = cfg.bands, cfg.msi_bands, cfg.width
        Kg, Km, Kf = cfg.dict_global, cfg.dict_mid, cfg.dict_fine
        self.op = CombinedOperator(cfg.scale, Bn, M, cfg.ksize, cfg.sigma,
                                   cfg.srf, cfg.cg_steps, cfg.ridge)
        # --- pyramid spectral dictionary ---
        self.D_global = nn.Parameter(torch.zeros(Bn, Kg))
        self.D_mid    = nn.Parameter(torch.zeros(Bn, Km))
        self.D_fine   = nn.Parameter(torch.zeros(Bn, Kf))
        self.Kg, self.Km, self.Kf = Kg, Km, Kf
        # --- conditioning encoder (cross-modal) ---
        self.hsi_stem = nn.Conv2d(Bn, W, 3, 1, 1)
        self.msi_stem = nn.Conv2d(M, W, 3, 1, 1)
        self.msi_detail = nn.Conv2d(M, W, 3, 1, 1)
        self.cross_attn = _CrossAttn(W, cfg.cross_attn_heads)
        self.fuse = nn.Conv2d(2 * W, W, 1)
        self.enc = nn.Sequential(*[_ResBlock(W) for _ in range(cfg.enc_depth)])
        self.up = nn.ConvTranspose2d(W, W, cfg.scale, cfg.scale, 0, bias=False)
        # --- code predictors for 3 scales ---
        self.prior_in = nn.Conv2d(2 * W + Bn, W, 3, 1, 1)
        body = []
        for i in range(cfg.prior_depth):
            body.append(_ResBlock(W))
        self.prior_body = nn.Sequential(*body)
        self.spect_gate_g = _SpectralGate(Kg, W)
        self.spect_gate_m = _SpectralGate(Km, W)
        self.spect_gate_f = _SpectralGate(Kf, W)
        self.code_head_g = nn.Conv2d(W, Kg, 3, 1, 1)
        self.code_head_m = nn.Conv2d(W, Km, 3, 1, 1)
        self.code_head_f = nn.Conv2d(W, Kf, 3, 1, 1)
        # --- wavelet high-frequency branch ---
        self.wavelet_branch = _WaveletDetailBranch(Bn, W // 2, depth=2)
        # --- spectral consistency projection ---
        self.consist_proj = nn.Conv2d(Kg + Km + Kf, Bn, 1)

    def set_srf(self, srf: torch.Tensor):
        s = srf if srf.shape[0] == self.cfg.bands else srf.t().contiguous()
        s = s.float().to(self.D_global.device)
        self.op.srf.data = s.to(self.op.srf.device)
        s_pinv = torch.linalg.pinv(s)
        self.register_buffer("srf_pinv_t", s_pinv.t().contiguous())
        with torch.no_grad():
            for D in (self.D_global, self.D_mid, self.D_fine):
                K = D.shape[1]
                M = torch.randn(self.cfg.bands, K, device=D.device)
                P = torch.linalg.pinv(s)
                Rng = s @ P
                D0 = M - Rng @ M
                D0 = D0 / (D0.norm(dim=0, keepdim=True) + 1e-8)
                D.copy_(D0)

    def _base(self, yH, yM):
        B, _, H, W = yM.shape
        base0 = torch.einsum("bmhw,cm->bchw", yM, self.srf_pinv_t)
        base1 = F.interpolate(yH, (H, W), mode="bicubic", align_corners=False)
        delta = base1 - base0
        base = base0 + delta - torch.einsum("bmhw,cm->bchw",
                                             self.op.R(delta), self.srf_pinv_t)
        return base

    def forward(self, yH, yM):
        base = self._base(yH, yM)
        f_hsi = self.hsi_stem(yH)
        msi_lr = F.adaptive_avg_pool2d(self.msi_stem(yM), f_hsi.shape[-2:])
        f_hsi = self.cross_attn(f_hsi, msi_lr)
        z = self.enc(self.fuse(torch.cat([f_hsi, msi_lr], dim=1)))
        f_hr = self.up(z)
        f_msi = self.msi_detail(yM)
        cond = torch.cat([f_hr, f_msi, base], dim=1)
        v = self.prior_in(cond)
        v = self.prior_body(v)
        # --- multi-scale codes with spectral gating ---
        alpha_g = F.softplus(self.code_head_g(v))
        alpha_m = F.softplus(self.code_head_m(v))
        alpha_f = F.softplus(self.code_head_f(v))
        alpha_g = self.spect_gate_g(alpha_g, v)
        alpha_m = self.spect_gate_m(alpha_m, v)
        alpha_f = self.spect_gate_f(alpha_f, v)
        # --- pyramid dictionary reconstruction ---
        null_g = torch.einsum("ck,bkhw->bchw", self.D_global, alpha_g)
        null_m = torch.einsum("ck,bkhw->bchw", self.D_mid, alpha_m)
        null_f = torch.einsum("ck,bkhw->bchw", self.D_fine, alpha_f)
        null_comp = null_g + null_m + null_f
        # exact MSI consistency
        null_comp = null_comp - self.op.Rt(self.op.R(null_comp))
        # --- wavelet high-frequency correction ---
        wf_corr = self.wavelet_branch(null_comp)
        out = base + null_comp + wf_corr
        # spectral consistency features (for loss)
        all_alpha = torch.cat([alpha_g, alpha_m, alpha_f], dim=1)
        consist = self.consist_proj(all_alpha)
        return {"out": out, "base": base, "null_comp": null_comp,
                "alpha_g": alpha_g, "alpha_m": alpha_m, "alpha_f": alpha_f,
                "wf_corr": wf_corr, "consist": consist}


# ===========================================================================
# 3. Metrics
# ===========================================================================
def calc_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12:
        return 100.0
    return -10.0 * np.log10(mse)


def calc_sam(pred, gt):
    p = pred.reshape(pred.shape[0], -1)
    g = gt.reshape(gt.shape[0], -1)
    p = p / (np.linalg.norm(p, axis=0, keepdims=True) + 1e-8)
    g = g / (np.linalg.norm(g, axis=0, keepdims=True) + 1e-8)
    cos = np.clip((p * g).sum(0), -1.0, 1.0)
    return np.mean(np.arccos(cos)) * 180.0 / np.pi


def calc_ergas(pred, gt, scale=4):
    C, H, W = pred.shape
    err = (pred - gt) ** 2
    ergas = 0.0
    for c in range(C):
        mg = gt[c].mean()
        if mg > 0:
            ergas += err[c].mean() / (mg ** 2)
    return math.sqrt(ergas / C) * 100.0 * scale


def calc_ssim(pred, gt):
    from scipy.ndimage import uniform_filter
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu1 = uniform_filter(pred, size=3, mode="reflect")
    mu2 = uniform_filter(gt, size=3, mode="reflect")
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    sigma12 = uniform_filter(pred * gt, size=3, mode="reflect") - mu1 * mu2
    sigma1 = uniform_filter(pred ** 2, size=3, mode="reflect") - mu1_sq
    sigma2 = uniform_filter(gt ** 2, size=3, mode="reflect") - mu2_sq
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2) + 1e-8)
    return np.mean(ssim)


# ===========================================================================
# 4. Config
# ===========================================================================
def build_config():
    class C:
        pass
    c = C()
    c.scale = 4
    c.bands = 128          # Chikusei has 128 bands
    c.msi_bands = 3
    c.ksize = 9
    c.sigma = 1.2
    c.cg_steps = 60        # More CG steps for harder 128->3 inverse problem
    c.ridge = 1e-6
    c.width = 80           # Reduced from 96 for memory with 128 bands
    c.enc_depth = 4
    c.prior_depth = 8
    c.cross_attn_heads = 4
    c.dict_global = 96     # Larger dictionaries for 128-band diversity
    c.dict_mid = 64
    c.dict_fine = 48
    c.epochs = 2000
    c.batch_size = 2       # Reduced for memory with 128 bands
    c.patch_size = 64      # Reduced from 80 for memory
    c.eval_every = 20
    c.lr = 2e-4
    c.weight_decay = 1e-4
    c.ema_decay = 0.999
    c.w_l1 = 1.0
    c.w_ssim = 0.5
    c.w_sam = 0.05
    c.w_phys = 0.1
    c.w_consist = 0.05
    c.time_budget_h = 9.0
    c.max_dim = 256        # Reduced for memory with 128 bands
    c.amp = True
    return c


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point and k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

    def restore_from(self, model):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}


# ===========================================================================
# 5. Training
# ===========================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    cfg = build_config()

    # Chikusei SRF
    srf_np = chikusei_srf(cfg.bands)
    srf_info = conditioning(srf_np)
    print(f"\nChikusei SRF conditioning: {srf_info['cond']:.2f}")
    print(f"Channel overlap: {srf_info['channel_overlap']:.4f}")
    srf_t = torch.from_numpy(srf_np).float().to(device)

    # Load dataset
    print("\nLoading Chikusei dataset...")
    from hsifusion.datasets import ChikuseiDataset

    train_ds = ChikuseiDataset(
        args.root, "train", cfg.bands, cfg.scale, cfg.patch_size, srf_mode="sensor"
    )
    test_ds = ChikuseiDataset(
        args.root, "test", cfg.bands, cfg.scale, cfg.patch_size, srf_mode="sensor"
    )
    print(f"Train patches: {len(train_ds)}, Test patches: {len(test_ds)}")

    # Build model
    cfg.srf = srf_t
    model = NullFusionNetV4Chikusei(cfg).to(device)
    model.set_srf(srf_t)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\nNullFusionNetV4-Chikusei params: {nparams/1e6:.2f}M")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    l1 = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_psnr = 0.0
    best_epoch = 0
    save_dir = os.path.join(args.output_dir, "nullfusion_chikusei_ckpt")
    os.makedirs(save_dir, exist_ok=True)
    T0 = time.time()
    LIMIT = cfg.time_budget_h * 3600

    steps_per_epoch = 200

    print(f"\nStarting training for {cfg.epochs} epochs (budget {cfg.time_budget_h}h)...")
    print(f"Batch {cfg.batch_size}, Patch {cfg.patch_size}, Eval every {cfg.eval_every}")
    print("-" * 70)

    for epoch in range(1, cfg.epochs + 1):
        if (time.time() - T0) > LIMIT:
            print(f"[time budget reached at epoch {epoch}]")
            break
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            gts, lhs, mss = [], [], []
            for _ in range(cfg.batch_size):
                g, l, m = train_ds[random.randrange(len(train_ds))]
                gts.append(g)
                lhs.append(l)
                mss.append(m)
            gt = torch.stack(gts, 0).to(device)
            yH = torch.stack(lhs, 0).to(device)
            yM = torch.stack(mss, 0).to(device)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                out_dict = model(yH, yM)
                out = out_dict["out"]
                loss_l1 = l1(out, gt)
                loss_ssim = 1.0 - torch.tensor(
                    calc_ssim(out[0].detach().cpu().numpy(), gt[0].detach().cpu().numpy()),
                    dtype=torch.float32, device=device)
                loss_sam = torch.tensor(
                    calc_sam(out[0].detach().cpu().numpy(), gt[0].detach().cpu().numpy()),
                    dtype=torch.float32, device=device)
                loss_phys = F.mse_loss(model.op.D(out), yH)
                consist = out_dict["consist"]
                loss_consist = F.l1_loss(consist, gt)
                loss = (cfg.w_l1 * loss_l1 + cfg.w_ssim * loss_ssim
                        + cfg.w_sam * loss_sam + cfg.w_phys * loss_phys
                        + cfg.w_consist * loss_consist)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / steps_per_epoch
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg.epochs} | Loss {avg:.5f} | "
                  f"LR {scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            ema.apply_to(model)
            psnrs, ssims, sams, ergas_list = [], [], [], []
            with torch.no_grad():
                for i in range(min(len(test_ds), 20)):  # eval on subset
                    g, l, m = test_ds[i]
                    g = g.unsqueeze(0).to(device)
                    l = l.unsqueeze(0).to(device)
                    m = m.unsqueeze(0).to(device)
                    pred = model(l, m)["out"][0].detach().cpu().numpy()
                    gt_np = g[0].detach().cpu().numpy()
                    psnrs.append(calc_psnr(pred, gt_np))
                    ssims.append(calc_ssim(pred, gt_np))
                    sams.append(calc_sam(pred, gt_np))
                    ergas_list.append(calc_ergas(pred, gt_np, cfg.scale))
            m_psnr = float(np.mean(psnrs))
            m_ssim = float(np.mean(ssims))
            m_sam = float(np.mean(sams))
            m_ergas = float(np.mean(ergas_list))
            improved = ""
            if m_psnr > best_psnr:
                best_psnr = m_psnr
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                improved = " [BEST]"
            print(f"  >>> Test@{epoch}: PSNR {m_psnr:.4f} | SSIM {m_ssim:.4f} | "
                  f"SAM {m_sam:.3f} | ERGAS {m_ergas:.3f}{improved}")
            ema.restore_from(model)

    # Final eval
    ckpt = os.path.join(save_dir, "best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    psnrs, ssims, sams, ergas_list = [], [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            g, l, m = test_ds[i]
            g = g.unsqueeze(0).to(device)
            l = l.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)
            pred = model(l, m)["out"][0].detach().cpu().numpy()
            gt_np = g[0].detach().cpu().numpy()
            psnrs.append(calc_psnr(pred, gt_np))
            ssims.append(calc_ssim(pred, gt_np))
            sams.append(calc_sam(pred, gt_np))
            ergas_list.append(calc_ergas(pred, gt_np, cfg.scale))
    final = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
             "sam": float(np.mean(sams)), "ergas": float(np.mean(ergas_list))}
    print("=" * 60)
    print(f"FINAL BEST (epoch {best_epoch}): PSNR {final['psnr']:.4f} | "
          f"SSIM {final['ssim']:.4f} | SAM {final['sam']:.3f} | "
          f"ERGAS {final['ergas']:.3f}")
    print("=" * 60)

    # Compare with SOTA
    print("\n--- SOTA Comparison (Chikusei x4) ---")
    sota = {
        "CoFusion (2026)": {"psnr": 49.14, "sam": 2.60},
        "RAMoE (2026)": {"psnr": 48.10, "sam": 0.79},
        "SMGU-Net (2025)": {"psnr": 48.82, "sam": 2.72},
        "PSRT (2023)": {"psnr": 47.99, "sam": 2.84},
        "KrylovNet (ours, 2.3k)": {"psnr": 43.69, "sam": 6.07},
    }
    for name, vals in sota.items():
        delta = final["psnr"] - vals["psnr"]
        print(f"  {name}: PSNR {vals['psnr']:.2f} (Δ={delta:+.2f}) SAM {vals['sam']:.2f}")

    with open(os.path.join(args.output_dir, "nullfusion_chikusei_results.json"), "w") as f:
        json.dump({
            "dataset": "Chikusei",
            "bands": cfg.bands,
            "protocol": "Chikusei x4, Sensor SRF, Wald blur",
            "params_M": nparams / 1e6,
            "best_epoch": best_epoch,
            "final": final,
            "sota_comparison": sota,
        }, f, indent=2)
    print("\nSaved nullfusion_chikusei_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input/chikusei",
                    help="Path to Chikusei dataset directory containing .mat file")
    ap.add_argument("--output_dir", default="/kaggle/working")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
