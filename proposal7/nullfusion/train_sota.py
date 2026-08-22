"""NullFusion SOTA training — CAVE x4, Nikon D700 SRF.

The pipeline reuses the repository's shared, protocol-audited library
(common/hsifusion) for SRF, degradation, metrics and dataset IO, and swaps in
proposal7.nullfusion.NullFusionNet as the method.  Both the LR-HSI and the
MSI are simulated from the GT with the Nikon D700 response (Wald's protocol),
exactly as the published CAVE x4 numbers are produced.

Run on Kaggle (GPU):   python train_sota.py
Local sanity (no data): python train_sota.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F

from hsifusion.degrade import gaussian_kernel2d
from hsifusion.io_utils import (available_splits, discover_dataset, find_pairs,
                                list_hsi, load_hsi)
from hsifusion.losses import sam_loss
from hsifusion.metrics import evaluate_arrays, ssim_torch
from hsifusion.srf import nikon_d700_srf

from proposal7.nullfusion import NullFusionConfig, NullFusionNet


class TrainConfig:
    scale = 4
    bands = 31
    msi_bands = 3
    ksize = 9
    sigma = 1.2
    cg_steps_train = 40
    cg_steps_eval = 200
    ridge_train = 1e-4
    ridge_eval = 1e-6
    width = 96
    enc_depth = 4
    prior_depth = 8
    use_attn = True
    cross_attn_heads = 4
    rank = 31
    iters = 250000
    time_budget_h = 9.0
    batch = 6
    patch = 80
    lr = 2e-4
    min_lr = 1e-6
    warmup = 1000
    grad_clip = 1.0
    amp = False
    grad_accum = 3
    ema_decay = 0.999
    w_l1 = 1.0
    w_ssim = 0.5
    w_sam = 0.05
    w_phys = 0.1
    val_every = 2000
    log_every = 200
    checkpoint_every = 3000
    seed = 42
    max_dim = 512
    val_patch = 200
    val_overlap = 32


def build_model(cfg: TrainConfig, device):
    mcfg = NullFusionConfig(scale=cfg.scale, bands=cfg.bands, msi_bands=cfg.msi_bands,
                            ksize=cfg.ksize, sigma=cfg.sigma,
                            cg_steps=cfg.cg_steps_train, ridge=cfg.ridge_train,
                            width=cfg.width, enc_depth=cfg.enc_depth,
                            prior_depth=cfg.prior_depth, use_attn=cfg.use_attn,
                            cross_attn_heads=cfg.cross_attn_heads,
                            rank=cfg.rank)
    net = NullFusionNet(mcfg).to(device)
    srf = nikon_d700_srf(cfg.bands)
    net.set_srf(torch.from_numpy(srf).float().to(device))
    return net


def simulate_obs(gt: torch.Tensor, op, srf_t: torch.Tensor, noise: float = 0.0):
    yH = op.D(gt)
    yM = torch.einsum("bchw,cm->bmhw", gt, srf_t)
    if noise > 0:
        yH = yH + noise * yH.std() * torch.randn_like(yH)
        yM = yM + noise * yM.std() * torch.randn_like(yM)
    return yH, yM


def _sam_deg(pred, target, eps=1e-8):
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    t = target.reshape(target.shape[0], target.shape[1], -1)
    cos = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1) + eps)
    return torch.acos(cos.clamp(-1, 1)).mean() * 180.0 / 3.14159265


def nullfusion_loss(out, gt, yH, yM, model, cfg):
    pred = out["out"]
    l_l1 = F.l1_loss(pred, gt)
    l_ssim = 1.0 - ssim_torch(pred, gt, 1.0)
    l_sam = _sam_deg(pred, gt).clamp(max=30.0)
    lp = (F.mse_loss(model.op.D(pred, model.op.D.default_kernel), yH)
          + F.mse_loss(model.op.R(pred), yM))
    total = (cfg.w_l1 * l_l1 + cfg.w_ssim * l_ssim
             + cfg.w_sam * l_sam + cfg.w_phys * lp)
    return total, l_l1, l_ssim, l_sam, lp


def _cosine_lr(it, total, warm, lr0, min_lr):
    if it < warm:
        return lr0 * it / max(warm, 1)
    return max(min_lr, 0.5 * lr0 * (1 + np.cos(np.pi * (it - warm) / max(total - warm, 1))))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

    def restore_from(self, model):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _chunked_forward(net, yH, yM, ps, ov, bands, scale):
    """Forward pass in overlapping HR patches, stitched with blending weights."""
    _, _, Hp, Wp = yM.shape
    out = torch.zeros(1, bands, Hp, Wp, device=yM.device)
    wmap = torch.zeros(1, 1, Hp, Wp, device=yM.device)
    for y0 in range(0, Hp, ps - ov):
        for x0 in range(0, Wp, ps - ov):
            y1 = min(y0 + ps, Hp)
            x1 = min(x0 + ps, Wp)
            y0r = max(0, y1 - ps)
            x0r = max(0, x1 - ps)
            lr_y0, lr_x0 = y0r // scale, x0r // scale
            lr_y1 = lr_y0 + (y1 - y0r) // scale
            lr_x1 = lr_x0 + (x1 - x0r) // scale
            yH_chunk = yH[:, :, lr_y0:lr_y1, lr_x0:lr_x1]
            yM_chunk = yM[:, :, y0r:y1, x0r:x1]
            with torch.no_grad():
                chunk_out = net(yH_chunk, yM_chunk)["out"]
            ph, pw = y1 - y0r, x1 - x0r
            chunk_out = chunk_out[:, :, :ph, :pw]
            blend = torch.ones(1, 1, ph, pw, device=yM.device)
            if ov > 0:
                edge = min(ov, ph // 2, pw // 2)
                ramp = torch.linspace(0, 1, edge, device=yM.device)
                if y0r > 0:
                    blend[:, :, :edge, :] *= ramp.view(1, 1, -1, 1)
                if y1 < Hp:
                    blend[:, :, -edge:, :] *= ramp.flip(0).view(1, 1, -1, 1)
                if x0r > 0:
                    blend[:, :, :, :edge] *= ramp.view(1, 1, 1, -1)
                if x1 < Wp:
                    blend[:, :, :, -edge:] *= ramp.flip(0).view(1, 1, 1, -1)
            out[:, :, y0r:y1, x0r:x1] += chunk_out * blend
            wmap[:, :, y0r:y1, x0r:x1] += blend
    return out / wmap.clamp_min(1e-8)


@torch.no_grad()
def validate(net, cfg, device, srf_t, scenes):
    net.eval()
    prev_ridge, prev_cg = net.cfg.ridge, net.cfg.cg_steps
    net.cfg.ridge = cfg.ridge_eval
    net.cfg.cg_steps = cfg.cg_steps_eval
    try:
        agg = {"psnr": [], "ssim": [], "sam": [], "ergas": []}
        for stem, hp in scenes:
            gt = torch.from_numpy(load_hsi(hp, cfg.bands, cfg.max_dim)).float()
            h = (gt.shape[1] // cfg.scale) * cfg.scale
            w = (gt.shape[2] // cfg.scale) * cfg.scale
            gt_c = gt[:, :h, :w]
            g = gt_c.unsqueeze(0).to(device)
            yH, yM = simulate_obs(g, net.op, srf_t)
            Hp, Wp = g.shape[2], g.shape[3]
            ps, ov = cfg.val_patch, cfg.val_overlap
            if Hp <= ps and Wp <= ps:
                with torch.no_grad():
                    out = net(yH, yM)["out"]
            else:
                out = _chunked_forward(net, yH, yM, ps, ov, cfg.bands, cfg.scale)
            m = evaluate_arrays(out[0].cpu().numpy().transpose(1, 2, 0),
                                gt_c.numpy().transpose(1, 2, 0), cfg.scale)
            for k in agg:
                agg[k].append(m[k])
    finally:
        net.cfg.ridge, net.cfg.cg_steps = prev_ridge, prev_cg
    return {k: float(np.mean(v)) for k, v in agg.items()}


def train(cfg: TrainConfig, device, root=None, smoke=False):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    srf = nikon_d700_srf(cfg.bands)
    srf_t = torch.from_numpy(srf).float().to(device)
    net = build_model(cfg, device)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"[model] NullFusionNet params = {nparams}")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=False)
    ema = EMA(net, cfg.ema_decay)
    total, warm = cfg.iters, cfg.warmup
    T0 = time.time()
    TIME_LIMIT = cfg.time_budget_h * 3600

    if smoke:
        scenes = None
        spec_train = [(f"s{i}", None) for i in range(4)]
    else:
        spec_root = discover_dataset(["CAVE"], required=True)
        splits = available_splits(spec_root)
        train_pairs = find_pairs(spec_root, splits.get("Train", "Train"))
        test_scenes = list_hsi(spec_root, splits.get("Test", "Test"))
        print(f"[data] CAVE train={len(train_pairs)} test={len(test_scenes)}")

    cache = {}

    def get_hsi(stem, hp):
        if hp is None:
            return torch.rand(1, cfg.bands, 64, 64)
        if stem not in cache:
            cache[stem] = load_hsi(hp, cfg.bands)
        return torch.from_numpy(cache[stem]).float()

    def sample_batch():
        lrs, msis, gts = [], [], []
        for _ in range(cfg.batch):
            if smoke:
                gt = torch.rand(1, cfg.bands, cfg.patch, cfg.patch)
            else:
                stem, hp, _ = random.choice(train_pairs)
                hsi = get_hsi(stem, hp)
                H, W = hsi.shape[1], hsi.shape[2]
                p = min(cfg.patch, H, W)
                Hp = (H // p) * p
                y = random.randrange(0, H - Hp + 1)
                x = random.randrange(0, W - Hp + 1)
                gt = hsi[:, y:y + p, x:x + p].unsqueeze(0)
                if random.random() < 0.5:
                    gt = gt.flip(3)
                if random.random() < 0.5:
                    gt = gt.flip(2)
            gt = gt.to(device)
            yH, yM = simulate_obs(gt, net.op, srf_t,
                                 noise=random.uniform(0, 0.02))
            lrs.append(yH); msis.append(yM); gts.append(gt)
        return (torch.cat(lrs), torch.cat(msis), torch.cat(gts))

    it0, best, best_state, history = 0, -1, None, []
    CP_FILE = os.path.join(os.getcwd(), "cp", "checkpoint.pt")
    os.makedirs(os.path.dirname(CP_FILE), exist_ok=True)

    if os.path.exists(CP_FILE):
        try:
            ck = torch.load(CP_FILE, map_location=device, weights_only=False)
            net.load_state_dict(ck["model"])
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            if "best_state" in ck and ck["best_state"] is not None:
                best_state = {k: v.to(device) for k, v in ck["best_state"].items()}
            best = float(ck.get("best_psnr", -1))
            history = ck.get("history", [])
            it0 = int(ck.get("it0", 0)) + 1
            print(f"[resume] {CP_FILE} @ iter {it0} best {best:.3f}")
        except Exception as e:
            print(f"[resume] failed: {e}")

    def _save_checkpoint(it):
        torch.save({"model": {k: v.detach().clone() for k, v in net.state_dict().items()},
                    "ema": ema.shadow, "opt": opt.state_dict(), "it0": it,
                    "best_psnr": best, "best_state": best_state, "history": history,
                    "nparams": nparams,
                    "cfg": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("__")}},
                   CP_FILE)
        print(f"[save] checkpoint @ {it} -> {CP_FILE}")

    val_scenes = test_scenes if not smoke else None

    try:
        for it in range(it0, total + 1):
            if (time.time() - T0) > TIME_LIMIT:
                print(f"[time budget reached at iter {it}]")
                break
            opt.param_groups[0]["lr"] = _cosine_lr(it, total, warm, cfg.lr, cfg.min_lr)
            net.train()
            lr, msi, gt = sample_batch()
            opt.zero_grad(set_to_none=True)
            out = net(lr, msi)
            loss, ll, ls, lsm, lp = nullfusion_loss(out, gt, lr, msi, net, cfg)
            (loss / cfg.grad_accum).backward()
            if (it + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(net)
            if it % cfg.log_every == 0:
                print(f"[{it}/{total}] loss={loss.item():.4f} l1={ll.item():.3f} "
                      f"ssim={ls.item():.3f} sam={lsm.item():.3f} phys={lp.item():.4f} "
                      f"({time.time()-T0:.0f}s)")
            if not smoke and ((it + 1) % cfg.val_every == 0 or it == total):
                ema.apply_to(net)
                vm = validate(net, cfg, device, srf_t, val_scenes)
                print(f"[val @{it}] PSNR={vm['psnr']:.3f} SSIM={vm['ssim']:.4f} "
                      f"SAM={vm['sam']:.3f} ERGAS={vm['ergas']:.3f}")
                history.append({"iter": it, **vm})
                if vm["psnr"] > best:
                    best = vm["psnr"]
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                ema.restore_from(net)
            if (it + 1) % cfg.checkpoint_every == 0:
                ema.apply_to(net)
                _save_checkpoint(it)
                ema.restore_from(net)
    except KeyboardInterrupt:
        print("interrupted")

    _save_checkpoint(it)

    if not smoke:
        if best_state is not None:
            net.load_state_dict(best_state)
        mean = validate(net, cfg, device, srf_t, test_scenes)
        print("=" * 60)
        print(f"FINAL CAVE TEST: PSNR={mean['psnr']:.3f} SSIM={mean['ssim']:.4f} "
              f"SAM={mean['sam']:.3f} ERGAS={mean['ergas']:.3f}")
        print("=" * 60)
        with open("sota_results.json", "w") as f:
            json.dump({"protocol": "CAVE x4, Nikon D700 SRF, Wald blur",
                       "nparams": nparams, "mean": mean, "history": history}, f, indent=2)
        print("saved sota_results.json")


def smoke():
    print("[smoke] synthetic training-path self-test")
    cfg = TrainConfig()
    cfg.bands = 8
    cfg.msi_bands = 3
    cfg.batch = 2
    cfg.patch = 32
    cfg.cg_steps_train = 20
    cfg.width = 32
    cfg.enc_depth = 2
    cfg.prior_depth = 4
    cfg.use_attn = False
    cfg.cross_attn_heads = 2
    device = torch.device("cpu")
    net = build_model(cfg, device)
    srf = nikon_d700_srf(cfg.bands)
    srf_t = torch.from_numpy(srf).float()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    losses = []
    for it in range(20):
        gt = torch.rand(cfg.batch, cfg.bands, cfg.patch, cfg.patch)
        yH, yM = simulate_obs(gt, net.op, srf_t)
        out = net(yH, yM)
        loss, *_ = nullfusion_loss(out, gt, yH, yM, net, cfg)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        yH2, yM2 = net.op(out["out"])
        e = max((yH2 - yH).abs().max().item() / max(yH.abs().max().item(), 1e-12),
                (yM2 - yM).abs().max().item() / max(yM.abs().max().item(), 1e-12))
        assert e < 5e-2, f"consistency broke during training: {e}"
    ok = losses[-1] < losses[0]
    print(f"[smoke] loss {losses[0]:.4f} -> {losses[-1]:.4f}  "
          f"consistency held  {'PASS' if ok else 'FAIL'}")
    assert ok, "training did not reduce loss"
    print("[smoke] ALL PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic self-test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    cfg = TrainConfig()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train(cfg, device)


if __name__ == "__main__":
    main()
