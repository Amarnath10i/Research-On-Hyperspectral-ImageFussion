"""Train / evaluate on Chikusei x4 (TIP'26 protocol: WV2 8-band MSI, Gaussian 7x7 sigma=2).

    python train.py --mat /kaggle/input/chikusei --model puformer --hours 10.5 --out /kaggle/working/puformer
    python train.py --mat ... --model ssrnet --iters 20000 --out .../ssrnet
    # fine-tune from earlier weights, 1200 epochs, SAM + SSIM loss, all GPUs:
    python train.py --mat ... --init_ckpt best_ema.pt --epochs 1200 --lr 1.5e-4 --w_sam 0.05 --w_ssim 0.1 --out ...
    # continue an interrupted run in a new session (same --epochs / --iters):
    python train.py --mat ... --resume /kaggle/input/<prev-output>/last.pt ... --out ...

Patches are random, so an "epoch" is one pass-equivalent over the training area:
(1632/64)*(2048/64) = 800 patches, i.e. 800/bs iterations. Every --eval_epochs the EMA
model is scored on the 64 validation patches (model selection) and on the 8 test tiles
(logged only; never used for selection), and last.pt is saved; an EMA snapshot is kept
every --snap_epochs. The final test uses the best-validation EMA weights, with and without
a 2-fold self-ensemble (identity + transpose; flips would shift the x4 sampling grid).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from hsifuse.data import PatchSampler, find_mat, load_chikusei, make_pairs, split
from hsifuse.losses import sam_loss, ssim_loss
from hsifuse.metrics import evaluate
from hsifuse.models import build
from hsifuse.ops import Degradation, wv2_srf

TRAIN_PATCHES = (1632 // 64) * (2048 // 64)  # 64x64 patches in the training area


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True, help=".mat file or a folder containing it")
    p.add_argument("--cache", default="", help="optional .npy cache of the normalised crop")
    p.add_argument("--model", default="puformer", choices=["puformer", "ssrnet"])
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--bs", type=int, default=8, help="total batch size (split across GPUs)")
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=300000)
    p.add_argument("--epochs", type=int, default=0, help="if > 0, overrides --iters (epoch = 800/bs iters)")
    p.add_argument("--eval_epochs", type=int, default=0, help="if > 0, overrides --eval_every")
    p.add_argument("--snap_epochs", type=int, default=0, help="keep an EMA snapshot every N epochs (0 = off)")
    p.add_argument("--hours", type=float, default=0.0, help="wall-clock budget; 0 = unlimited")
    p.add_argument("--sched", default="auto", choices=["auto", "time", "iter"],
                   help="cosine progress: time = whichever of iters / time budget runs out first; "
                        "iter = iterations only (multi-session runs); auto = time unless resuming")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--w_sam", type=float, default=0.0, help="weight of the SAM loss (radians)")
    p.add_argument("--w_ssim", type=float, default=0.0, help="weight of the (1 - SSIM) loss")
    p.add_argument("--init_ckpt", default="", help="warm start: model weights (e.g. an earlier best_ema.pt)")
    p.add_argument("--resume", default="", help="last.pt from an earlier session (default: <out>/last.pt)")
    p.add_argument("--gpus", type=int, default=0, help="GPUs for training (0 = all visible)")
    p.add_argument("--test_log", type=int, default=1, help="also score the test set at each eval (logged only)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny synthetic run for CI / CPU")
    return p.parse_args()


@torch.no_grad()
def predict(model, lr, ms, bs=1):
    return torch.cat([model(lr[i:i + bs], ms[i:i + bs]) for i in range(0, lr.shape[0], bs)])


@torch.no_grad()
def predict_tta(model, lr, ms, bs=1):
    """Identity + transpose. Transposing keeps the x4 decimation phase, so D stays exact."""
    t = lambda z: z.transpose(-1, -2)
    return 0.5 * (predict(model, lr, ms, bs) + t(predict(model, t(lr), t(ms), bs)))


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed), np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    deg = Degradation(wv2_srf()).to(dev)

    if a.smoke:
        cube = np.random.rand(128, 2048, 128).astype(np.float32) * 0.2
        tr, te, va = cube[:, 416:, :], np.stack([cube[:, :64, :64]] * 2), np.stack([cube[:, 272:336, :64]] * 2)
    else:
        path = a.mat if a.mat.endswith(".mat") else find_mat(a.mat)
        cube = load_chikusei(path, a.cache or None)
        tr, te, va = split(np.asarray(cube))
    print(f"train {tr.shape} test {te.shape} val {va.shape}  ({time.time()-t_start:.0f}s)", flush=True)

    epoch_iters = max(1, round(TRAIN_PATCHES / a.bs))
    if a.epochs:
        a.iters = a.epochs * epoch_iters
    if a.eval_epochs:
        a.eval_every = a.eval_epochs * epoch_iters
    snap_every = a.snap_epochs * epoch_iters
    print(f"epoch = {epoch_iters} iters (bs {a.bs}); total {a.iters} iters = {a.iters / epoch_iters:.0f} epochs; "
          f"eval every {a.eval_every} iters", flush=True)

    test = make_pairs(torch.from_numpy(te).to(dev), deg)
    val = make_pairs(torch.from_numpy(va).to(dev), deg)
    sampler = PatchSampler(tr, deg, a.patch, dev)
    del cube

    model = build(a.model, deg, **(dict(width=a.width, stages=a.stages) if a.model == "puformer" else {})).to(dev)
    if a.model == "puformer":
        model.amp = bool(a.amp)
    if a.init_ckpt:
        model.load_state_dict(torch.load(a.init_ckpt, map_location=dev))
        print(f"warm start from {a.init_ckpt}", flush=True)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    n_par = sum(q.numel() for q in model.parameters()) / 1e6
    n_gpu = torch.cuda.device_count() if dev == "cuda" else 0
    n_gpu = min(n_gpu, a.gpus) if a.gpus else n_gpu
    net = torch.nn.DataParallel(model, device_ids=list(range(n_gpu))) if n_gpu > 1 else model
    print(f"{a.model}: {n_par:.2f}M params on {dev} x{max(n_gpu, 1)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp) and dev == "cuda")

    ck = a.resume or os.path.join(a.out, "last.pt")
    it, best, hist = 0, -1.0, []
    resumed = os.path.exists(ck)
    if resumed:
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]), ema.load_state_dict(s["ema"]), opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"]); it, best, hist = s["it"], s["best"], s["hist"]
        src, dst = os.path.join(os.path.dirname(ck), "best_ema.pt"), os.path.join(a.out, "best_ema.pt")
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            torch.save(torch.load(src, map_location=dev), dst)
        print(f"resumed from {ck} at it={it} (epoch {it / epoch_iters:.1f}) best_val={best:.3f}", flush=True)
    time_sched = a.sched == "time" or (a.sched == "auto" and not resumed)

    def sched(i):  # warmup, then cosine over iterations (or over the time budget, if that ends first)
        if i < a.warmup:
            return (i + 1) / a.warmup
        prog = i / a.iters
        if a.hours and time_sched:
            prog = max(prog, (time.time() - t_start) / (a.hours * 3600 * 0.97))
        return max(0.5 * (1 + math.cos(math.pi * min(prog, 1.0))), 1e-3)

    # bicubic reference on test (sanity check of the pipeline)
    up = F.interpolate(test[0], scale_factor=4, mode="bicubic", align_corners=False)
    bic = evaluate(test[2].cpu(), up.cpu())
    print("bicubic test:", {k: round(v, 4) for k, v in bic.items()}, flush=True)

    budget = a.hours * 3600
    it0, t_train0 = it, time.time()
    model.train()
    while it < a.iters:
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(it)
        lr_, ms_, gt_ = sampler(a.bs)
        outs = net(lr_, ms_, return_all=True)
        loss = F.l1_loss(outs[-1], gt_) + 0.1 * sum(F.l1_loss(o, gt_) for o in outs[:-1])
        if a.w_sam:
            loss = loss + a.w_sam * sam_loss(outs[-1], gt_)
        if a.w_ssim:
            loss = loss + a.w_ssim * ssim_loss(outs[-1], gt_)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        it += 1

        out_of_time = budget and (time.time() - t_start) > budget
        if it % a.eval_every == 0 or it == a.iters or out_of_time:
            v = evaluate(val[2].cpu(), predict(ema, val[0], val[1], 16).cpu())
            rec = dict(it=it, epoch=round(it / epoch_iters, 1), loss=loss.item(), lr=opt.param_groups[0]["lr"],
                       **{f"val_{k}": x for k, x in v.items()})
            msg = (f"ep {it / epoch_iters:7.1f} it {it:6d} loss {loss.item():.5f} lr {opt.param_groups[0]['lr']:.2e} "
                   f"val PSNR {v['PSNR']:.3f} SAM {v['SAM']:.3f}")
            if a.test_log:
                tm = evaluate(test[2].cpu(), predict(ema, test[0], test[1]).cpu())
                rec.update({f"test_{k}": x for k, x in tm.items()})
                msg += (f" | test PSNR {tm['PSNR']:.3f} SSIM {tm['SSIM']:.4f}/{tm['SSIM_psrt']:.4f} "
                        f"SAM {tm['SAM']:.4f} ERGAS {tm['ERGAS']:.4f} RMSE {tm['RMSE_DN']:.2f}")
            hist.append(rec)
            if v["PSNR"] > best:
                best = v["PSNR"]
                torch.save(ema.state_dict(), os.path.join(a.out, "best_ema.pt"))
            if snap_every and it % snap_every == 0:
                torch.save(ema.state_dict(), os.path.join(a.out, f"ema_ep{it // epoch_iters:05d}.pt"))
            rate = (it - it0) / max(time.time() - t_train0, 1e-6)
            print(msg + f" | best val {best:.3f} ({(time.time()-t_start)/3600:.2f} h, {rate:.2f} it/s)", flush=True)
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                            scaler=scaler.state_dict(), it=it, best=best, hist=hist, args=vars(a)),
                       os.path.join(a.out, "last.pt"))
            json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
        if out_of_time:
            print(f"time budget reached at epoch {it / epoch_iters:.1f}; continue with --resume <out>/last.pt", flush=True)
            break

    ema.load_state_dict(torch.load(os.path.join(a.out, "best_ema.pt"), map_location=dev))
    t1 = time.time()
    pred = predict(ema, test[0], test[1])
    infer_s = (time.time() - t1) / test[0].shape[0]
    res = evaluate(test[2].cpu(), pred.cpu())
    pred_tta = predict_tta(ema, test[0], test[1])
    res_tta = evaluate(test[2].cpu(), pred_tta.cpu())
    per_img = [evaluate(test[2][i:i + 1].cpu(), pred[i:i + 1].cpu()) for i in range(pred.shape[0])]
    per_img_tta = [evaluate(test[2][i:i + 1].cpu(), pred_tta[i:i + 1].cpu()) for i in range(pred.shape[0])]
    summary = dict(model=a.model, params_M=n_par, iters=it, epochs=round(it / epoch_iters, 1), epoch_iters=epoch_iters,
                   best_val_PSNR=best, test=res, test_tta=res_tta, test_per_image=per_img,
                   test_per_image_tta=per_img_tta, bicubic_test=bic, infer_s_per_256=infer_s, args=vars(a))
    json.dump(summary, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    np.save(os.path.join(a.out, "test_pred.npy"), pred.cpu().numpy().astype(np.float16))
    np.save(os.path.join(a.out, "test_pred_tta.npy"), pred_tta.cpu().numpy().astype(np.float16))
    print("TEST:", {k: round(v, 4) for k, v in res.items()}, flush=True)
    print("TEST (self-ensemble):", {k: round(v, 4) for k, v in res_tta.items()}, flush=True)


if __name__ == "__main__":
    main()
