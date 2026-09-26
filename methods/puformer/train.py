"""Train / evaluate on Chikusei x4 (TIP'26 protocol: WV2 8-band MSI, Gaussian 7x7 sigma=2).

    python train.py --mat /kaggle/input/chikusei --model puformer --hours 10.5 --out /kaggle/working/puformer
    python train.py --mat ... --model ssrnet --iters 20000 --out .../ssrnet
    # fine-tune from earlier weights on 2 GPUs (DDP, --bs is per GPU), SAM + SSIM loss:
    torchrun --standalone --nproc_per_node 2 train.py --mat ... --init_ckpt best_ema.pt --lr 1.5e-4 \
        --w_sam 0.05 --w_ssim 0.1 --deadline <unix time> --out ...
    # continue an interrupted run in a new session (same --epochs / --iters):
    python train.py --mat ... --resume /kaggle/input/<prev-output>/last.pt ... --out ...

Patches are random, so an "epoch" is one pass-equivalent over the training area:
(1632/64)*(2048/64) = 800 patches, i.e. 800/(bs * GPUs) iterations. Every --eval_every
iterations the EMA model is scored on the 64 validation patches (model selection) and on the
8 test tiles (logged only; never used for selection), and last.pt is saved. The final test uses
the best-validation EMA weights, with and without a 2-fold self-ensemble (identity + transpose;
flips would shift the x4 sampling grid).

Time budget: --hours (from the start of this script) or --deadline (absolute unix time). With a
time budget the cosine schedule runs over the number of iterations that fit before the deadline,
re-estimated from the measured speed; the run stops cleanly at the deadline in any case.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from hsifuse.data import PatchSampler, find_mat, load_chikusei, make_pairs, split
from hsifuse.evaluation import final_test, predict
from hsifuse.losses import sam_loss, ssim_loss
from hsifuse.metrics import evaluate
from hsifuse.models import build
from hsifuse.ops import Degradation, wv2_srf

TRAIN_PATCHES = (1632 // 64) * (2048 // 64)  # 64x64 patches in the training area
SYNC_EVERY = 50                              # iterations between stop / schedule broadcasts (DDP)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True, help=".mat file or a folder containing it")
    p.add_argument("--cache", default="", help="optional .npy cache of the normalised crop")
    p.add_argument("--model", default="puformer", choices=["puformer", "ssrnet"])
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--pad", default="reflect", choices=["reflect", "zeros"],
                   help="border mode of D / Dt inside the network; reflect = the simulation operator "
                        "(zeros = the first run's setting)")
    p.add_argument("--bs", type=int, default=8, help="batch size per process (GPU)")
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=300000)
    p.add_argument("--epochs", type=int, default=0, help="if > 0, overrides --iters (epoch = 800/(bs*GPUs) iters)")
    p.add_argument("--eval_epochs", type=int, default=0, help="if > 0, overrides --eval_every")
    p.add_argument("--snap_epochs", type=int, default=0, help="keep an EMA snapshot every N epochs (0 = off)")
    p.add_argument("--hours", type=float, default=0.0, help="wall-clock budget from script start; 0 = unlimited")
    p.add_argument("--deadline", type=float, default=0.0, help="absolute unix time to stop at (overrides --hours)")
    p.add_argument("--sched", default="auto", choices=["auto", "time", "iter"],
                   help="cosine length: time = iterations that fit before the deadline (re-estimated); "
                        "iter = --iters only (multi-session runs); auto = time unless resuming")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--eval_init", type=int, default=1, help="score the starting weights before training")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--w_sam", type=float, default=0.0, help="weight of the SAM loss (radians)")
    p.add_argument("--w_ssim", type=float, default=0.0, help="weight of the (1 - SSIM) loss")
    p.add_argument("--init_ckpt", default="", help="warm start: model weights (e.g. an earlier best_ema.pt)")
    p.add_argument("--resume", default="", help="last.pt from an earlier session (default: <out>/last.pt)")
    p.add_argument("--test_log", type=int, default=1, help="also score the test set at each eval (logged only)")
    p.add_argument("--q2n", type=int, default=1, help="include Q2n in the final test metrics")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe", type=int, default=0,
                   help="speed test: time N iterations (after 30 warm-up ones), print PROBE samples/s, exit")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic run for CI / CPU")
    return p.parse_args()


def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, 0
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            timeout=datetime.timedelta(minutes=30))
    return rank, world, local


def host_mem():
    """Current RSS of this process and the machine's available memory (GB), Linux only."""
    out = []
    try:
        with open("/proc/self/status") as f:
            rss = next(int(l.split()[1]) for l in f if l.startswith("VmRSS:"))
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:"))
        out.append(f"RSS {rss / 2**20:.2f} GB, host free {avail / 2**20:.1f} GB")
    except (OSError, StopIteration):
        pass
    return ", ".join(out)


def main():
    a = get_args()
    rank, world, local = setup_dist()
    main_proc = rank == 0
    log = (lambda *s: print(*s, flush=True)) if main_proc else (lambda *s: None)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed), np.random.seed(a.seed)
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    t_start = time.time()
    deadline = a.deadline or (t_start + a.hours * 3600 if a.hours else math.inf)
    deg = Degradation(wv2_srf(), pad=a.pad).to(dev)

    if a.smoke:
        cube = np.random.rand(128, 2048, 128).astype(np.float32) * 0.2
        tr, te, va = cube[:, 416:, :], np.stack([cube[:, :64, :64]] * 2), np.stack([cube[:, 272:336, :64]] * 2)
    else:
        if world > 1 and not main_proc:
            dist.barrier()                   # rank 0 builds the cache first
        path = a.mat if a.mat.endswith(".mat") else find_mat(a.mat)
        cube = load_chikusei(path, a.cache or None)
        if world > 1 and main_proc:
            dist.barrier()
        tr, te, va = split(np.asarray(cube))
    log(f"train {tr.shape} test {te.shape} val {va.shape}  ({time.time() - t_start:.0f}s)")

    epoch_iters = max(1, round(TRAIN_PATCHES / (a.bs * world)))
    if a.epochs:
        a.iters = a.epochs * epoch_iters
    if a.eval_epochs:
        a.eval_every = a.eval_epochs * epoch_iters
    snap_every = a.snap_epochs * epoch_iters

    test = make_pairs(torch.from_numpy(te).to(dev), deg) if main_proc else None
    val = make_pairs(torch.from_numpy(va).to(dev), deg) if main_proc else None
    torch.manual_seed(a.seed + 1000 * rank)  # different patches on every GPU
    sampler = PatchSampler(tr, deg, a.patch, dev)
    del cube, tr, te, va

    model = build(a.model, deg, **(dict(width=a.width, stages=a.stages) if a.model == "puformer" else {})).to(dev)
    if a.model == "puformer":
        model.amp = bool(a.amp)
    if a.init_ckpt:
        model.load_state_dict(torch.load(a.init_ckpt, map_location=dev))
        log(f"warm start from {a.init_ckpt}")
    n_par = sum(q.numel() for q in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp) and dev.type == "cuda")

    ck = a.resume or os.path.join(a.out, "last.pt")
    it, best, hist = 0, -1.0, []
    resumed = os.path.exists(ck)
    ema_state = None
    if resumed:
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]), opt.load_state_dict(s["opt"]), scaler.load_state_dict(s["scaler"])
        it, best, hist, ema_state = s["it"], s["best"], s["hist"], s["ema"]
        src, dst = os.path.join(os.path.dirname(ck), "best_ema.pt"), os.path.join(a.out, "best_ema.pt")
        if main_proc and os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            torch.save(torch.load(src, map_location=dev), dst)
        log(f"resumed from {ck} at it={it} (epoch {it / epoch_iters:.1f}) best_val={best:.3f}")
    ema = None
    if main_proc:
        ema = copy.deepcopy(model).eval()
        if ema_state is not None:
            ema.load_state_dict(ema_state)
        for q in ema.parameters():
            q.requires_grad_(False)
    net = model
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if dev.type == "cuda" else None, broadcast_buffers=False,
            gradient_as_bucket_view=True)
    time_sched = a.sched == "time" or (a.sched == "auto" and not resumed)
    log(f"{a.model} ({a.pad} borders): {n_par:.2f}M params on {world} x {dev.type}; batch {a.bs} x {world}; "
        f"epoch = {epoch_iters} iters; eval every {a.eval_every} iters; "
        f"deadline in {(deadline - time.time()) / 3600:.2f} h; schedule by {'time' if time_sched else 'iters'}")

    def lr_at(i, total):  # warmup, then cosine over `total` iterations, floor 1e-3 * lr
        if i < a.warmup:
            return a.lr * (i + 1) / a.warmup
        return a.lr * max(0.5 * (1 + math.cos(math.pi * min(i / max(total, 1), 1.0))), 1e-3)

    def evaluate_ema(i, loss_avg, lr_now):
        nonlocal best
        v = evaluate(val[2], predict(ema, val[0], val[1], 16))
        rec = dict(it=i, epoch=round(i / epoch_iters, 1), loss=loss_avg, lr=lr_now,
                   hours=round((time.time() - t_start) / 3600, 3), **{f"val_{k}": x for k, x in v.items()})
        msg = (f"ep {i / epoch_iters:7.1f} it {i:6d} loss {loss_avg:.5f} lr {lr_now:.2e} "
               f"val PSNR {v['PSNR']:.3f} SSIM {v['SSIM']:.4f} SAM {v['SAM']:.3f}")
        if a.test_log:
            tm = evaluate(test[2], predict(ema, test[0], test[1]))
            rec.update({f"test_{k}": x for k, x in tm.items()})
            msg += (f" | test PSNR {tm['PSNR']:.3f} SSIM {tm['SSIM']:.4f}/{tm['SSIM_psrt']:.4f} "
                    f"SAM {tm['SAM']:.4f} ERGAS {tm['ERGAS']:.4f} RMSE {tm['RMSE_DN']:.2f} SCC {tm['SCC']:.4f}")
        hist.append(rec)
        if v["PSNR"] > best:
            best = v["PSNR"]
            torch.save(ema.state_dict(), os.path.join(a.out, "best_ema.pt"))
        if snap_every and i and i % snap_every == 0:
            torch.save(ema.state_dict(), os.path.join(a.out, f"ema_ep{i // epoch_iters:05d}.pt"))
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                        scaler=scaler.state_dict(), it=i, best=best, hist=hist, args=vars(a)),
                   os.path.join(a.out, "last.pt"))
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)
        return msg + f" | best val {best:.3f} ({(time.time() - t_start) / 3600:.2f} h)"

    bic = None
    if main_proc and not a.probe:
        up = F.interpolate(test[0], scale_factor=4, mode="bicubic", align_corners=False)
        bic = evaluate(test[2], up)
        log("bicubic test:", {k: round(v, 4) for k, v in bic.items()})
        if a.eval_init and (a.init_ckpt or resumed) and not any(r["it"] == it for r in hist):
            log("start: " + evaluate_ema(it, float("nan"), 0.0))

    total = a.iters
    it0, t_train0 = it, time.time()
    stop = False
    loss_acc, n_acc, t_log, it_log = torch.zeros((), device=dev), 0, time.time(), it
    model.train()
    while it < a.iters and not stop:
        lr_now = lr_at(it, total)
        for g in opt.param_groups:
            g["lr"] = lr_now
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
        scaler.step(opt)
        scaler.update()
        it += 1
        if main_proc:
            with torch.no_grad():
                d = min(a.ema, (1 + it) / (10 + it))
                pe, pm = list(ema.parameters()), [q.detach() for q in model.parameters()]
                torch._foreach_mul_(pe, d)
                torch._foreach_add_(pe, pm, alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            loss_acc += loss.detach()
            n_acc += 1

        if a.probe:
            if it - it0 == 30:
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t_probe = time.time()
            elif it - it0 == 30 + a.probe:
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                rate = a.probe / (time.time() - t_probe)
                gpu = f", GPU {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB" if dev.type == "cuda" else ""
                log(f"PROBE {rate * a.bs * world:.2f} samples/s {rate:.3f} it/s world {world}{gpu}, {host_mem()}")
                break
            continue
        is_eval = it % a.eval_every == 0 or it == a.iters
        if it % SYNC_EVERY == 0 or is_eval:
            # rank 0 decides when to stop and how long the cosine is; every rank applies the same
            flag = torch.tensor([0.0, float(total)], dtype=torch.float64, device=dev)
            if main_proc:
                now = time.time()
                if time_sched and deadline < math.inf and it - it0 >= 200:
                    rate = (it - it0) / (now - t_train0)
                    flag[1] = min(a.iters, it + rate * max(deadline - now, 0.0))
                flag[0] = float(now >= deadline)
            if world > 1:
                dist.broadcast(flag, 0)
            stop, total = bool(flag[0].item()), int(flag[1].item())

        if main_proc and it % a.log_every == 0 and not is_eval:
            rate = (it - it_log) / max(time.time() - t_log, 1e-6)
            gpu = f", GPU {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB" if dev.type == "cuda" else ""
            log(f"  it {it} loss {(loss_acc / max(n_acc, 1)).item():.5f} lr {lr_now:.2e} {rate:.2f} it/s "
                f"(x{a.bs * world} patches) cosine end ~{total} it{gpu}, {host_mem()}")
            t_log, it_log = time.time(), it
        if main_proc and (is_eval or stop):
            loss_avg = (loss_acc / max(n_acc, 1)).item()
            loss_acc.zero_()
            n_acc = 0
            log(evaluate_ema(it, loss_avg, lr_now))
    if stop:
        log(f"time budget reached at it {it} (epoch {it / epoch_iters:.1f}); continue with --resume <out>/last.pt")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if not main_proc or a.probe:
        return

    torch.save(ema.state_dict(), os.path.join(a.out, "ema_last.pt"))
    ema.load_state_dict(torch.load(os.path.join(a.out, "best_ema.pt"), map_location=dev))
    ema.eval()
    res = final_test(ema, test, deg, a.out, full=bool(a.q2n))
    summary = dict(model=a.model, params_M=n_par, iters=it, epochs=round(it / epoch_iters, 1), epoch_iters=epoch_iters,
                   gpus=world, best_val_PSNR=best, **res, bicubic_test=bic,
                   train_hours=round((time.time() - t_start) / 3600, 3), args=vars(a))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("TEST:", {k: round(v, 4) for k, v in res["test"].items()})
    log("TEST (self-ensemble):", {k: round(v, 4) for k, v in res["test_tta"].items()})
    log("consistency:", res["consistency"], "| self-ensemble:", res["consistency_tta"])


if __name__ == "__main__":
    main()
