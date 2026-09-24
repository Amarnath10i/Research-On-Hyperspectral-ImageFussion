"""Train / evaluate on Chikusei x4 (TIP'26 protocol: WV2 8-band MSI, Gaussian 7x7 sigma=2).

    python train.py --mat /kaggle/input/chikusei --model puformer --hours 10.5 --out /kaggle/working/puformer
    python train.py --mat ... --model ssrnet --iters 20000 --out .../ssrnet

Model selection uses the 64 validation patches only; the 8 test images are
evaluated once, at the end, with the best-validation EMA weights.
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
from hsifuse.metrics import evaluate
from hsifuse.models import build
from hsifuse.ops import Degradation, wv2_srf


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True, help=".mat file or a folder containing it")
    p.add_argument("--cache", default="", help="optional .npy cache of the normalised crop")
    p.add_argument("--model", default="puformer", choices=["puformer", "ssrnet"])
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--iters", type=int, default=300000)
    p.add_argument("--hours", type=float, default=0.0, help="wall-clock budget; 0 = unlimited")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny synthetic run for CI / CPU")
    return p.parse_args()


@torch.no_grad()
def predict(model, lr, ms, bs=1):
    return torch.cat([model(lr[i:i + bs], ms[i:i + bs]) for i in range(0, lr.shape[0], bs)])


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

    test = make_pairs(torch.from_numpy(te).to(dev), deg)
    val = make_pairs(torch.from_numpy(va).to(dev), deg)
    sampler = PatchSampler(tr, deg, a.patch, dev)
    del cube

    model = build(a.model, deg, **(dict(width=a.width, stages=a.stages) if a.model == "puformer" else {})).to(dev)
    if a.model == "puformer":
        model.amp = bool(a.amp)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    n_par = sum(q.numel() for q in model.parameters()) / 1e6
    print(f"{a.model}: {n_par:.2f}M params on {dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    def sched(it):  # warmup, then cosine over whichever runs out first: iterations or wall-clock
        if it < 1000:
            return it / 1000
        prog = it / a.iters
        if a.hours:
            prog = max(prog, (time.time() - t_start) / (a.hours * 3600 * 0.97))
        return max(0.5 * (1 + math.cos(math.pi * min(prog, 1.0))), 1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp) and dev == "cuda")

    ck = os.path.join(a.out, "last.pt")
    it, best, hist = 0, -1.0, []
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]), ema.load_state_dict(s["ema"]), opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"]); it, best, hist = s["it"], s["best"], s["hist"]
        print(f"resumed at it={it} best_val={best:.3f}", flush=True)

    # bicubic reference on test (sanity check of the pipeline)
    up = F.interpolate(test[0], scale_factor=4, mode="bicubic", align_corners=False)
    bic = evaluate(test[2].cpu(), up.cpu())
    print("bicubic test:", {k: round(v, 4) for k, v in bic.items()}, flush=True)

    t0, budget = time.time(), a.hours * 3600
    model.train()
    while it < a.iters:
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(it)
        lr_, ms_, gt_ = sampler(a.bs)
        outs = model(lr_, ms_, return_all=True)
        loss = F.l1_loss(outs[-1], gt_) + 0.1 * sum(F.l1_loss(o, gt_) for o in outs[:-1])
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
            hist.append(dict(it=it, loss=loss.item(), **{f"val_{k}": x for k, x in v.items()}))
            if v["PSNR"] > best:
                best = v["PSNR"]
                torch.save(ema.state_dict(), os.path.join(a.out, "best_ema.pt"))
            print(f"it {it:6d} loss {loss.item():.5f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"val PSNR {v['PSNR']:.3f} SAM {v['SAM']:.3f} best {best:.3f} "
                  f"({(time.time()-t_start)/3600:.2f} h)", flush=True)
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                            scaler=scaler.state_dict(), it=it, best=best, hist=hist), ck)
            json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
        if out_of_time:
            print("time budget reached", flush=True)
            break

    ema.load_state_dict(torch.load(os.path.join(a.out, "best_ema.pt"), map_location=dev))
    t1 = time.time()
    pred = predict(ema, test[0], test[1])
    infer_s = (time.time() - t1) / test[0].shape[0]
    res = evaluate(test[2].cpu(), pred.cpu())
    per_img = [evaluate(test[2][i:i + 1].cpu(), pred[i:i + 1].cpu()) for i in range(pred.shape[0])]
    summary = dict(model=a.model, params_M=n_par, iters=it, best_val_PSNR=best, test=res,
                   test_per_image=per_img, bicubic_test=bic, infer_s_per_256=infer_s, args=vars(a))
    json.dump(summary, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    np.save(os.path.join(a.out, "test_pred.npy"), pred.cpu().numpy().astype(np.float16))
    print("TEST:", {k: round(v, 4) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
