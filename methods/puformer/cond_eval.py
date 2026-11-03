"""One network for a family of sensors: test conditions with changed blur, SRF, noise, and all three at once.

    python cond_eval.py --dataset pavia --mat ... --ckpt best_ema.pt --model puformer [--pad zeros] --out DIR

Meant for the operator-randomized models (train.py --rand_sigma / --rand_shift / --rand_snr), but runs on any
checkpoint. Each model is scored with
    fixed      the operator of the training protocol in its data steps (or no operator at all, for other networks)
    oracle     the true test operator in its data steps
    estimated  an operator estimated from the test pair alone: D (spatial) and R (spectral) commute, so for the true
               blur width sigma and SRF shift d, D_sigma(Y_M) = R_d(Y_H) exactly, and
                   (sigma_hat, d_hat) = argmin ||D_sigma(Y_M) - R_d(Y_H)||^2
               over a grid (0.025 sigma0 and 1 nm steps), per test image
    estimated_ext  the same criterion over the wider separable family: blur widths along rows and columns and the
               sub-pixel offsets of the LR-HSI grid (coordinate search started at the estimate above); for models
               trained with --reg 1 this is their 'estimated' operator
Only operator-explicit models (PUFormer with a data step and a fixed exact or zero-border operator) use the oracle
and the estimate. GSA with the true operator is listed as a reference. Noise is drawn from a fixed seed per
condition, so every model sees the same inputs. NEON: the measured RGB cannot be re-simulated with another
response, so only the blur and the noise change. Writes DIR/cond_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from hsifuse.baselines import gsa
from hsifuse.config import DATASETS, MODELS, add_ablation_args, blur_of, build_model, set_metric_scale
from hsifuse.data import load_dataset, make_pairs
from hsifuse.metrics import evaluate
from hsifuse.ops import Degradation, corrupt, dataset_srf

KEEP = ("PSNR", "SAM", "ERGAS", "SSIM", "RMSE_DN")


def conditions(neon: bool):
    """(name, blur factor x sigma0, SRF shift nm, SNR dB or None, true kernel or None, MSI misregistration px).
    The last group lies outside the operator family the network and the estimator know (isotropic Gaussian blur,
    shifted SRF, co-registered inputs): an anisotropic or a defocus (disk) blur, and an HR-MSI shifted by a
    fraction of a pixel against the LR-HSI. Their 'oracle' gives the network the true kernel (none exists for the
    misregistration)."""
    c = [("nominal", 1.0, 0, None)]
    c += [(f"sigma x{f:g}", f, 0, None) for f in (0.5, 0.75, 1.25, 1.5, 2.0)]
    if not neon:
        c += [(f"shift {d:+d} nm", 1.0, d, None) for d in (-12, -8, -4, 4, 8, 12)]
    c += [(f"snr {n} dB", 1.0, 0, n) for n in (40, 35, 30)]
    c += [("sigma x1.5, shift +8 nm, snr 35 dB", 1.5, 8, 35), ("sigma x0.75, shift -8 nm, snr 40 dB", 0.75, -8, 40),
          ("sigma x2, shift +12 nm, snr 30 dB", 2.0, 12, 30)]
    if neon:
        c = [(n.replace(", shift +8 nm", "").replace(", shift -8 nm", "").replace(", shift +12 nm", ""), f, 0, s)
             for n, f, _, s in c]
    c = [(n, f, d, s, None, 0.0) for n, f, d, s in c]
    c += [("aniso blur 1.5/0.75 sigma0, 30 deg", 1.0, 0, None, ("aniso", 1.5, 0.75, 30.0), 0.0),
          ("aniso blur 1.25/0.5 sigma0, 0 deg", 1.0, 0, None, ("aniso", 1.25, 0.5, 0.0), 0.0),
          ("disk blur r = 1.25 sigma0", 1.0, 0, None, ("disk", 1.25), 0.0)]
    if not neon:
        c += [("MSI misregistered 0.5 px", 1.0, 0, None, None, 0.5), ("MSI misregistered 1 px", 1.0, 0, None, None, 1.0),
              ("aniso blur, MSI 0.5 px, snr 35 dB", 1.0, 0, 35, ("aniso", 1.5, 0.75, 30.0), 0.5)]
    # the LR-HSI sampled on a grid offset against the HR grid of the HR-MSI and the reference (misregistration with
    # the output on the HR-MSI grid), alone and with an axis-aligned anisotropic blur and noise
    c += [("HSI grid offset 0.5/0.5 px", 1.0, 0, None, ("gauss", 1.0, 1.0, 0.5, 0.5), 0.0),
          ("HSI grid offset 1/-1 px", 1.0, 0, None, ("gauss", 1.0, 1.0, 1.0, -1.0), 0.0),
          ("aniso 0.75/1.5 sigma0 + offset 0.5/-0.25 px, snr 35 dB", 1.0, 0, 35, ("gauss", 0.75, 1.5, 0.5, -0.25), 0.0)]
    # corrupted observations with a clean reference: local changes in the HR-MSI (another scene's content, as between
    # two acquisition dates) and dead or miscalibrated LR-HSI columns (hsifuse.ops.corrupt, fixed seed)
    c += [("MSI local changes", 1.0, 0, None, ("corrupt", True, False, False, False), 0.0),
          ("HSI stripes", 1.0, 0, None, ("corrupt", False, True, False, False), 0.0),
          ("MSI changes + HSI stripes, snr 35 dB", 1.0, 0, 35, ("corrupt", True, True, False, False), 0.0)]
    # registered and unregistered pixels in one image, and band-dependent, partly signal-dependent HSI noise
    c += [("MSI locally misregistered", 1.0, 0, None, ("corrupt", False, False, True, False), 0.0),
          ("HSI band-dependent noise 25-45 dB", 1.0, 0, None, ("corrupt", False, False, False, True), 0.0),
          ("MSI locally misregistered + HSI band noise", 1.0, 0, None, ("corrupt", False, False, True, True), 0.0)]
    # misregistration with noise only; and a sub-pixel misregistration of a sharp HR-MSI (band-limited shift, as a
    # sensor sampling the shifted scene records it, instead of the bilinear shift above, which also blurs it)
    if not neon:
        c += [("MSI misregistered 0.5 px + snr 35 dB", 1.0, 0, 35, None, 0.5),
              ("MSI misregistered 1 px + snr 35 dB", 1.0, 0, 35, None, 1.0),
              ("MSI misregistered 0.5 px band-limited", 1.0, 0, None, None, 0.5),
              ("MSI misregistered 0.5 px band-limited + snr 35 dB", 1.0, 0, 35, None, 0.5)]
    return c


def true_kernel(spec, ksize, s0):
    """(ksize, ksize) normalized blur kernel outside the Gaussian family of the protocol."""
    ax = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    if spec[0] == "aniso":
        _, fx, fy, deg = spec
        t = np.deg2rad(deg)
        xr, yr = np.cos(t) * xx + np.sin(t) * yy, -np.sin(t) * xx + np.cos(t) * yy
        k = np.exp(-(xr ** 2 / (2 * (fx * s0) ** 2) + yr ** 2 / (2 * (fy * s0) ** 2)))
    else:
        k = (xx ** 2 + yy ** 2 <= (spec[1] * s0) ** 2).astype(np.float64)
    return (k / k.sum()).astype(np.float32)


def translate(x, px):
    """Every band shifted by px pixels along both axes (bilinear, reflected border)."""
    n, _, h, w = x.shape
    theta = torch.tensor([[1.0, 0.0, 2 * px / (w - 1)], [0.0, 1.0, 2 * px / (h - 1)]], dtype=x.dtype, device=x.device)
    grid = torch.nn.functional.affine_grid(theta.expand(n, 2, 3), list(x.shape), align_corners=True)
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def translate2(x, pr, pc):
    """x shifted by pr pixels along rows and pc along columns: out(u) = x(u + p) (bicubic, reflected border)."""
    n, _, h, w = x.shape
    theta = torch.tensor([[1.0, 0.0, 2 * pc / (w - 1)], [0.0, 1.0, 2 * pr / (h - 1)]], dtype=x.dtype, device=x.device)
    grid = torch.nn.functional.affine_grid(theta.expand(n, 2, 3), list(x.shape), align_corners=True)
    return torch.nn.functional.grid_sample(x, grid, mode="bicubic", padding_mode="reflection", align_corners=True)


def fshift(x, pr, pc, pad=16):
    """x shifted by pr pixels along rows and pc along columns, out(u) = x(u + p), with a band-limited (Fourier)
    shift of the reflect-padded image: no interpolation blur."""
    n, b, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    fr = torch.fft.fftfreq(h + 2 * pad, device=x.device, dtype=x.dtype)[:, None]
    fc = torch.fft.fftfreq(w + 2 * pad, device=x.device, dtype=x.dtype)[None, :]
    ramp = torch.exp(2j * torch.pi * (fr * pr + fc * pc))
    out = torch.fft.ifft2(torch.fft.fft2(xp) * ramp).real
    return out[..., pad:pad + h, pad:pad + w]


def deg_at(a, sigma, shift=0.0, pad=None, sigma_c=None, off=(0.0, 0.0), ksize=None):
    return Degradation(dataset_srf(a.dataset, shift_nm=shift), scale=a.scale, ksize=ksize or blur_of(a.scale)[0],
                       sigma=sigma, pad=pad or a.pad, shift_nm=shift, sigma_c=sigma_c, off=off)


def add_noise(x, snr_db, g):
    p = x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
    return x + torch.randn(x.shape, generator=g, device="cpu").to(x) * torch.sqrt(p / 10 ** (snr_db / 10))


@torch.no_grad()
def estimate(lr, ms, a, sig_grid, shift_grid, dev):
    """Per-image (sigma_hat, d_hat) minimising ||D_sigma(Y_M) - R_d(Y_H)||^2 (reflect border, as the simulation)."""
    ryh = torch.stack([deg_at(a, 1.0, float(d)).to(dev).R(lr) for d in shift_grid])          # (Gd, N, M, h, w)
    err = torch.stack([((deg_at(a, float(sg), pad="reflect").to(dev).D(ms, pad_mode="reflect")[None] - ryh) ** 2)
                       .flatten(2).mean(2) for sg in sig_grid])                              # (Gs, Gd, N)
    flat = err.flatten(0, 1).argmin(0).cpu().numpy()
    return sig_grid[flat // len(shift_grid)], shift_grid[flat % len(shift_grid)]


OFF_GRID = np.arange(-2.0, 2.0 + 1e-9, 0.125)
# the estimator's kernel support: wide enough for a Gaussian moved by the largest offset (the protocol's
# (2s-1)-tap kernel would cut it off and make the search trade offset against blur width)
EST_PAD = 2 * int(np.ceil(np.abs(OFF_GRID).max()))


@torch.no_grad()
def estimate_ext(lr, ms, a, sig_grid, shift_grid, sig0, sh0, dev, sweeps=3, ksize=None):
    """Per image, coordinate search for (sigma_r, sigma_c, off_r, off_c, d) minimising ||D(Y_M) - R_d(Y_H)||^2 over
    the separable family, started at the isotropic, centered estimate (sig0, sh0)."""
    out = []
    for i in range(lr.shape[0]):
        yh, ym = lr[i:i + 1], ms[i:i + 1]
        th = dict(s=float(sig0[i]), c=float(sig0[i]), r=0.0, q=0.0, d=float(sh0[i]))
        ryh = {}

        def j(t):
            if t["d"] not in ryh:
                ryh[t["d"]] = deg_at(a, 1.0, t["d"]).to(dev).R(yh)
            d = deg_at(a, t["s"], pad="reflect", sigma_c=t["c"], off=(t["r"], t["q"]), ksize=ksize).to(dev)
            return float(((d.D(ym, pad_mode="reflect") - ryh[t["d"]]) ** 2).mean())

        for _ in range(sweeps):
            for key, grid in (("r", OFF_GRID), ("q", OFF_GRID), ("s", sig_grid), ("c", sig_grid), ("d", shift_grid)):
                vals = [j(dict(th, **{key: float(g)})) for g in grid]
                th[key] = float(grid[int(np.argmin(vals))])
        out.append(th)
    return out


@torch.no_grad()
def estimate_np(lr, ms, a, sh, dev, ksize=None, ridge=1e-8):
    """Per image, the blur kernel of any shape from D_k(Y_M) = R_d(Y_H) (Theorem ident.(a)): the relation is
    linear in k, so k is the least-squares solution over all bands and low-resolution pixels with taps summing to
    one (KKT system; d: the response shift of the separable estimate). Patches are taken exactly as D samples:
    reflect padding by ksize // 2, a ksize x ksize correlation window, stride = scale, starting at 0."""
    ks = ksize or blur_of(a.scale)[0]
    p, n, out = ks // 2, ks * ks, []
    for i in range(lr.shape[0]):
        ryh = deg_at(a, 1.0, float(sh[i])).to(dev).R(lr[i:i + 1])                      # (1, M, h, w)
        ym = torch.nn.functional.pad(ms[i:i + 1], (p, p, p, p), mode="reflect")
        m = ym.shape[1]
        patches = torch.nn.functional.unfold(ym, ks, stride=a.scale)                     # (1, M*n, h*w)
        A = patches.view(m, n, -1).permute(0, 2, 1).reshape(-1, n).double()
        b = ryh.reshape(-1).double()
        assert A.shape[0] == b.shape[0], (A.shape, b.shape)
        G, g = A.T @ A, A.T @ b
        G = G + ridge * torch.trace(G) / n * torch.eye(n, dtype=G.dtype, device=G.device)
        K = torch.zeros(n + 1, n + 1, dtype=G.dtype, device=G.device)
        K[:n, :n], K[:n, n], K[n, :n] = G, 1.0, 1.0
        rhs = torch.cat([g, torch.ones(1, dtype=G.dtype, device=G.device)])
        out.append(torch.linalg.solve(K, rhs)[:n].view(ks, ks).float())
    return out


def deg_np(a, k, sig, sh, dev):
    """The operator with the nonparametric kernel k (not separable: the network takes its general data step)."""
    d = deg_at(a, float(sig), float(sh)).to(dev)
    d.k = k[None, None].to(dev)
    d.gaussian = False
    return d


def deg_of(a, th, dev):
    return deg_at(a, th["s"], th["d"], sigma_c=th["c"], off=(th["r"], th["q"])).to(dev)


@torch.no_grad()
def change_auroc(model, lr, ms, degs, masks):
    """How well the confidence-weighted data step finds the corrupted inputs: ROC AUC of 1 - w (last stage)
    against the corruption masks, for the HR-MSI (local changes) and the LR-HSI (stripes)."""
    from sklearn.metrics import roc_auc_score
    sh, sm = [], []
    for i in range(lr.shape[0]):
        model.deg = degs[i]
        model(lr[i:i + 1], ms[i:i + 1])
        wh, wm = model.last_conf
        sh.append(1 - wh), sm.append(1 - wm)
    out = {}
    for key, sc, m in (("hsi", torch.cat(sh), masks[0]), ("msi", torch.cat(sm), masks[1])):
        y = m.flatten().cpu().numpy()
        if 0 < y.mean() < 1:
            out[key] = float(roc_auc_score(y, sc.flatten().cpu().numpy()))
    return out


@torch.no_grad()
def residual_auroc(lr, ms, degs, masks, scale):
    """Model-free change detector: per low-resolution pixel the energy of D Y_M - R Y_H over the bands (estimated
    operator, Theorem ident.(a)), normalized by its median per image; scored as is against the LR-HSI mask and
    upsampled (nearest) against the HR-MSI mask."""
    from sklearn.metrics import roc_auc_score
    maps = []
    for i in range(lr.shape[0]):
        d = degs[i]
        e = (d.D(ms[i:i + 1], pad_mode="reflect") - d.R(lr[i:i + 1])).pow(2).mean(1, keepdim=True)
        maps.append(e / (e.median() + 1e-12))
    e = torch.cat(maps)
    out = {}
    for key, sc, m in (("hsi", e, masks[0]), ("msi", torch.nn.functional.interpolate(e, scale_factor=scale), masks[1])):
        y = m.flatten().cpu().numpy()
        if 0 < y.mean() < 1:
            out[key] = float(roc_auc_score(y, sc.flatten().cpu().numpy()))
    return out


@torch.no_grad()
def run(model, lr, ms, degs=None):
    out = []
    for i in range(lr.shape[0]):
        if degs is not None:
            model.deg = degs[i]
        out.append(model(lr[i:i + 1], ms[i:i + 1]))
    return torch.cat(out).clamp(0, 1)


def score(gt, pred):
    m = evaluate(gt, pred)
    return {k: round(float(m[k]), 4) for k in KEEP if k in m}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--cache", default="")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="puformer", choices=MODELS)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--pad", default="reflect", choices=["reflect", "zeros"])
    add_ablation_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--np_kernel", type=int, default=0, help="1: also give the network the nonparametric kernel "
                   "estimated by least squares from D_k(Y_M) = R_d(Y_H) (mode estimated_np)")
    p.add_argument("--only", default="", help="comma-separated parts of condition names: evaluate only the matching "
                   "conditions and skip the sensitivity sweep, which runs alone with --only sensitivity "
                   "(fetch_results.py merges the result into the full file)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    _, te, _, peak = load_dataset(a.dataset, a.mat, a.cache or None)
    set_metric_scale(a, peak)
    gt = torch.from_numpy(np.asarray(te)).to(dev)
    neon = a.dataset == "neon"
    s0 = blur_of(a.scale)[1]
    base = deg_at(a, s0).to(dev)
    gtb = gt[:, :base.srf.shape[0]]
    model = build_model(a, base).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    if hasattr(model, "amp"):
        model.amp = dev == "cuda"
    swappable = a.model == "puformer" and a.op == "exact" and bool(a.physics)

    sig_grid = np.round(np.arange(0.25 * s0, 2.5 * s0 + 1e-9, 0.025 * s0), 4)
    shift_grid = np.array([0.0]) if neon else np.arange(-16.0, 16.0 + 1e-9, 1.0)
    res = dict(dataset=a.dataset, scale=a.scale, model=a.model, pad=a.pad, op=a.op, physics=a.physics,
               ckpt=a.ckpt, nominal_sigma=s0, swappable=swappable, conditions={})
    only = [t for t in a.only.split(",") if t]
    if swappable and (not only or "sensitivity" in only):
        try:
            res["sensitivity"] = sensitivity(a, gt, gtb, base, model, s0, dev)
        except Exception as err:
            print("FAILED sensitivity", repr(err), flush=True)
            model.deg = base
        with open(os.path.join(a.out, "cond_eval.json"), "w") as fh:     # also when no condition follows
            json.dump(res, fh, indent=1)
    for idx, cond in enumerate(conditions(neon)):
        if only and not any(t in cond[0] for t in only):
            continue
        try:
            evaluate_condition(idx, cond, a, gt, gtb, base, model, swappable, s0, sig_grid, shift_grid, dev, res)
        except Exception as err:                         # one failing condition must not lose the others
            print("FAILED", cond[0], repr(err), flush=True)
            if swappable:
                model.deg = base
        with open(os.path.join(a.out, "cond_eval.json"), "w") as fh:
            json.dump(res, fh, indent=1)


@torch.no_grad()
def sensitivity(a, gt, gtb, base, model, s0, dev):
    """Data made with the nominal operator; the network is given a wrong blur width sigma~ (Theorem 2). For each
    sigma~: the relative operator residual ||(D~ - D) X*|| / ||D X*|| on the test images, and the test PSNR."""
    lr, ms, _ = make_pairs(gt, base)
    yh = base.D(gtb, pad_mode="reflect")
    out = []
    for f in (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
        d = deg_at(a, f * s0).to(dev)
        rel = float((d.D(gtb, pad_mode=a.pad) - yh).flatten(1).norm(dim=1).mean() / yh.flatten(1).norm(dim=1).mean())
        out.append(dict(sigma_factor=f, rel_op_residual=rel, **score(gtb, run(model, lr, ms, [d] * len(lr)))))
        print(f"sensitivity sigma x{f:g}: residual {rel:.4f} PSNR {out[-1]['PSNR']:.2f}", flush=True)
    model.deg = base
    return out


def evaluate_condition(idx, cond, a, gt, gtb, base, model, swappable, s0, sig_grid, shift_grid, dev, res):
    name, f, shift, snr, kernel, msi_px = cond
    if kernel is not None and kernel[0] == "gauss":           # separable: anisotropic widths, grid offsets
        _, fr, fc, orr, oc = kernel
        true = deg_at(a, fr * s0, shift, sigma_c=fc * s0, off=(orr, oc)).to(dev)
    else:
        true = deg_at(a, f * s0, shift).to(dev)
    if kernel is not None and kernel[0] not in ("gauss", "corrupt"):
        true.k = torch.from_numpy(true_kernel(kernel, true.ksize, s0))[None, None].to(dev)
        true.gaussian = False
    lr, ms, _ = make_pairs(gt, true)
    masks = None
    if kernel is not None and kernel[0] == "corrupt":
        lr, ms, masks = corrupt(lr, ms, np.random.default_rng(2000 + idx), changes=kernel[1], stripes=kernel[2],
                                misreg=kernel[3], bandnoise=kernel[4], return_masks=True)
    sharp = "band-limited" in name
    if msi_px:
        ms = true.R(fshift(gtb, msi_px, msi_px) if sharp else translate(gtb, msi_px))
    if snr:
        g = torch.Generator().manual_seed(1000 + idx)
        lr, ms = add_noise(lr, snr, g), add_noise(ms, snr, g)
    sig, sh = estimate(lr, ms, a, sig_grid, shift_grid, dev)
    resid = (residual_auroc(lr, ms, [deg_at(a, float(sg), float(d)).to(dev) for sg, d in zip(sig, sh)], masks,
                           a.scale) if masks is not None else None)
    ext = estimate_ext(lr, ms, a, sig_grid, shift_grid, sig, sh, dev) if swappable or msi_px else None
    gauss = kernel is None or kernel[0] == "corrupt"
    row = dict(sigma=f * s0 if gauss else None, shift_nm=shift, snr_db=snr, kernel=kernel, msi_px=msi_px,
               sigma_hat=[float(v) for v in sig], shift_hat=[float(v) for v in sh],
               sigma_abs_err=float(np.abs(sig - f * s0).mean()) if gauss else None,
               shift_abs_err=float(np.abs(sh - shift).mean()))
    if ext is not None:
        row["ext_hat"] = ext
        if kernel is not None and kernel[0] == "gauss":
            row["ext_abs_err"] = dict(
                sigma_r=float(np.mean([abs(t["s"] - kernel[1] * s0) for t in ext])),
                sigma_c=float(np.mean([abs(t["c"] - kernel[2] * s0) for t in ext])),
                off_r=float(np.mean([abs(t["r"] - kernel[3]) for t in ext])),
                off_c=float(np.mean([abs(t["q"] - kernel[4]) for t in ext])))
    row["gsa_oracle"] = score(gtb, gsa(lr, ms, true).clamp(0, 1))
    if swappable:
        model.deg = base
    row["fixed"] = score(gtb, run(model, lr, ms))
    if swappable:
        if not msi_px:                                     # a misregistration has no operator to supply
            row["oracle"] = score(gtb, run(model, lr, ms, [true] * len(lr)))
        iso = [deg_at(a, float(sg), float(d)).to(dev) for sg, d in zip(sig, sh)]
        wide = [deg_of(a, t, dev) for t in ext]
        reg = bool(getattr(model, "reg", False))
        row["estimated"] = score(gtb, run(model, lr, ms, wide if reg else iso))
        if not reg:
            row["estimated_ext"] = score(gtb, run(model, lr, ms, wide))
        if getattr(a, "np_kernel", 0):
            npk = estimate_np(lr, ms, a, sh, dev)
            row["estimated_np"] = score(gtb, run(model, lr, ms, [deg_np(a, k, sg, d, dev)
                                                             for k, sg, d in zip(npk, sig, sh)]))
            kt = true.k[0, 0].float()
            row["np_kernel_relerr"] = float(np.mean([float((k.to(kt) - kt).norm() / kt.norm()) for k in npk]))
        if masks is not None and getattr(model, "cwd", False):
            row["change_auroc"] = change_auroc(model, lr, ms, wide if reg else iso, masks)
        model.deg = base
    if msi_px and ext is not None:
        # misregistered HR-MSI; the reference lies on the LR-HSI grid. The wide estimate gives the offset of the
        # LR-HSI grid against the HR-MSI (about -msi_px), which registers either the HR-MSI before fusion (any
        # method: "register, then fuse") or PUFormer's result, fused with that offset in its operator, afterwards
        # registration estimate with a kernel support that holds the moved Gaussian (EST_PAD)
        ext_reg = estimate_ext(lr, ms, a, sig_grid, shift_grid, sig, sh, dev, ksize=blur_of(a.scale)[0] + EST_PAD)
        row["reg_hat"] = ext_reg
        shift_by = fshift if sharp else translate2         # the same interpolation as the misregistration
        register = lambda x, es: torch.cat([shift_by(x[i:i + 1], es[i]["r"], es[i]["q"]) for i in range(len(x))])
        ms_reg = register(ms, ext_reg)
        row["gsa_reg"] = score(gtb, gsa(lr, ms_reg, base).clamp(0, 1))
        row["reg_msi"] = score(gtb, run(model, lr, ms_reg, [base] * len(lr) if swappable else None))
        if swappable:          # the offsets of the operator it fused with
            row["reg_out"] = score(gtb, register(run(model, lr, ms, [deg_of(a, t, dev) for t in ext]), ext))
            model.deg = base
    if resid:
        row["residual_auroc"] = resid
    res["conditions"][name] = row
    err = f"sigma_err {row['sigma_abs_err']:.3f}" if gauss else f"sigma_hat {np.mean(sig):.2f}"
    print(name, err, f"shift_err {row['shift_abs_err']:.2f}",
          {k: row[k]["PSNR"] for k in ("gsa_oracle", "fixed", "oracle", "estimated") if k in row}, flush=True)


if __name__ == "__main__":
    main()
