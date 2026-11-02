"""Identifiability, conditioning and stability of the operator estimate (Theorem ident.) on the test images.

    python operator_analysis.py --dataset chikusei --scale 4 --mat ... --cache ... --out DIR [--ckpt (ignored)]

For the residual r(theta) = D_theta(Y_M) - R_delta(Y_H) with theta = (sigma_r, sigma_c, off_r, off_c, delta):
  jacobian    G = dr/dtheta at the true parameters (central differences); local identifiability of the continuous
              parameters is full column rank of G, its conditioning sigma_max / sigma_min (physical units, and with
              unit-norm columns, which measures how collinear the parameters are)
  crb         covariance of the least-squares estimate under white observation noise, to first order:
              (G^T G)^-1 G^T S G (G^T G)^-1 with S the variance of D N_M - R N_H per residual entry
  empirical   grid start and Gauss-Newton refinement from noisy pairs (several SNRs and noise seeds): RMSE, bias and
              the coverage of the Laplace interval +-2 sd evaluated at the estimate
  degenerate  conditioning for scenes where the theory predicts indistinguishable configurations: spatially constant
              (blur and offsets unidentifiable), spectrally flat (response shift unidentifiable), spatially smooth
  residual    relative operator residual ||(D_theta' - D_theta) X|| / ||D_theta X|| for blur errors of 2-50 %, true and
              first-order (the quantity the mismatch theorem bounds the reconstruction displacement with)
Observations are simulated with a kernel wide enough for the offsets (blur_of + EST_PAD), so the family is exact.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from cond_eval import EST_PAD, add_noise, deg_at, estimate
from hsifuse.config import DATASETS, blur_of
from hsifuse.data import load_dataset, make_pairs

PARAMS = ("sigma_r", "sigma_c", "off_r", "off_c", "shift_nm")
STEPS = np.array([0.01, 0.01, 0.01, 0.01, 0.1])            # central-difference steps: px, px, px, px, nm


def operators(a, th, ks, dev):
    d = deg_at(a, float(th[0]), pad="reflect", sigma_c=float(th[1]), off=(float(th[2]), float(th[3])), ksize=ks)
    return d.to(dev), deg_at(a, 1.0, float(th[4])).to(dev)


def residual(a, th, yh, ym, ks, dev):
    d, r = operators(a, th, ks, dev)
    return (d.D(ym, pad_mode="reflect") - r.R(yh)).flatten()


def jacobian(a, th, yh, ym, ks, dev):
    cols = []
    for j, h in enumerate(STEPS):
        tp, tm = np.array(th, float), np.array(th, float)
        tp[j] += h
        tm[j] -= h
        cols.append((residual(a, tp, yh, ym, ks, dev) - residual(a, tm, yh, ym, ks, dev)) / (2 * h))
    return torch.stack(cols, 1)                                      # (n, 5), float64


def noise_var(a, th, yh_clean, ym_clean, snr, ks, dev):
    """Variance of D N_M - R N_H per residual entry (band-major, as residual() flattens), for add_noise at snr."""
    d, r = operators(a, th, ks, dev)
    s2h = float(yh_clean.pow(2).mean()) / 10 ** (snr / 10)
    s2m = float(ym_clean.pow(2).mean()) / 10 ** (snr / 10)
    k2 = float(d.k.double().pow(2).sum())
    srf2 = r.srf.double().pow(2).sum(0)                              # (M,)
    hw = yh_clean.shape[-1] * yh_clean.shape[-2]
    return (s2m * k2 + s2h * srf2).repeat_interleave(hw).to(dev)


def ls_cov(G, v):
    hinv = torch.linalg.pinv(G.T @ G)
    return hinv @ (G.T @ (G * v[:, None])) @ hinv


def svd_stats(G):
    s = torch.linalg.svdvals(G)
    gn = G / G.norm(dim=0, keepdim=True).clamp_min(1e-300)       # a zero column (unidentifiable) stays zero
    sn = torch.linalg.svdvals(gn)
    return dict(sv=s.tolist(), cond=float(s[0] / s[-1]), sv_norm=sn.tolist(), cond_norm=float(sn[0] / sn[-1]))


LIMIT = np.array([0.5, 0.5, 0.5, 0.5, 2.0])                    # largest step per iteration: px, px, px, px, nm


def gauss_newton(a, th0, yh, ym, ks, dev, iters=20):
    """Levenberg-Marquardt refinement of theta from th0: damped normal equations, step-limited, descent only."""
    th, lam = np.array(th0, float), 1e-3
    j_old = float(residual(a, th, yh, ym, ks, dev).pow(2).sum())
    for _ in range(iters):
        r = residual(a, th, yh, ym, ks, dev)
        G = jacobian(a, th, yh, ym, ks, dev)
        H = G.T @ G
        damp = lam * torch.diag(torch.diag(H).clamp_min(1e-12))
        step = torch.linalg.lstsq(H + damp, (G.T @ r)[:, None]).solution[:, 0].cpu().numpy()
        step = np.clip(step, -LIMIT, LIMIT)
        cand = th - step
        cand[:2] = np.maximum(cand[:2], 0.2)
        j_new = float(residual(a, cand, yh, ym, ks, dev).pow(2).sum())
        if j_new < j_old:
            th, j_old, lam = cand, j_new, max(lam / 3, 1e-9)
            if np.abs(step).max() < 1e-6:
                break
        else:
            lam *= 10
            if lam > 1e8:
                break
    return th


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--cache", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--snrs", default="40,30,20")
    p.add_argument("--seeds", type=int, default=3)
    a, _ = p.parse_known_args()                                      # model flags of the eval kernel are ignored
    a.pad = "reflect"
    os.makedirs(a.out, exist_ok=True)
    dev = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    _, te, _, _ = load_dataset(a.dataset, a.mat, a.cache or None)
    gt = torch.from_numpy(np.asarray(te)).double()
    ks0, s0 = blur_of(a.scale)
    ks = ks0 + EST_PAD
    snrs = [int(v) for v in a.snrs.split(",")]
    sig_grid = np.round(np.arange(0.25 * s0, 2.5 * s0 + 1e-9, 0.025 * s0), 4)
    shift_grid = np.arange(-16.0, 16.0 + 1e-9, 1.0)
    cases = {"nominal": (s0, s0, 0.0, 0.0, 0.0),
             "anisotropic": (1.5 * s0, 0.75 * s0, 0.0, 0.0, 0.0),
             "offsets and shift": (s0, s0, 0.5, -0.25, 8.0)}
    res = dict(dataset=a.dataset, scale=a.scale, sigma0=s0, ksize=ks, params=PARAMS, steps=STEPS.tolist(),
               snrs=snrs, seeds=a.seeds, cases={}, degenerate={}, residual=[])
    out = os.path.join(a.out, "operator_analysis.json")

    def save():
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)

    for name, th_true in cases.items():
        d_true, r_true = operators(a, th_true, ks, dev)
        lr, ms = d_true.D(gt, pad_mode="reflect"), r_true.R(gt)
        per = [svd_stats(jacobian(a, th_true, lr[i:i + 1], ms[i:i + 1], ks, dev)) for i in range(len(gt))]
        row = dict(theta_true=list(th_true), cond=float(np.mean([q["cond"] for q in per])),
                   cond_norm=float(np.mean([q["cond_norm"] for q in per])),
                   cond_norm_max=float(np.max([q["cond_norm"] for q in per])),
                   sv=np.mean([q["sv"] for q in per], 0).tolist(), crb={}, empirical={})
        for snr in snrs:
            sd = []
            for i in range(len(gt)):
                G = jacobian(a, th_true, lr[i:i + 1], ms[i:i + 1], ks, dev)
                sd.append(torch.sqrt(torch.diag(ls_cov(G, noise_var(a, th_true, lr[i:i + 1], ms[i:i + 1], snr, ks,
                                                                     dev)))).tolist())
            row["crb"][str(snr)] = np.mean(sd, 0).tolist()
            errs, cover, failed = [], [], []
            for seed in range(a.seeds):
                g = torch.Generator().manual_seed(5000 + 97 * seed + snr)
                lrn, msn = add_noise(lr.float(), snr, g).double(), add_noise(ms.float(), snr, g).double()
                sig, sh = estimate(lrn.float(), msn.float(), a, sig_grid, shift_grid, dev)
                for i in range(len(gt)):
                    th0 = (float(sig[i]), float(sig[i]), 0.0, 0.0, float(sh[i]))
                    try:
                        th = gauss_newton(a, th0, lrn[i:i + 1], msn[i:i + 1], ks, dev)
                        G = jacobian(a, th, lrn[i:i + 1], msn[i:i + 1], ks, dev)
                        sd_hat = torch.sqrt(torch.diag(ls_cov(G, noise_var(a, th, lr[i:i + 1], ms[i:i + 1], snr, ks,
                                                                           dev)))).numpy()
                    except Exception as err:                          # noqa: BLE001 - record and continue
                        failed.append(f"{snr} dB seed {seed} image {i}: {err!r}"[:200])
                        continue
                    e = th - np.array(th_true)
                    errs.append(e)
                    cover.append(np.abs(e) <= 2 * sd_hat)
            errs = np.array(errs) if errs else np.full((1, 5), np.nan)
            row["empirical"][str(snr)] = dict(rmse=np.sqrt((errs ** 2).mean(0)).tolist(), bias=errs.mean(0).tolist(),
                                              max_abs=np.abs(errs).max(0).tolist(),
                                              coverage=np.mean(cover, 0).tolist() if cover else None,
                                              n=len(errs), failed=failed)
            print(name, snr, "dB: crb sd", np.round(row["crb"][str(snr)], 4), "rmse",
                  np.round(row["empirical"][str(snr)]["rmse"], 4), "coverage",
                  np.round(row["empirical"][str(snr)]["coverage"] or [], 2), flush=True)
        res["cases"][name] = row
        save()

    # scenes for which the theory predicts indistinguishable configurations
    th_true = cases["nominal"]
    d_true, r_true = operators(a, th_true, ks, dev)
    inten = gt.mean(1, keepdim=True)
    smooth_k = deg_at(a, 6.0, pad="reflect", ksize=25)
    scenes = {"spatially constant": gt.mean((2, 3), keepdim=True).expand_as(gt).clone(),
              "spectrally flat": inten.expand_as(gt).clone(),
              "spatially smooth (6 px blur)": smooth_k.blur(gt.float(), pad_mode="reflect").double()}
    for name, x in scenes.items():
        lr, ms = d_true.D(x, pad_mode="reflect"), r_true.R(x)
        per = []
        for i in range(len(x)):
            G = jacobian(a, th_true, lr[i:i + 1], ms[i:i + 1], ks, dev)
            norms = G.norm(dim=0)
            st = dict(col_norm=norms.tolist())
            try:
                st.update(svd_stats(G))
                st["crb30"] = torch.sqrt(torch.diag(ls_cov(G, noise_var(a, th_true, lr[i:i + 1], ms[i:i + 1], 30, ks,
                                                                        dev)))).tolist()
            except Exception as err:                                  # singular: unidentifiable
                st["singular"] = repr(err)
            per.append(st)
        res["degenerate"][name] = dict(
            col_norm=np.mean([q["col_norm"] for q in per], 0).tolist(),
            cond_norm=float(np.median([q.get("cond_norm", np.inf) for q in per])),
            crb30=np.median([q.get("crb30", [np.inf] * 5) for q in per], 0).tolist())
        print(name, res["degenerate"][name], flush=True)
        save()

    # operator residual of a blur error: true value and first-order prediction
    lr = d_true.D(gt, pad_mode="reflect")
    dsig = (operators(a, (s0 + 0.01, s0 + 0.01, 0, 0, 0), ks, dev)[0].D(gt, pad_mode="reflect")
            - operators(a, (s0 - 0.01, s0 - 0.01, 0, 0, 0), ks, dev)[0].D(gt, pad_mode="reflect")) / 0.02
    for f in (0.02, 0.05, 0.1, 0.25, 0.5):
        for sgn in (-1, 1):
            s1 = s0 * (1 + sgn * f)
            dx = operators(a, (s1, s1, 0, 0, 0), ks, dev)[0].D(gt, pad_mode="reflect") - lr
            res["residual"].append(dict(rel_sigma_err=sgn * f, rho_true=float(dx.norm() / lr.norm()),
                                        rho_first_order=float(abs(s1 - s0) * dsig.norm() / lr.norm())))
    save()
    print("residual", res["residual"], flush=True)


if __name__ == "__main__":
    main()
