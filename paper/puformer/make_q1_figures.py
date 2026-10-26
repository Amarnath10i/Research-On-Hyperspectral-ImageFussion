"""Figures of the extended study (run from this folder, on Kaggle with the datasets attached).

    Q1_PREDS=<folder with <job>/test_pred_f16.npy> python make_q1_figures.py
writes figures/q1_scales.pdf (PSNR against scale), figures/q1_visual.pdf (error maps of every method) and
figures/q1_operator.pdf (operator ablation under blur mismatch). The visual figure needs the test ground
truth, which it rebuilds with methods/puformer/hsifuse (the datasets must be attached); it is skipped
otherwise.
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from make_figures import AQUA, BLUE, FIG, INK2, MARK, ORANGE  # noqa: E402  (shared style)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("Q1_RESULTS", os.path.join(HERE, "..", "..", "results", "q1"))
PREDS = os.environ.get("Q1_PREDS", "")
DATASETS = {"chikusei": ("Chikusei", (4,)), "pavia": ("Pavia Centre", (4,))}   # scope of the paper
METHODS = [("ssrnet", "SSR-NET"), ("psrt", "PSRT"), ("dct", "DCT"), ("mimformer", "MIMFormer"),
           ("dhif", "DHIF-Net"), ("puformer", "PUFormer")]
COLOR = {"ssrnet": "#8c8b86", "psrt": AQUA, "dct": ORANGE, "mimformer": "#7d52c7", "dhif": "#c2410c",
         "puformer": BLUE, "gsa": INK2}
FALSE_RGB = {"chikusei": (60, 40, 20), "pavia": (57, 34, 3), "cave": (25, 15, 5), "harvard": (25, 15, 5)}


def load(job, name="results.json"):
    p = os.path.join(RES, job, name)
    return json.load(open(p)) if os.path.exists(p) else None


def scales_figure():
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.9))
    for ax, (ds, (title, scales)) in zip(axes, DATASETS.items()):
        for m, lab in METHODS + [("gsa", "GSA")]:
            xs, ys = [], []
            for s in scales:
                r = load(f"bench-{ds}-x{s}-{'puformer' if m == 'gsa' else m}")
                v = (r or {}).get("gsa_test" if m == "gsa" else "test")
                if v:
                    xs.append(s), ys.append(v["PSNR"])
            if xs:
                ax.plot(xs, ys, marker="o", color=COLOR[m], label=lab, lw=1.6 if m == "puformer" else 1.1, **MARK)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(scales), [f"$\\times${s}" for s in scales])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_ylabel("PSNR (dB)" if ds == "chikusei" else "")
    axes[0].legend(loc="lower left", frameon=False, fontsize=6, ncol=1)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "q1_scales.pdf"))
    plt.close(fig)


def operator_figure():
    """Test PSNR against the test-time blur sigma, for the operator variants (swap where possible)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    variants = [("full", "exact operator, swapped", BLUE, "swap"), ("zeros", "zero border, swapped", ORANGE, "swap"),
                ("learned", "learned operator", AQUA, "fixed"), ("nophys", "no data step", "#8c8b86", "fixed"),
                ("full", "exact operator, not swapped", BLUE, "fixedfull")]
    for ax, ds, title in ((axes[0], "pavia", "Pavia Centre $\\times$4"), (axes[1], "chikusei", "Chikusei $\\times$4")):
        for v, lab, col, mode in variants:
            job = f"bench-{ds}-x4-puformer" if v == "full" else f"abl-{ds}-x4-{v}"
            g = load(job, "gaps.json")
            if not g:
                continue
            xs, ys = [], []
            for sig, key in ((1.5, "sigma=1.5"), (2.0, "nominal"), (2.5, "sigma=2.5"), (3.0, "sigma=3.0")):
                row = g.get(key, {})
                k = "fixed" if key == "nominal" or mode != "swap" else "swap"
                if k in row:
                    xs.append(sig), ys.append(row[k]["PSNR"])
            if xs:
                ax.plot(xs, ys, marker="o", color=col, ls="--" if mode == "fixedfull" else "-", label=lab, **MARK)
        ax.axvline(2.0, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel("test-time blur $\\sigma$ (training: 2.0)")
        ax.set_title(title)
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].legend(loc="lower center", frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q1_operator.pdf"))
    plt.close(fig)


def false_color(x, ds):
    r, g, b = FALSE_RGB[ds]
    rgb = np.stack([x[r], x[g], x[b]], -1).astype(np.float64)
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99.5)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def visual_figure():
    if not PREDS:
        print("Q1_PREDS not set: visual figure skipped")
        return
    sys.path.insert(0, os.path.join(HERE, "..", "..", "methods", "puformer"))
    from hsifuse.data import load_dataset  # noqa: E402
    data = dict(chikusei=sorted(p for p in glob.glob("/kaggle/input/**/*.mat", recursive=True)
                                if "chikusei" in p.lower() and "Ground" not in p),
                pavia=glob.glob("/kaggle/input/**/Pavia.mat", recursive=True),
                cave=[os.path.dirname(os.path.dirname(os.path.dirname(p))) for p in
                      glob.glob("/kaggle/input/**/Train/HSI/balloons.mat", recursive=True)],
                harvard=[os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p)))) for p in
                         glob.glob("/kaggle/input/**/Data/Train/HSI/img1.mat", recursive=True)])
    cols = [("gsa", "GSA")] + METHODS
    rows = [d for d in DATASETS if data[d]]
    fig, axes = plt.subplots(len(rows), len(cols) + 1, figsize=(7.16, 1.02 * len(rows) + 0.25))
    axes = np.atleast_2d(axes)
    for i, ds in enumerate(rows):
        _, te, _, peak = load_dataset(ds, sorted(data[ds], key=len)[0], "/tmp/harvard" if ds == "harvard" else None)
        p = load(f"bench-{ds}-x4-puformer")
        k = int(np.argmin([x["PSNR"] for x in p["test_per_image"]])) if p else 0   # hardest image for PUFormer
        gt = np.asarray(te[k], np.float32)
        errs = {}
        import torch  # noqa: E402
        from hsifuse.baselines import gsa  # noqa: E402
        from hsifuse.ops import Degradation, dataset_srf  # noqa: E402
        deg = Degradation(dataset_srf(ds), scale=4, pad="reflect")
        lr, ms = deg.simulate(torch.from_numpy(gt)[None])
        errs["gsa"] = np.abs(gsa(lr, ms, deg).clamp(0, 1)[0].numpy() - gt).mean(0) * peak
        for m, _ in cols:
            if m == "gsa":
                continue
            f = os.path.join(PREDS, f"bench-{ds}-x4-{m}", "test_pred_f16.npy")
            if os.path.exists(f):
                errs[m] = np.abs(np.load(f, mmap_mode="r")[k].astype(np.float32) - gt).mean(0) * peak
        # color scale from the learned methods only (GSA's much larger error saturates)
        learned = [e.ravel() for m, e in errs.items() if m != "gsa"]
        vmax = np.percentile(np.concatenate(learned), 99.5) if learned else 1
        axes[i, 0].imshow(false_color(gt, ds))
        axes[i, 0].set_ylabel(DATASETS[ds][0], fontsize=7)
        for j, (m, lab) in enumerate(cols, start=1):
            ax = axes[i, j]
            if m in errs:
                ax.imshow(errs[m], cmap="magma", vmin=0, vmax=vmax)
                ax.text(0.03, 0.04, f"{errs[m].mean():.2f}", transform=ax.transAxes, color="white", fontsize=5.5)
            if i == 0:
                ax.set_title(lab, fontsize=7)
        if i == 0:
            axes[i, 0].set_title("Ground truth", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.2)
    fig.savefig(os.path.join(FIG, "q1_visual.pdf"), dpi=220)
    fig.savefig(os.path.join(FIG, "q1_visual.png"), dpi=150)       # preview for checking the figure
    plt.close(fig)


COND_LINES = [("bench-{d}-x4-mimformer", "fixed", "MIMFormer, nominal", "#7d52c7", ":"),
              ("cond-{d}-x4-mimformer", "fixed", "MIMFormer, randomized", "#7d52c7", "-"),
              ("bench-{d}-x4-puformer", "estimated", "PUFormer, nominal, est. operator", BLUE, ":"),
              ("cond-{d}-x4-nocond", "estimated", "PUFormer, randomized data, est.", AQUA, "-"),
              ("cond-{d}-x4-puformer", "estimated", "PUFormer, randomized + operator, est.", BLUE, "-")]


def cond_figure():
    """PSNR against the test blur and against the SRF shift, for the operator-randomized study (cond_eval.json)."""
    rows = [(d, t) for d, t in (("chikusei", "Chikusei"), ("pavia", "Pavia Centre"))
            if load(f"cond-{d}-x4-puformer", "cond_eval.json")]
    if not rows:
        print("cond_eval.json not available: cond figure skipped")
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.0, 2.0 * len(rows) + 0.3), squeeze=False)
    for i, (d, title) in enumerate(rows):
        for job, mode, lab, col, ls in COND_LINES:
            e = load(job.format(d=d), "cond_eval.json")
            if not e:
                continue
            c = e["conditions"]
            s0 = e["nominal_sigma"]
            blur = sorted([(r["sigma"] / s0, r[mode]["PSNR"]) for k, r in c.items()
                           if (k == "nominal" or (k.startswith("sigma x") and "snr" not in k)) and mode in r])
            shift = sorted([(r["shift_nm"], r[mode]["PSNR"]) for k, r in c.items()
                            if (k == "nominal" or k.startswith("shift ")) and mode in r])
            for ax, pts in ((axes[i, 0], blur), (axes[i, 1], shift)):
                if pts:
                    ax.plot(*zip(*pts), marker="o", color=col, ls=ls, label=lab, lw=1.5 if "operator" in lab else 1.1,
                            **MARK)
        axes[i, 0].axvline(1.0, color=INK2, lw=0.6, ls=":")
        axes[i, 1].axvline(0.0, color=INK2, lw=0.6, ls=":")
        axes[i, 0].set_ylabel(f"{title}\nPSNR (dB)")
    axes[-1, 0].set_xlabel("test blur $\\sigma/\\sigma_0$")
    axes[-1, 1].set_xlabel("test SRF shift (nm)")
    axes[0, 1].legend(loc="lower center", frameon=False, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "q1_cond.pdf"))
    plt.close(fig)


NEON_FIG = [("bench-neon-x4-mimformer", "MIMFormer"), ("bench-neon-x4-dhif", "DHIF-Net"),
            ("bench-neon-x4-puformer", "PUFormer")]


def neon_figure():
    """Full-resolution fusion of the real NEON data: HSI (1 m), camera RGB (0.25 m), GSA and three networks."""
    ref = os.path.join(RES, "bench-neon-x4-puformer", "neon_fullres_crop.npz")
    if not os.path.exists(ref):
        print("neon_fullres_crop.npz not available: NEON figure skipped")
        return
    base = np.load(ref)
    panels = [("HSI, 1 m", base["hsi"]), ("Camera RGB, 0.25 m", base["rgb"]), ("GSA", base["gsa"])]
    for job, lab in NEON_FIG:
        f = os.path.join(RES, job, "neon_fullres_crop.npz")
        if os.path.exists(f):
            panels.append((lab, np.load(f)["model"]))
    hs = np.stack([p for _, p in panels if _ != "Camera RGB, 0.25 m"]).astype(np.float32)
    lo, hi = np.percentile(hs, 1), np.percentile(hs, 99.5)                   # one stretch for all HSI panels
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 7.16 / len(panels) + 0.25))
    for ax, (lab, x) in zip(axes, panels):
        x = np.moveaxis(x.astype(np.float32), 0, -1)
        if lab.startswith("Camera"):
            img = np.clip(x / max(np.percentile(x, 99.5), 1e-6), 0, 1)
        else:
            img = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(img)
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.2, w_pad=0.15)
    fig.savefig(os.path.join(FIG, "q1_neon.pdf"), dpi=220)
    fig.savefig(os.path.join(FIG, "q1_neon.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    operator_figure()          # (scales_figure needs several scale factors; the paper reports x4 only)
    cond_figure()
    neon_figure()
    visual_figure()
