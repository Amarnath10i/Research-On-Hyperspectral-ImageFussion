"""Figures for the PUFormer paper, drawn from the result files in results/ (run from this folder).

    python make_figures.py            # writes figures/*.pdf
The visual comparison and band-wise figures need results/puformer_chikusei_x4_v3/paper_analysis/
(per_band.json, q2n.json, visual.npz), produced on Kaggle by methods/puformer/paper_analysis.py.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
V3 = os.path.join(RES, "puformer_chikusei_x4_v3")
ANA = os.path.join(V3, "paper_analysis")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# validated categorical slots 1-3 (blue, orange, aqua) + neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3de", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
MARK = dict(markersize=3.2, markeredgecolor=SURF, markeredgewidth=0.6)


def load(path):
    with open(path) as f:
        return json.load(f)


def training_curves():
    h1 = load(os.path.join(RES, "puformer_chikusei_x4", "puformer", "history.json"))
    h3 = [r for r in load(os.path.join(V3, "puformer", "history.json")) if r["it"] > 0]
    s1 = lambda it: it * 8 / 1e6                       # first run: batch 8
    base = s1(h1[-1]["it"])
    s3 = lambda it: base + it * 16 / 1e6               # v3 continues from the first run, batch 2 x 8
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    for ax, key, ylab, lim in ((axes[0], "val_PSNR", "Validation PSNR (dB)", (56.0, 59.0)),
                               (axes[1], "val_SAM", "Validation SAM (deg)", (0.68, 0.86))):
        ax.plot([s1(r["it"]) for r in h1], [r[key] for r in h1], color=ORANGE, marker="o", **MARK, label="First run (from scratch)")
        ax.plot([s3(r["it"]) for r in h3], [r[key] for r in h3], color=BLUE, marker="s", **MARK, label="v3 fine-tune (exact physics)")
        ax.axvline(base, color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.set_ylim(*lim)
        ax.set_xlabel("Training patches seen (millions)")
        ax.set_ylabel(ylab)
    axes[0].text(base + 0.03, 58.84, "fine-tune starts", color=INK2, fontsize=6.5)
    axes[0].legend(loc="center right", frameon=False)
    fig.savefig(os.path.join(FIG, "training.pdf"))
    plt.close(fig)


def robustness():
    g = load(os.path.join(V3, "puformer", "gaps.json"))
    rows = [("Blur $\\sigma$=1.5", "sigma=1.5"), ("Blur $\\sigma$=2.5", "sigma=2.5"), ("Blur $\\sigma$=3.0", "sigma=3.0"),
            ("SRF shift $-$8 nm", "srf_shift=-8nm"), ("SRF shift +8 nm", "srf_shift=+8nm"),
            ("Noise 40 dB", "noise_40dB"), ("Noise 35 dB", "noise_35dB"), ("Noise 30 dB", "noise_30dB")]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    y = np.arange(len(rows))[::-1]
    for (name, key, color, marker) in (("GSA", "gsa", AQUA, "^"), ("PUFormer, training operator", "fixed", ORANGE, "o"),
                                       ("PUFormer, true operator swapped in", "swap", BLUE, "s")):
        xs = [g[k][key]["PSNR"] if key in g[k] else np.nan for _, k in rows]
        ax.plot(xs, y, ls="none", marker=marker, color=color, markersize=4.2, markeredgecolor=SURF, markeredgewidth=0.6, label=name)
    ax.axvline(g["nominal"]["fixed"]["PSNR"], color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.text(g["nominal"]["fixed"]["PSNR"] - 0.3, len(rows) - 0.35, "nominal 58.15", ha="right", color=INK2, fontsize=6.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Test PSNR (dB)")
    ax.set_xlim(40, 60)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.38, -0.2), ncol=1, frameon=False)
    fig.savefig(os.path.join(FIG, "robustness.pdf"))
    plt.close(fig)


def band_wise():
    d = load(os.path.join(ANA, "per_band.json"))
    wl = np.array(d["wavelengths_nm"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    series = (("GSA", AQUA), ("PUFormer (first run)", ORANGE), ("PUFormer v3", BLUE))
    for ax, key, ylab in ((axes[0], "psnr", "Band PSNR (dB, peak 1)"), (axes[1], "rmse_dn", "Band RMSE (DN)")):
        for name, c in series:
            ax.plot(wl, d["per_band"][name][key], color=c, lw=1.1, label=name)
        ax.axvspan(363, 396, color=GRID, alpha=0.8, lw=0)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(ylab)
        ax.set_xlim(363, 1018)
    axes[1].set_yscale("log")
    axes[0].text(366, axes[0].get_ylim()[0] + 1.0, "no MSI\ncoverage", fontsize=6, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.savefig(os.path.join(FIG, "bandwise.pdf"))
    plt.close(fig)


def q2n_analysis():
    q = load(os.path.join(ANA, "q2n.json"))
    csv = os.path.join(RES, "puformer_chikusei_x4", "tip26_table4_comparison.csv")
    import csv as _csv
    rows = [r for r in _csv.DictReader(open(csv)) if "TIP'26 Table IV" in r["source"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    rel = np.array(q["relative_error_worst10pct_by_group_of_8_bands"])
    centers = np.linspace(363, 1018, 128).reshape(16, 8).mean(1)
    axes[0].bar(centers, rel, width=33, color=BLUE, edgecolor=SURF, linewidth=0.8)
    axes[0].set_xlabel("Band group centre wavelength (nm)")
    axes[0].set_ylabel("Error energy / local GT variance")
    axes[0].set_title("Worst 10% of 32$\\times$32 blocks", loc="left")
    c = q["q2n_vs_psnr_blend_to_bicubic"]
    axes[1].plot([r["PSNR"] for r in c], [r["Q2n"] for r in c], color=BLUE, marker="s", **MARK,
                 label="PUFormer v3 blended toward bicubic (our Q2n)")
    axes[1].plot([float(r["PSNR"]) for r in rows], [float(r["Q2n"]) for r in rows], ls="none", marker="o",
                 color=ORANGE, markersize=3.6, markeredgecolor=SURF, markeredgewidth=0.6, label="Methods as reported in TIP'26")
    axes[1].set_xlabel("PSNR (dB)")
    axes[1].set_ylabel("Q2n")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, borderaxespad=0.2)
    fig.savefig(os.path.join(FIG, "q2n.pdf"))
    plt.close(fig)


def visual():
    v = np.load(os.path.join(ANA, "visual.npz"))
    names = list(v["names"])
    tile = 6                                             # hardest test tile (lowest PSNR)
    gt = v["gt"][tile].astype(np.float32)
    lo, hi = np.percentile(gt, 1, axis=(0, 1)), np.percentile(gt, 99, axis=(0, 1))
    show = lambda x: np.clip((x.astype(np.float32) - lo) / (hi - lo), 0, 1)
    fig, axes = plt.subplots(2, len(names) + 1, figsize=(7.0, 2.9))
    axes[0, 0].imshow(show(gt)); axes[0, 0].set_title("Ground truth", fontsize=7)
    axes[1, 0].axis("off")
    vmax = 60.0
    for i, n in enumerate(names):
        axes[0, i + 1].imshow(show(v[f"rgb_{i}"][tile]))
        axes[0, i + 1].set_title(n.replace("PUFormer ", "PUFormer\n"), fontsize=7)
        im = axes[1, i + 1].imshow(v[f"mae_{i}"][tile].astype(np.float32), cmap="magma", vmin=0, vmax=vmax)
        axes[1, i + 1].set_xlabel(f"MAE {float(v[f'mae_{i}'][tile].astype(np.float32).mean()):.1f} DN", fontsize=6.5)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes[1, 1:].tolist(), fraction=0.02, pad=0.01)
    cb.set_label("Mean abs. error (DN)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(FIG, "visual.pdf"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    training_curves()
    robustness()
    if os.path.exists(os.path.join(ANA, "per_band.json")):
        band_wise(); q2n_analysis(); visual()
    print("figures:", sorted(os.listdir(FIG)))
