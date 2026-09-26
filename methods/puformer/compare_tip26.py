"""Rank a PUFormer results.json against every Chikusei row of TIP'26 Table IV.

    python compare_tip26.py path/to/results.json [--key test_tta] [--name "PUFormer v3"]

Prints a markdown table (best per column in bold) and the rank of our row on each metric.
SSIM: TIP'26 does not say how it computes SSIM. We show both of our definitions:
SSIM_psrt (PSRT cal_ssim.py, data range 1 = the dataset peak, consistent with the fixed-peak PSNR
that TIP'26 uses) and the stricter SSIM with data range = each image's own max.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "results", "puformer_chikusei_x4", "tip26_table4_comparison.csv")
# (column, higher is better, digits); SSIM is filled from our SSIM_psrt / strict SSIM separately
COLS = [("PSNR", True, 4), ("SSIM", True, 4), ("SAM", False, 4), ("ERGAS", False, 4), ("Q2n", True, 4),
        ("CC", True, 4), ("SCC", True, 4), ("RMSE_DN", False, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--key", default="test_tta", help="test (plain) or test_tta (self-ensemble)")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CSV)) if "TIP'26 Table IV" in r["source"]]
    table = [(r["method"], {c: float(r[c]) for c, _, _ in COLS if r.get(c)}) for r in rows]
    ours = []
    for i, path in enumerate(a.results):
        r = json.load(open(path))
        res = r.get(a.key, r["test"])
        name = a.name or f"PUFormer ({os.path.basename(os.path.dirname(os.path.abspath(path)))}, {a.key})"
        if "SSIM_psrt" in res:
            ours.append((f"**{name}** (SSIM = SSIM_psrt)", dict(res, SSIM=res["SSIM_psrt"])))
        ours.append((f"{name} (SSIM = strict, per-image max)", dict(res)))
    allrows = table + ours
    best = {c: (max if hi else min)(v[c] for _, v in allrows if c in v) for c, hi, _ in COLS}
    print("| Method | " + " | ".join(f"{c} {'↑' if hi else '↓'}" for c, hi, _ in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for name, v in allrows:
        cells = []
        for c, _, d in COLS:
            if c not in v:
                cells.append("–")
                continue
            s = f"{v[c]:.{d}f}"
            cells.append(f"**{s}**" if abs(v[c] - best[c]) < 10 ** -d / 2 else s)
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    for name, v in ours:
        ranks = []
        for c, hi, _ in COLS:
            if c in v:
                others = [t[c] for _, t in table if c in t]
                ranks.append(f"{c} {1 + sum((o > v[c]) if hi else (o < v[c]) for o in others)}/{len(others) + 1}")
        print(f"{name}: rank " + ", ".join(ranks))


if __name__ == "__main__":
    main()
