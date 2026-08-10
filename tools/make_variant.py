"""Produce a run variant of the notebook without editing the source.

    python tools/make_variant.py --quick   -> a short smoke run
    python tools/make_variant.py --full    -> the full run

Used to validate on Kaggle cheaply before spending GPU quota on a full run.
"""

from __future__ import annotations

import argparse
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "proposal1", "notebooks", "DAETF_Net_Kaggle_P100.ipynb")


def patch(nb: dict, quick: bool, ablation: bool) -> int:
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        for i, line in enumerate(src):
            if line.startswith("QUICK = ") and quick:
                src[i] = "QUICK = True\n"
                n += 1
            elif line.startswith("RUN_ABLATION = ") and ablation:
                src[i] = "RUN_ABLATION = True\n"
                n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="short smoke run")
    p.add_argument("--full", action="store_true", help="full run (default)")
    p.add_argument("--ablation", action="store_true", help="also run the ablation")
    p.add_argument("--out", help="output path")
    a = p.parse_args()

    with open(SRC, "r", encoding="utf-8") as f:
        nb = json.load(f)
    n = patch(nb, quick=a.quick, ablation=a.ablation)

    suffix = "quick" if a.quick else "full"
    if a.ablation:
        suffix += "-ablation"
    out = a.out or os.path.join(REPO, "proposal1", "notebooks",
                                f"DAETF_Net_Kaggle_P100_{suffix}.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"patched {n} line(s) -> {out}")


if __name__ == "__main__":
    main()
