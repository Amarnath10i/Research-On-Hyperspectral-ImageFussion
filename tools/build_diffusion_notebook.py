"""Regenerate the self-contained Kaggle notebook for Diffusion-NullFusion Chikusei.

The notebook embeds train_diffusion_nullfusion_chikusei.py in a %%writefile
cell, so it must be regenerated (not hand-edited) after any script change:

    python tools/build_diffusion_notebook.py            # default @1 output
    python tools/build_diffusion_notebook.py --tag @2   # next version

Outputs:
    experiments/notebooks/diffusion_nullfusion_push/
        diffusion_nullfusion_chikusei_x4<TAG>.ipynb
"""

from __future__ import annotations

import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "experiments", "scripts",
                      "train_diffusion_nullfusion_chikusei.py")
SRC_NB = os.path.join(REPO, "experiments", "notebooks",
                      "diffusion_nullfusion_push",
                      "diffusion_nullfusion_chikusei_x4.ipynb")
OUT_DIR = os.path.join(REPO, "experiments", "notebooks",
                       "diffusion_nullfusion_push")

WRITEFILE = "%%writefile train_diffusion_nullfusion_chikusei.py\n"


def build(tag: str) -> str:
    with open(SCRIPT, "r", encoding="utf-8") as f:
        script = f.read()
    with open(SRC_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    replaced = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if src.startswith("%%writefile train_diffusion_nullfusion_chikusei.py"):
            cell["source"] = [WRITEFILE] + script.splitlines(keepends=True)
            replaced = True
        # fresh version: clear stale outputs / execution counts
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    if not replaced:
        raise SystemExit("build_diffusion_notebook: %%writefile cell not found "
                         f"in {SRC_NB}")

    out = os.path.join(OUT_DIR, f"diffusion_nullfusion_chikusei_x4{tag}.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="@1",
                    help="version suffix for the output notebook")
    args = ap.parse_args()
    out = build(args.tag)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
