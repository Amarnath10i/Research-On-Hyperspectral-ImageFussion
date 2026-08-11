"""Push every proposal notebook to Kaggle as its own kernel.

    python tools/push_all_proposals.py            # all four
    python tools/push_all_proposals.py proposal3  # just one
    python tools/push_all_proposals.py --quick    # QUICK=True variants

Four separate kernels, one per proposal, each with both datasets attached.
They can then run concurrently.

ACCELERATOR - read this before running anything
-----------------------------------------------
kernel-metadata carries `enable_gpu`, which Kaggle resolves to its current
default GPU. The `accelerator` field is accepted by the API and then ignored.
Measured three times: a push lands on a P100, and torch >= 2.6 ships no sm_60
kernels, so every CUDA op fails.

A push also RESETS an accelerator chosen in the web UI, so the order matters:

    1. run this script once to create/update the kernels
    2. for each kernel: open it -> Settings -> Accelerator -> "GPU T4 x2" -> Save
    3. Run All from the UI
    4. watch from here:
         python tools/kaggle_autorun.py --watch-only --slug <slug> ...

Do not push again between steps 2 and 3, or the accelerator reverts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import kaggle_autorun as K  # noqa: E402

DATASETS = [
    "nikeshreddypatlolla/cave-dataset-2",
    "nikeshreddypatlolla/harvard-hsi-2",
]

NOTEBOOKS = {
    "proposal1": ("proposal1/notebooks/DAETF_Net_Kaggle_GPU.ipynb", "daetf-net-fusion"),
    "proposal2": ("proposal2/notebooks/unfoldfusion_Kaggle_GPU.ipynb", "unfoldfusion-fusion"),
    "proposal3": ("proposal3/notebooks/continuumfusion_Kaggle_GPU.ipynb", "continuumfusion-fusion"),
    "proposal4": ("proposal4/notebooks/zerofusion_Kaggle_GPU.ipynb", "zerofusion-fusion"),
}


def make_quick(nb: dict) -> int:
    """Flip QUICK on so the first push validates cheaply."""
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if line.startswith("QUICK = False"):
                cell["source"][i] = "QUICK = True\n"
                n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("which", nargs="*", default=list(NOTEBOOKS))
    p.add_argument("--quick", action="store_true",
                   help="push QUICK=True variants (minutes, validates the run)")
    p.add_argument("--state-dir", default=os.path.join(REPO, "kaggle_runs"))
    a = p.parse_args()

    username, _ = K.load_credentials()
    api = K.kaggle_api()
    os.makedirs(a.state_dir, exist_ok=True)

    pushed = []
    for key in a.which:
        rel, slug_base = NOTEBOOKS[key]
        src = os.path.join(REPO, rel)
        if not os.path.exists(src):
            print(f"!! {key}: notebook missing at {src}")
            continue

        with open(src, "r", encoding="utf-8") as f:
            nb = json.load(f)
        slug_name = slug_base + ("-quick" if a.quick else "")
        if a.quick:
            n = make_quick(nb)
            print(f"{key}: QUICK enabled ({n} line(s))")

        work = os.path.join(a.state_dir, key)
        os.makedirs(work, exist_ok=True)
        staged = os.path.join(work, os.path.basename(rel))
        with open(staged, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        slug = f"{username}/{slug_name}"
        print(f"{key}: pushing -> https://www.kaggle.com/{slug}")
        try:
            K.push(api, staged, slug, os.path.join(work, "_push"),
                   DATASETS, [], gpu=True, private=True, internet=False)
            pushed.append((key, slug))
            print("   ok")
        except Exception as exc:
            print(f"   FAILED: {exc}")

    print("\n" + "=" * 68)
    print("PUSHED KERNELS")
    for key, slug in pushed:
        print(f"  {key:11s} https://www.kaggle.com/{slug}")
    print("=" * 68)
    print("""
NEXT - one manual step per kernel, because the API cannot select the GPU:

  open the kernel  ->  Settings  ->  Accelerator  ->  "GPU T4 x2"  ->  Save
  then  Run All

A P100 (Kaggle's default here) cannot run torch >= 2.6 at all: it is sm_60 and
the shipped build targets sm_70+. Each notebook checks this in its first cell
and will tell you immediately if the accelerator is wrong.

Then watch a run without disturbing its settings:
  python tools/kaggle_autorun.py --watch-only \\
      --notebook <path> --slug <slug> --poll 60 --timeout 540
""")
    return 0 if pushed else 1


if __name__ == "__main__":
    raise SystemExit(main())
