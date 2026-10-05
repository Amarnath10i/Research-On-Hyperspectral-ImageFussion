"""Download the output of finished session kernels and keep the small result files in results/q1/<job>/.

    KAGGLE_USERNAME=... KAGGLE_KEY=... python fetch_results.py --kernel <user>/puf-s01 [--kaggle <cli path>] [--keep DIR]

Copied into git: results.json, history.json, gaps.json, train.log. best_ema.pt and test_pred_f16.npy are
large; they stay in --keep (default: a folder next to the download) for the figure kernel.
"""
import argparse
import glob
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.normpath(os.path.join(HERE, "..", "..", "..", "results", "q1"))
SMALL = ("results.json", "history.json", "gaps.json", "train.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--kaggle", default="kaggle")
    ap.add_argument("--keep", default="")
    a = ap.parse_args()
    tmp = a.keep or tempfile.mkdtemp(prefix="puf_")
    os.makedirs(tmp, exist_ok=True)
    subprocess.run([a.kaggle, "kernels", "output", a.kernel, "-p", tmp, "-o"], check=True)
    jobs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(tmp, "**", "results.json"), recursive=True)})
    for d in jobs:
        name = os.path.basename(d)
        out = os.path.join(DEST, name)
        os.makedirs(out, exist_ok=True)
        for f in SMALL:
            if os.path.exists(os.path.join(d, f)):
                shutil.copy2(os.path.join(d, f), os.path.join(out, f))
        print("kept", name)
    logs = glob.glob(os.path.join(tmp, "*.log"))
    for p in logs:
        shutil.copy2(p, os.path.join(DEST, f"{a.kernel.split('/')[1]}.kernel.log"))
    print(f"{len(jobs)} jobs from {a.kernel}; large files in {tmp}")


if __name__ == "__main__":
    main()
