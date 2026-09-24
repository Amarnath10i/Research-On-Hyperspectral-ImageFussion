"""Write the Kaggle notebook + kernel-metadata.json that trains PUFormer on Chikusei x4.

The notebook clones this repo at a fixed commit, so every Kaggle run is tied to a commit.
    python build_notebook.py --commit <sha> --hours 9.5
    python -c "from kaggle.api.kaggle_api_extended import KaggleApi as K; a=K(); a.authenticate(); a.kernels_push('.')"
"""
import argparse
import json
import os

REPO = "https://github.com/Amarnath10i/Research-On-Hyperspectral-ImageFussion.git"


def cells(commit, hours, width, stages, bs):
    run = lambda cmd: (
        "import subprocess, sys\n"
        f"p = subprocess.Popen({cmd!r}, shell=True, cwd='/kaggle/working/repo/methods/puformer',\n"
        "                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
        "for line in p.stdout: print(line, end='', flush=True)\n"
        "assert p.wait() == 0, 'command failed'\n")
    common = "--mat /kaggle/input --cache /tmp/chikusei_crop.npy"
    pu = f"--model puformer --width {width} --stages {stages}"
    return [
        "!nvidia-smi\n"
        f"!git clone -q {REPO} /kaggle/working/repo && cd /kaggle/working/repo && git checkout -q {commit} && git log --oneline -1\n"
        "!python -c \"import torch, h5py; print(torch.__version__, torch.cuda.get_device_name(0))\"",
        "# 1) PUFormer - main model\n" + run(
            f"python train.py {common} {pu} --bs {bs} --hours {hours} --out /kaggle/working/out/puformer"),
        "# 2) gap evaluation: consistency / blur+SRF mismatch with operator swap / noise\n" + run(
            f"python eval_gaps.py {common} {pu} --ckpt /kaggle/working/out/puformer/best_ema.pt "
            "--out /kaggle/working/out/puformer"),
        "# 3) SSRNet anchor under the identical protocol\n" + run(
            f"python train.py {common} --model ssrnet --bs 16 --lr 1e-3 --iters 30000 --hours 0.5 "
            "--eval_every 2000 --out /kaggle/working/out/ssrnet"),
        "import json, glob\n"
        "for f in sorted(glob.glob('/kaggle/working/out/*/results.json')):\n"
        "    r = json.load(open(f)); print(f, r['params_M'], {k: round(v, 4) for k, v in r['test'].items()})\n"
        "!rm -rf /kaggle/working/repo /kaggle/working/out/*/last.pt",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--hours", type=float, default=9.5)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--slug", default="puformer-chikusei-x4")
    ap.add_argument("--user", default="amarnathmadaka")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id=f"c{i}",
                          source=src) for i, src in enumerate(cells(a.commit, a.hours, a.width, a.stages, a.bs))])
    name = f"{a.slug}.ipynb"
    json.dump(nb, open(os.path.join(here, name), "w"), indent=1)
    meta = dict(id=f"{a.user}/{a.slug}", title="PUFormer Chikusei x4", code_file=name, language="python",
                kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=True,
                machine_shape="NvidiaTeslaT4", dataset_sources=["mingliu123/chikusei"],
                competition_sources=[], kernel_sources=[])
    json.dump(meta, open(os.path.join(here, "kernel-metadata.json"), "w"), indent=1)
    print("wrote", name, "for commit", a.commit)


if __name__ == "__main__":
    main()
