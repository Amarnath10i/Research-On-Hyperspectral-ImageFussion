"""Write the Kaggle notebook + kernel-metadata.json that trains PUFormer on Chikusei x4.

The notebook clones this repo at a fixed commit, so every Kaggle run is tied to a commit.
    # first run (from scratch, + gap evaluation + SSRNet anchor):
    python build_notebook.py --commit <sha> --hours 9.5 --gaps 1 --ssrnet 1
    # fine-tune run v2 (warm start from a dataset holding best_ema.pt, both T4s):
    python build_notebook.py --commit <sha> --user amarnath10chinu --slug puformer-chikusei-x4-v2 \
        --init_dataset amarnath10chinu/puformer-chikusei-x4-init --hours 10.8 \
        --train_args "--bs 16 --lr 1.5e-4 --epochs 2000 --eval_epochs 40 --snap_epochs 200 --w_sam 0.05 --w_ssim 0.1"
    # continue an interrupted run: add --resume_kernel <user>/<slug> (its output is mounted as input)
    python -c "from kaggle.api.kaggle_api_extended import KaggleApi as K; a=K(); a.authenticate(); a.kernels_push('.')"
"""
import argparse
import json
import os

REPO = "https://github.com/Amarnath10i/Research-On-Hyperspectral-ImageFussion.git"


def cells(a):
    run = lambda cmd: (
        "import subprocess, sys\n"
        f"p = subprocess.Popen({cmd!r}, shell=True, cwd='/kaggle/working/repo/methods/puformer',\n"
        "                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
        "for line in p.stdout: print(line, end='', flush=True)\n"
        "assert p.wait() == 0, 'command failed'\n")
    common = "--mat /kaggle/input --cache /tmp/chikusei_crop.npy"
    pu = f"--model puformer --width {a.width} --stages {a.stages}"
    out = "/kaggle/working/out/puformer"
    extra = ""
    if a.init_dataset:
        extra += " --init_ckpt $(find /kaggle/input -path '*init*' -name best_ema.pt | head -1)"
    if a.resume_kernel:
        extra += " --resume $(find /kaggle/input -name last.pt | head -1)"
    out_cells = [
        "!nvidia-smi\n"
        f"!git clone -q {REPO} /kaggle/working/repo && cd /kaggle/working/repo && git checkout -q {a.commit} && git log --oneline -1\n"
        "!python -c \"import torch, h5py; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))\"\n"
        "!find /kaggle/input -maxdepth 4 | head -30",
        "# 1) PUFormer training (checkpoints: last.pt every eval, best_ema.pt, EMA snapshots)\n" + run(
            f"python train.py {common} {pu} --hours {a.hours} {a.train_args}{extra} --out {out}"),
    ]
    if a.gaps:
        out_cells.append("# 2) gap evaluation: consistency / blur+SRF mismatch with operator swap / noise\n" + run(
            f"python eval_gaps.py {common} {pu} --ckpt {out}/best_ema.pt --out {out}"))
    if a.ssrnet:
        out_cells.append("# 3) SSRNet anchor under the identical protocol\n" + run(
            f"python train.py {common} --model ssrnet --bs 16 --lr 1e-3 --iters 30000 --hours 0.5 "
            "--eval_every 2000 --out /kaggle/working/out/ssrnet"))
    out_cells.append(
        "import json, glob\n"
        "for f in sorted(glob.glob('/kaggle/working/out/*/results.json')):\n"
        "    r = json.load(open(f)); print(f, r['params_M'], r.get('iters'), r.get('epochs'))\n"
        "    for k in ('test', 'test_tta'):\n"
        "        if k in r: print(' ', k, {m: round(v, 4) for m, v in r[k].items()})\n"
        "!rm -rf /kaggle/working/repo; ls -la /kaggle/working/out/*")
    return out_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--hours", type=float, default=9.5)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--train_args", default="--bs 8")
    ap.add_argument("--init_dataset", default="", help="Kaggle dataset holding best_ema.pt to warm start from")
    ap.add_argument("--resume_kernel", default="", help="earlier kernel whose output holds last.pt")
    ap.add_argument("--gaps", type=int, default=0)
    ap.add_argument("--ssrnet", type=int, default=0)
    ap.add_argument("--slug", default="puformer-chikusei-x4")
    ap.add_argument("--title", default="")
    ap.add_argument("--user", default="amarnathmadaka")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id=f"c{i}",
                          source=src) for i, src in enumerate(cells(a))])
    name = f"{a.slug}.ipynb"
    json.dump(nb, open(os.path.join(here, name), "w"), indent=1)
    title = a.title or a.slug.replace("puformer", "PUFormer").replace("chikusei", "Chikusei").replace("-", " ")
    meta = dict(id=f"{a.user}/{a.slug}", title=title, code_file=name, language="python",
                kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=True,
                machine_shape="NvidiaTeslaT4",
                dataset_sources=["mingliu123/chikusei"] + ([a.init_dataset] if a.init_dataset else []),
                competition_sources=[], kernel_sources=[a.resume_kernel] if a.resume_kernel else [])
    json.dump(meta, open(os.path.join(here, "kernel-metadata.json"), "w"), indent=1)
    print("wrote", name, "and kernel-metadata.json for", meta["id"], "at commit", a.commit)


if __name__ == "__main__":
    main()
