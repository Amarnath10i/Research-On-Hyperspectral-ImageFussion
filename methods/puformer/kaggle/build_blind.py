"""Write a CPU Kaggle kernel that runs blind_eval.py on checkpoints from an earlier session of the same account.

    python build_blind.py --dataset pavia --source puf-s01 --user amarnath10chinu \
        --full bench-pavia-x4-puformer --zeros abl-pavia-x4-zeros --learned abl-pavia-x4-learned
    kaggle kernels push -p blind/<slug>

CPU only (no GPU quota); the source session's output is attached as a kernel source.
"""
import argparse
import json
import os

from build_jobs import DATASETS, notebook
from build_notebook import HERE, code_blob, git_rev

CELL = r'''import base64, glob, hashlib, io, os, subprocess, tarfile
raw = base64.b64decode('{b64}')
assert hashlib.sha256(raw).hexdigest() == '{sha}'
CODE, OUT = '/kaggle/working/code', '/kaggle/working/out'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(CODE)
print('code sha256 {sha} from git {rev}', flush=True)


def ck(job):
    hits = glob.glob(f'/kaggle/input/**/{{job}}/best_ema.pt', recursive=True)
    return hits[0] if hits else ''


def find(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True), key=len)
    return hits[0] if hits else None


chik = [p for p in glob.glob('/kaggle/input/**/*.mat', recursive=True) if 'chikusei' in p.lower() and 'Ground' not in p]
DATA = dict(chikusei=sorted(chik, key=len)[0] if chik else None, pavia=find('Pavia.mat'),
            cave=os.path.dirname(os.path.dirname(os.path.dirname(find('Train/HSI/balloons.mat') or '/x/x/x/x'))),
            harvard=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(find('Data/Train/HSI/img*.mat') or '/x/x/x/x/x')))))
args = f"--dataset {dataset} --scale {scale} --mat '{{DATA['{dataset}']}}' --ckpt {{ck('{full}')}} --out {{OUT}}/blind-{dataset}-x{scale}"
if ck('{zeros}'):
    args += f" --ckpt_zeros {{ck('{zeros}')}}"
if ck('{learned}'):
    args += f" --ckpt_learned {{ck('{learned}')}}"
print(args, flush=True)
r = subprocess.run(f'python blind_eval.py {{args}}', shell=True, cwd=CODE)
assert r.returncode == 0
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--source", required=True, help="session slug whose output holds the checkpoints")
    ap.add_argument("--user", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--zeros", default="none")
    ap.add_argument("--learned", default="none")
    a = ap.parse_args()
    b64, sha = code_blob()
    slug = f"puf-blind-{a.dataset}-x{a.scale}"
    out = os.path.join(HERE, "blind", slug)
    os.makedirs(out, exist_ok=True)
    cell = CELL.format(b64=b64, sha=sha, rev=git_rev(), dataset=a.dataset, scale=a.scale, full=a.full,
                       zeros=a.zeros, learned=a.learned)
    with open(os.path.join(out, f"{slug}.ipynb"), "w", newline="\n") as f:
        json.dump(notebook([cell]), f, indent=1)
    meta = dict(id=f"{a.user}/{slug}", title=slug, code_file=f"{slug}.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=False, enable_internet=False,
                dataset_sources=[DATASETS[a.dataset]], competition_sources=[],
                kernel_sources=[f"{a.user}/{a.source}"])
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote blind/{slug} for {meta['id']} (source {a.source})")


if __name__ == "__main__":
    main()
