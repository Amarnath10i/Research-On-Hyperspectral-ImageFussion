"""Write a Kaggle kernel that runs cond_eval.py (or --script neon_fullres.py) on checkpoints from earlier sessions
of the same account.

    python build_cond_eval.py --user tummapudinavadeep --sources puf-r01,puf-r02,puf-s06 \
        --jobs cond-chikusei-x4-puformer,cond-chikusei-x4-nocond,bench-chikusei-x4-puformer [--slug puf-cond-eval-1]
    kaggle kernels push -p condeval/<slug>

CPU by default (no GPU quota; --gpu for neon_fullres.py). Every job's model flags come from jobs.py; its checkpoint is <job>/best_ema.pt in the
attached session outputs. Writes /kaggle/working/out/<job>/cond_eval.json (neon_fullres.json).
"""
import argparse
import json
import os

from build_jobs import DATASETS, notebook
from build_notebook import HERE, code_blob, git_rev
from jobs import all_jobs

CELL = r'''import base64, glob, hashlib, io, os, shlex, subprocess, tarfile
raw = base64.b64decode('{b64}')
assert hashlib.sha256(raw).hexdigest() == '{sha}'
CODE, OUT = '/kaggle/working/code', '/kaggle/working/out'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(CODE)
print('code sha256 {sha} from git {rev}', flush=True)
subprocess.run('nproc; free -g | head -2', shell=True)


def find(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True), key=len)
    return hits[0] if hits else None


chik = [p for p in glob.glob('/kaggle/input/**/*.mat', recursive=True) if 'chikusei' in p.lower() and 'Ground' not in p]
DATA = dict(chikusei=sorted(chik, key=len)[0] if chik else None, pavia=find('Pavia.mat'),
            cave=os.path.dirname(os.path.dirname(os.path.dirname(find('Train/HSI/balloons.mat') or '/x/x/x/x'))),
            harvard=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(find('Data/Train/HSI/img*.mat') or '/x/x/x/x/x')))),
            neon='/kaggle/input')
CACHE = dict(chikusei='/tmp/chikusei.npy', harvard='/tmp/harvard')
JOBS = {jobs}
for name, rargs, ds in JOBS:
    ck = find(f'{{name}}/best_ema.pt')
    if not ck:
        print('no checkpoint for', name, flush=True); continue
    cache = f'--cache {{CACHE[ds]}}' if ds in CACHE else ''
    out = f'{{OUT}}/{{name}}'
    cmd = f"python {script} --mat {{shlex.quote(DATA[ds])}} {{cache}} {{rargs}} --ckpt {{ck}} --out {{out}}"
    print('RUN', name, ':', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=CODE)
    print('END', name, 'rc', r.returncode, flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--sources", required=True, help="comma-separated session slugs of the same account")
    ap.add_argument("--jobs", required=True, help="comma-separated job names (jobs.py)")
    ap.add_argument("--slug", default="puf-cond-eval")
    ap.add_argument("--script", default="cond_eval.py", choices=["cond_eval.py", "neon_fullres.py"])
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    J = all_jobs()
    jobs = []
    for n in a.jobs.split(","):
        j = J[n]
        rargs = j["rargs"].replace("--wide 1", "").strip()
        jobs.append((n, rargs, j["dataset"]))
    b64, sha = code_blob()
    out = os.path.join(HERE, "condeval", a.slug)
    os.makedirs(out, exist_ok=True)
    cell = CELL.format(b64=b64, sha=sha, rev=git_rev(), jobs=repr(jobs), script=a.script)
    with open(os.path.join(out, f"{a.slug}.ipynb"), "w", newline="\n") as f:
        json.dump(notebook([cell]), f, indent=1)
    used = sorted({d for _, _, d in jobs})
    srcs = [f"{a.user}/{s}" for s in a.sources.split(",")]
    if "neon" in used:
        srcs += [f"{a.user}/puf-neon-data", f"{a.user}/puf-neon-fit"]
    meta = dict(id=f"{a.user}/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=a.gpu, enable_internet=False,
                dataset_sources=[DATASETS[d] for d in used if DATASETS[d]], competition_sources=[],
                kernel_sources=srcs)
    if a.gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote condeval/{a.slug} for {meta['id']}: {[n for n, _, _ in jobs]} from {srcs}")


if __name__ == "__main__":
    main()
