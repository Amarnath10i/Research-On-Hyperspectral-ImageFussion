"""Write a Kaggle kernel that runs shell commands on the embedded code (checks and analyses).

    python build_script.py --user amarnath10chinu --slug puf-kron-check --gpu --run "python kron_check.py"
    python build_script.py --user amarnath10chinu --slug puf-null --data chikusei,pavia \
        --extra amarnath10chinu/puformer-q1-preds \
        --run "python null_analysis.py --dataset pavia --mat {pavia} --out {OUT}/null-pavia"
    kaggle kernels push -p scripts/<slug>

Placeholders in --run: {chikusei}, {pavia}, {cave}, {harvard} (data paths), {cache_chikusei}, {OUT}.
Outputs go to /kaggle/working/out; the log of every command is printed.
"""
import argparse
import json
import os

from build_jobs import DATASETS, notebook
from build_notebook import HERE, code_blob, git_rev

CELL = r'''import base64, glob, hashlib, io, os, shlex, subprocess, tarfile
raw = base64.b64decode('{b64}')
assert hashlib.sha256(raw).hexdigest() == '{sha}'
CODE, OUT = '/kaggle/working/code', '/kaggle/working/out'
os.makedirs(OUT, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(CODE)
print('code sha256 {sha} from git {rev}', flush=True)


def find(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True), key=len)
    return hits[0] if hits else ''


chik = [p for p in glob.glob('/kaggle/input/**/*.mat', recursive=True) if 'chikusei' in p.lower() and 'Ground' not in p]
P = dict(chikusei=sorted(chik, key=len)[0] if chik else '', pavia=find('Pavia.mat'),
         cave=os.path.dirname(os.path.dirname(os.path.dirname(find('Train/HSI/balloons.mat') or '/x/x/x/x'))),
         harvard=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(find('Data/Train/HSI/img*.mat') or '/x/x/x/x/x')))),
         cache_chikusei='/tmp/chikusei.npy', OUT=OUT)
P = {{k: shlex.quote(v) for k, v in P.items()}}
for cmd in {cmds}:
    cmd = cmd.format(**P)
    print('RUN', cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=CODE)
    print('END rc', r.returncode, flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--data", default="", help="comma-separated dataset keys of build_jobs.DATASETS")
    ap.add_argument("--extra", default="", help="comma-separated extra Kaggle datasets <owner>/<slug>")
    ap.add_argument("--sources", default="", help="comma-separated kernel sources <owner>/<slug>")
    ap.add_argument("--run", action="append", required=True)
    a = ap.parse_args()
    b64, sha = code_blob()
    out = os.path.join(HERE, "scripts", a.slug)
    os.makedirs(out, exist_ok=True)
    cell = CELL.format(b64=b64, sha=sha, rev=git_rev(), cmds=repr(a.run))
    with open(os.path.join(out, f"{a.slug}.ipynb"), "w", newline="\n") as f:
        json.dump(notebook([cell]), f, indent=1)
    data = [DATASETS[d] for d in a.data.split(",") if d and DATASETS.get(d)]
    data += [d for d in a.extra.split(",") if d]
    meta = dict(id=f"{a.user}/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=a.gpu, enable_internet=False,
                dataset_sources=data, competition_sources=[],
                kernel_sources=[s for s in a.sources.split(",") if s])
    if a.gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote scripts/{a.slug} for {meta['id']}")


if __name__ == "__main__":
    main()
