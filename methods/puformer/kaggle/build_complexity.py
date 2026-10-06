"""Write a short GPU Kaggle kernel that runs complexity.py (parameters, FLOPs, T4 time and memory of every method).

    python build_complexity.py --user amarnath10chinu && kaggle kernels push -p complexity
"""
import argparse
import json
import os

from build_jobs import notebook
from build_notebook import HERE, code_blob, git_rev

CELL = r'''import base64, hashlib, io, os, subprocess, sys, tarfile
raw = base64.b64decode('{b64}')
assert hashlib.sha256(raw).hexdigest() == '{sha}'
CODE, OUT = '/kaggle/working/code', '/kaggle/working/out'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(CODE)
os.makedirs(OUT, exist_ok=True)
print('code sha256 {sha} from git {rev}', flush=True)
env = dict(os.environ, PUF_EXT='/kaggle/working/ext')
r = subprocess.run(f'python complexity.py --out {{OUT}}/complexity.json', shell=True, cwd=CODE, env=env)
assert r.returncode == 0
subprocess.run('rm -rf /kaggle/working/ext', shell=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    a = ap.parse_args()
    b64, sha = code_blob()
    slug = "puf-complexity"
    out = os.path.join(HERE, "complexity")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{slug}.ipynb"), "w", newline="\n") as f:
        json.dump(notebook([CELL.format(b64=b64, sha=sha, rev=git_rev())]), f, indent=1)
    meta = dict(id=f"{a.user}/{slug}", title=slug, code_file=f"{slug}.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=True,
                machine_shape="NvidiaTeslaT4", dataset_sources=[], competition_sources=[], kernel_sources=[])
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote complexity/{slug} for {meta['id']}")


if __name__ == "__main__":
    main()
