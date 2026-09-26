"""Write the CPU Kaggle notebook that computes the paper's figures data from a finished run.

    python build_analysis_notebook.py --source amarnath10chinu/puformer-chikusei-x4-v3
    kaggle kernels push -p analysis
It mounts the training kernel's output (test predictions) and runs paper_analysis.py.
"""
import argparse
import json
import os

from build_notebook import HERE, code_blob, git_rev

CELL = r'''import base64, hashlib, io, os, subprocess, tarfile, time
B64 = '{b64}'
raw = base64.b64decode(B64)
assert hashlib.sha256(raw).hexdigest() == '{sha}'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall('/kaggle/working/code')
print('code sha256 {sha} from git {rev}')
subprocess.run('nproc; free -g | head -2; find /kaggle/input -maxdepth 6 -name "*.npy" | head', shell=True)
p = subprocess.Popen('python paper_analysis.py --mat /kaggle/input --cache /tmp/crop.npy --pred_dir /kaggle/input '
                     '--out /kaggle/working/paper', shell=True, cwd='/kaggle/working/code',
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end='', flush=True)
assert p.wait() == 0, 'analysis failed'
subprocess.run('ls -la /kaggle/working/paper', shell=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="amarnath10chinu/puformer-chikusei-x4-v3", help="training kernel with the predictions")
    ap.add_argument("--slug", default="puformer-chikusei-paper-analysis")
    ap.add_argument("--user", default="amarnath10chinu")
    a = ap.parse_args()
    b64, sha = code_blob()
    out = os.path.join(HERE, "analysis")
    os.makedirs(out, exist_ok=True)
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id="c0",
                          source=CELL.format(b64=b64, sha=sha, rev=git_rev()))])
    name = f"{a.slug}.ipynb"
    with open(os.path.join(out, name), "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
    meta = dict(id=f"{a.user}/{a.slug}", title="PUFormer Chikusei paper analysis", code_file=name, language="python",
                kernel_type="notebook", is_private=True, enable_gpu=False, enable_internet=False,
                dataset_sources=["mingliu123/chikusei"], competition_sources=[], kernel_sources=[a.source])
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote analysis/{name} (sha256 {sha[:12]}) for {meta['id']}, source {a.source}")


if __name__ == "__main__":
    main()
