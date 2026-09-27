"""Write a CPU Kaggle notebook that computes analysis data for the paper from a finished run.

    python build_analysis_notebook.py --source amarnath10chinu/puformer-chikusei-x4-v3
    python build_analysis_notebook.py --dataset pavia --source amarnath10chinu/puformer-pavia-x4 \
        --scripts sam_analysis --slug puformer-pavia-sam-analysis --folder analysis_pavia
    kaggle kernels push -p analysis
It mounts the training kernel's output (test predictions) and runs paper_analysis.py and/or sam_analysis.py.
"""
import argparse
import json
import os

from build_notebook import HERE, code_blob, git_rev

CELL = r'''import base64, glob, hashlib, io, os, shlex, subprocess, tarfile, time
B64 = '{b64}'
raw = base64.b64decode(B64)
assert hashlib.sha256(raw).hexdigest() == '{sha}'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall('/kaggle/working/code')
print('code sha256 {sha} from git {rev}')
subprocess.run('nproc; free -g | head -2; find /kaggle/input -maxdepth 7 -name "*.npy" | head', shell=True)
MAT = {mat}
for script in {scripts}:
    p = subprocess.Popen(f'python {{script}}.py --dataset {dataset} --mat {{MAT}} --cache /tmp/crop.npy '
                         f'--pred_dir /kaggle/input --out /kaggle/working/{{script}}', shell=True,
                         cwd='/kaggle/working/code', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
    assert p.wait() == 0, script + ' failed'
subprocess.run('ls -la /kaggle/working/*', shell=True)
'''
DATA = {"chikusei": dict(mat="'/kaggle/input'", source="mingliu123/chikusei"),
        "pavia": dict(mat="shlex.quote(sorted(glob.glob('/kaggle/input/**/Pavia.mat', recursive=True))[0])",
                      source="mlxlx0000/paviadata")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chikusei", choices=list(DATA))
    ap.add_argument("--source", default="amarnath10chinu/puformer-chikusei-x4-v3", help="training kernel with the predictions")
    ap.add_argument("--scripts", default="paper_analysis", help="comma-separated: paper_analysis, sam_analysis")
    ap.add_argument("--slug", default="puformer-chikusei-paper-analysis")
    ap.add_argument("--folder", default="analysis")
    ap.add_argument("--user", default="amarnath10chinu")
    a = ap.parse_args()
    b64, sha = code_blob()
    out = os.path.join(HERE, a.folder)
    os.makedirs(out, exist_ok=True)
    src = CELL.format(b64=b64, sha=sha, rev=git_rev(), mat=DATA[a.dataset]["mat"], dataset=a.dataset,
                      scripts=repr(a.scripts.split(",")))
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id="c0", source=src)])
    name = f"{a.slug}.ipynb"
    with open(os.path.join(out, name), "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
    title = f"PUFormer {a.dataset.capitalize()} " + ("paper analysis" if a.scripts == "paper_analysis" else "analysis")
    meta = dict(id=f"{a.user}/{a.slug}", title=title, code_file=name, language="python",
                kernel_type="notebook", is_private=True, enable_gpu=False, enable_internet=False,
                dataset_sources=[DATA[a.dataset]["source"]], competition_sources=[], kernel_sources=[a.source])
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {a.folder}/{name} (sha256 {sha[:12]}) for {meta['id']}, source {a.source}")


if __name__ == "__main__":
    main()
