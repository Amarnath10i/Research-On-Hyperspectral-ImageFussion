"""Write the Kaggle CPU notebook that builds the paper (tables, figures and PDF) from the committed result files.

    python build_paper_kernel.py              # the working tree: runs make_numbers/make_tables/make_figures, then Tectonic
    python build_paper_kernel.py --test       # the committed (HEAD) paper, Tectonic only: checks the toolchain
    kaggle kernels push -p kaggle_paper

The notebook embeds paper/puformer and the result JSONs (base64 tar with its sha256), takes the visual.npz files,
which are not in git, from the analysis kernels' outputs, downloads Tectonic and writes main.pdf, figures/ and
tables/ to /kaggle/working/paper.
"""
import argparse
import base64
import glob
import hashlib
import io
import json
import os
import subprocess
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PAPER = ["paper/puformer/" + f for f in ("main.tex", "references.bib", "numbers.tex", "make_numbers.py",
                                          "make_tables.py", "make_figures.py", "make_q1_tables.py",
                                          "make_q1_figures.py")]
PAPER += ["methods/puformer/hsifuse/" + f for f in sorted(os.listdir(os.path.join(REPO, "methods", "puformer", "hsifuse")))
          if f.endswith(".py")]
RESULTS = ["results/puformer_chikusei_x4/puformer/*.json", "results/puformer_chikusei_x4/ssrnet/*.json",
           "results/puformer_chikusei_x4/tip26_table4_comparison.csv", "results/puformer_chikusei_x4_v3/*/*.json",
           "results/puformer_pavia_x4/*.csv", "results/puformer_pavia_x4/*/*.json",
           "results/q1/*/results.json", "results/q1/*/gaps.json", "results/q1/*/blind.json",
           "results/q1/complexity.json"]
# datasets (test ground truth for the visual comparison) and the uploaded test predictions of the benchmark jobs
Q1_DATA = ["mingliu123/chikusei", "mlxlx0000/paviadata", "liptee/hyperspectral-image-restoration-based-on-cave",
           "nikeshreddypatlolla/harvard-hsi-2", "amarnath10chinu/puformer-q1-preds"]
SOURCES = ["amarnath10chinu/puformer-chikusei-paper-analysis", "amarnath10chinu/puformer-pavia-analysis"]

CELL = r'''import base64, glob, hashlib, io, os, shutil, subprocess, tarfile
B64 = '{b64}'
raw = base64.b64decode(B64)
assert hashlib.sha256(raw).hexdigest() == '{sha}'
REPO, OUT = '/kaggle/working/repo', '/kaggle/working/paper'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(REPO)
print('paper sources sha256 {sha} from git {rev}')
P = REPO + '/paper/puformer'


def sh(cmd, cwd=P):
    r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print('\n'.join(l for l in r.stdout.splitlines() if 'Underfull' not in l)[-6000:], flush=True)
    assert r.returncode == 0, cmd


# .npz files are not in git: take them from the analysis kernels' outputs
for src in glob.glob('/kaggle/input/**/visual.npz', recursive=True) + glob.glob('/kaggle/input/**/sam_maps.npz', recursive=True):
    ds = 'puformer_pavia_x4' if 'pavia' in src.lower() else 'puformer_chikusei_x4_v3'
    sub = 'sam_analysis' if src.endswith('sam_maps.npz') else 'paper_analysis'
    dst = f'{{REPO}}/results/{{ds}}/{{sub}}/' + os.path.basename(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print('npz:', src, '->', dst)
if {generate}:
    sh('python make_numbers.py && python make_tables.py && python make_figures.py && python make_q1_tables.py')
    preds = sorted(glob.glob('/kaggle/input/**/bench-*/test_pred_f16.npy', recursive=True))
    pdir = os.path.dirname(os.path.dirname(preds[0])) if preds else ''
    sh(f'Q1_PREDS={{pdir}} python make_q1_figures.py')
os.makedirs('/kaggle/working/bin', exist_ok=True)
# the statically linked (musl) build: the glibc build needs a newer libc than Kaggle's image has
for v in ['0.15.0', '0.14.1']:
    url = (f'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{{v}}/'
           f'tectonic-{{v}}-x86_64-unknown-linux-musl.tar.gz')
    if subprocess.run(f'curl -fsSL {{url}} | tar xz', shell=True, cwd='/kaggle/working/bin').returncode == 0:
        break
sh('./tectonic --version', cwd='/kaggle/working/bin')
sh('/kaggle/working/bin/tectonic -X compile main.tex')
os.makedirs(OUT, exist_ok=True)
for f in ['main.pdf', 'numbers.tex']:
    shutil.copyfile(f'{{P}}/{{f}}', f'{{OUT}}/{{f}}')
for d in ['figures', 'tables']:
    if os.path.isdir(f'{{P}}/{{d}}'):
        shutil.copytree(f'{{P}}/{{d}}', f'{{OUT}}/{{d}}', dirs_exist_ok=True)
sh('ls -laR ' + OUT, cwd=OUT)
'''


def git_rev():
    rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "paper", "results"], cwd=REPO, text=True).strip()
    return rev + ("+local changes" if dirty else "")


def blob(test):
    """tar.gz of the paper sources and result files, working tree or (test) HEAD."""
    if test:
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD", "paper/puformer"],
                                        cwd=REPO, text=True).split()
        names = [n for n in names if n.endswith((".tex", ".bib")) or n.startswith("paper/puformer/figures/")]
        read = lambda n: subprocess.check_output(["git", "show", f"HEAD:{n}"], cwd=REPO)
    else:
        names = PAPER + sorted({os.path.relpath(p, REPO).replace(os.sep, "/")
                                for pat in RESULTS for p in glob.glob(os.path.join(REPO, pat))})
        read = lambda n: open(os.path.join(REPO, n), "rb").read()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for n in names:
            data = read(n)
            data = data if n.endswith(".pdf") else data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(n)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest(), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="compile the committed paper only (toolchain check)")
    ap.add_argument("--slug", default="puformer-paper-build")
    ap.add_argument("--user", default="amarnath10chinu")
    ap.add_argument("--q1data", action="store_true", help="attach the datasets and the prediction dataset")
    a = ap.parse_args()
    b64, sha, names = blob(a.test)
    out = os.path.join(HERE, "kaggle_paper")
    os.makedirs(out, exist_ok=True)
    src = CELL.format(b64=b64, sha=sha, rev="HEAD" if a.test else git_rev(), generate=not a.test)
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id="c0", source=src)])
    name = f"{a.slug}.ipynb"
    with open(os.path.join(out, name), "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
    meta = dict(id=f"{a.user}/{a.slug}", title="PUFormer paper build", code_file=name, language="python",
                kernel_type="notebook", is_private=True, enable_gpu=False, enable_internet=True,
                dataset_sources=Q1_DATA if a.q1data else [], competition_sources=[],
                kernel_sources=[] if a.test else SOURCES)
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote kaggle_paper/{name}: {len(names)} files, {len(b64) / 1e6:.1f} MB base64, sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
