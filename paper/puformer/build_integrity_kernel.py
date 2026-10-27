"""Write the Kaggle notebook that screens the compiled paper for text similarity and AI-generated text.

    python build_integrity_kernel.py
    kaggle kernels push -p kaggle_integrity

The notebook takes main.pdf and supplementary.pdf from the output of the paper-build kernel, embeds
integrity_check.py, the bibliography and the project's public README files (base64 tar with its sha256), and writes
report.md and the raw scores to /kaggle/working/integrity. It needs internet (sources, models) and 2 x T4.
"""
import base64
import hashlib
import io
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FILES = ["paper/puformer/integrity_check.py", "paper/puformer/references.bib", "README.md",
         "paper/puformer/README.md", "docs/THEORY_AND_RESULTS.md"]
OWN = ["README.md", "paper/puformer/README.md", "docs/THEORY_AND_RESULTS.md"]

CELL = r'''import base64, glob, hashlib, io, subprocess, tarfile
raw = base64.b64decode('{b64}')
assert hashlib.sha256(raw).hexdigest() == '{sha}'
R = '/kaggle/working/ic'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(R)
subprocess.run('pip install -q pymupdf', shell=True, check=True)
pdfs = [sorted(glob.glob(f'/kaggle/input/**/paper/{{n}}.pdf', recursive=True))[0] for n in {names}]
print(pdfs, flush=True)
own = ' '.join(f'{{R}}/{{p}}' for p in {own})
r = subprocess.run(f'python {{R}}/paper/puformer/integrity_check.py --pdf {{" ".join(pdfs)}} '
                   f'--bib {{R}}/paper/puformer/references.bib --own {{own}} --out /kaggle/working/integrity{extra}', shell=True)
assert r.returncode == 0
'''


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", default="main,supplementary", help="comma-separated PDF names from the paper build")
    ap.add_argument("--no-ai", action="store_true", help="similarity only (CPU kernel, no AI-text detectors)")
    a = ap.parse_args()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for n in FILES:
            data = open(os.path.join(REPO, n), "rb").read().replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(n)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    b64, sha = base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()
    out = os.path.join(HERE, "kaggle_integrity")
    os.makedirs(out, exist_ok=True)
    name = "puformer-integrity-check.ipynb"
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id="c0",
                          source=CELL.format(b64=b64, sha=sha, own=repr(OWN), names=repr(tuple(a.pdfs.split(","))),
                                            extra=" --no-ai" if a.no_ai else ""))])
    with open(os.path.join(out, name), "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
    meta = dict(id="amarnath10chinu/puformer-integrity-check", title="PUFormer integrity check", code_file=name,
                language="python", kernel_type="notebook", is_private=True, enable_gpu=not a.no_ai, enable_internet=True,
                dataset_sources=[], competition_sources=[],
                kernel_sources=["amarnath10chinu/puformer-paper-build"])
    if not a.no_ai:
        meta["machine_shape"] = "NvidiaTeslaT4"
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote kaggle_integrity/{name}: {len(FILES)} files, sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
