"""Generate the self-contained KrylovNet Kaggle notebook.

Embeds both packages with %%writefile so the notebook needs no clone and no
internet, and cannot drift from the source: regenerate after any code change.

    python tools/build_krylovnet_notebook.py

Pins torch 2.5.1, which still ships sm_60 kernels. Kaggle's push API resolves
enable_gpu to its default GPU (a P100) and silently overrides an accelerator
chosen in the UI, so pinning a torch that runs on Pascal is what makes an
unattended push work at all - torch >= 2.6 dropped Pascal and fails on the
first CUDA op.
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAETF = os.path.join(REPO, "proposal1", "daetf")
KRY = os.path.join(REPO, "proposal2", "krylovnet")
OUT = os.path.join(REPO, "proposal2", "notebooks", "KrylovNet_Kaggle.ipynb")

# only what KrylovNet actually imports, plus their own dependencies
DAETF_MODULES = ["io_utils", "config", "degrade", "metrics", "nullspace",
                 "spectral_embed", "modules", "model", "losses", "data",
                 "engine", "baselines", "experiments", "selfcheck", "__init__"]
KRY_MODULES = ["config", "solver", "model", "losses", "engine", "experiments",
               "selfcheck", "__init__"]


def _lines(text: str) -> list:
    """nbformat keeps the newline on each source line; splitting without
    keepends collapses the whole cell onto one line."""
    return text.splitlines(keepends=True)


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src.strip())}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(src.strip("\n"))}


def _check_complete() -> None:
    import glob
    for pkg, listed in ((DAETF, DAETF_MODULES), (KRY, KRY_MODULES)):
        on_disk = {os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(pkg, "*.py"))}
        missing = on_disk - set(listed)
        if missing:
            raise SystemExit(
                f"build_krylovnet_notebook: {sorted(missing)} exist in {pkg} "
                f"but are not listed, so the notebook would not write them and "
                f"would fail at import time on Kaggle.")


def build() -> dict:
    cells = []

    cells.append(md("""
# KrylovNet - unrolled Krylov fusion with a learned proximal prior

Proposal 2. The fusion problem is posed as the normal equation of the two
observation models,

    A x = b,   A = D^T D + S^T S + rho I,   b = D^T X + S^T M,

and solved by an unrolled GMRES-style Krylov solver with a learned spectral
graph preconditioner.

**What changed for this run.** The solver alone is Tikhonov least squares and
carries no image prior, so it returns the smooth minimum-norm member of the
solution set. Measured, it scored 29.49 dB against plain bicubic at 31.31 dB -
worse than doing nothing, with 2287 trainable parameters. A learned proximal
denoiser is now interleaved between data steps (plug-and-play / half-quadratic
splitting): the data step enforces agreement with the observations, the prior
supplies what they underdetermine. That is 1.38 M parameters, and in a
single-patch overfit test - budget removed as a confound - it beats bicubic by
5.9 dB where the solver alone was 1.8 dB behind it.

**Protocol.** The MSI is simulated with the **Nikon D700** response, which is
what the published CAVE/Harvard numbers use. The three-Gaussian response this
repository used previously is a better-conditioned mixing matrix (cond 1.42 vs
1.86), i.e. an easier problem, so numbers obtained under it are not comparable
to published ones.
""".strip()))

    cells.append(md("## 1. Environment"))
    cells.append(code("""
# torch 2.5.1 still ships sm_60 kernels, so this runs on a P100. Kaggle's push
# API resolves enable_gpu to its default GPU and overrides an accelerator set
# in the UI, and torch >= 2.6 dropped Pascal entirely - so pinning here is what
# makes an unattended push work.
!pip install -q torch==2.5.1 torchvision==0.20.1 2>/dev/null || echo "pin skipped"
"""))
    cells.append(code("""
import os, sys, json, time, math, warnings
warnings.filterwarnings('ignore')
import torch, numpy as np

print('torch  ', torch.__version__)
print('cuda   ', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    arch = f'sm_{p.major}{p.minor}'
    built = list(torch.cuda.get_arch_list())
    print('gpu    ', p.name, f'{p.total_memory / 2**30:.1f} GB', arch)
    print('built  ', ' '.join(built))
    if arch not in built:
        raise RuntimeError(
            f'This torch build has no kernels for {arch}. Either the pin above '
            f'failed or the accelerator changed; set Settings -> Accelerator -> '
            f'"GPU T4 x2" and re-run.')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
os.chdir(WORK); print('workdir', os.getcwd())
"""))

    cells.append(md("""
## 2. Packages

Written out module by module so the notebook is self-contained. Generated from
`proposal1/daetf/` and `proposal2/krylovnet/` by
`tools/build_krylovnet_notebook.py` - edit the packages and regenerate rather
than editing these cells.
"""))
    cells.append(code(
        "import os\n"
        "for d in ('proposal1', 'proposal1/daetf', 'proposal2', 'proposal2/krylovnet'):\n"
        "    os.makedirs(d, exist_ok=True)\n"
        "for d in ('proposal1', 'proposal2'):\n"
        "    open(os.path.join(d, '__init__.py'), 'w').close()\n"
        "print('package dirs ready')"))

    for mod in DAETF_MODULES:
        with open(os.path.join(DAETF, f"{mod}.py"), "r", encoding="utf-8") as f:
            cells.append(code(f"%%writefile proposal1/daetf/{mod}.py\n{f.read()}"))
    for mod in KRY_MODULES:
        with open(os.path.join(KRY, f"{mod}.py"), "r", encoding="utf-8") as f:
            cells.append(code(f"%%writefile proposal2/krylovnet/{mod}.py\n{f.read()}"))

    cells.append(code("""
import sys
for m in [k for k in list(sys.modules) if k.startswith(('proposal1', 'proposal2'))]:
    del sys.modules[m]
sys.path.insert(0, os.getcwd())
import proposal2.krylovnet as K
print('krylovnet loaded')
"""))

    cells.append(md("""
## 3. Self-checks

Adjointness of D and S, the exact solve, and the preconditioner - verified
before any training time is spent.
"""))
    cells.append(code("K.selfcheck(device='cpu')"))

    cells.append(md("""
## 4. Configuration

`ITERS` is the one knob to set from the measured rate. Run with `QUICK = True`
first: it prints the achieved it/s and the iteration count that fills the
session, then set `ITERS` and re-run.
"""))
    cells.append(code("""
QUICK = True          # short validation run; prints the rate to size the real one
SESSION_HOURS = 8.0   # Kaggle GPU sessions cap at 9h; leave headroom for eval

cfg = K.Config(
    scale=4, patch=96, batch=12,
    iters=200 if QUICK else 100000,
    warmup=20 if QUICK else 2000,
    val_every=200 if QUICK else 5000,
    log_every=50 if QUICK else 250,
    use_prior=True, prior_width=96, prior_blocks=8, n_outer=4,
    w_recon=1.0,
    out_dir=os.path.join(os.getcwd(), 'krylov_out'),
)
cfg.resolve()
print(json.dumps({k: v for k, v in cfg.to_dict().items()
                  if k in ('source_root','target_root','bands','msi_bands',
                           'scale','patch','batch','iters','use_prior',
                           'prior_width','prior_blocks','n_outer','w_recon')},
                 indent=1))
"""))

    cells.append(md("## 5. Train"))
    cells.append(code("""
t0 = time.time()
result = K.train(cfg, device=DEVICE)
elapsed = time.time() - t0
rate = cfg.iters / max(elapsed, 1e-9)
print(f'\\ntrained {cfg.iters} iters in {elapsed/60:.1f} min  ({rate:.2f} it/s)')
print(f'best: {result["best"] if isinstance(result, dict) and "best" in result else result}')

if QUICK:
    fits = int(rate * SESSION_HOURS * 3600 * 0.85)   # 15% headroom for eval
    print(f'\\n>>> at {rate:.2f} it/s, {SESSION_HOURS}h fits about {fits:,} iterations')
    print(f'>>> set QUICK = False and iters = {fits // 1000 * 1000:,}, then re-run')
"""))

    cells.append(md("""
## 6. Evaluation against the same-protocol baselines

Bicubic, GSA and a coupled subspace estimator need no checkpoints, so they run
through the identical pipeline - same degradation, same scale factor, same
scenes, same metric code. They are the only strictly comparable rows available
and the floor a learned method has to clear.
"""))
    cells.append(code("""
from proposal1.daetf.engine import evaluate_dataset as shared_eval
from proposal1.daetf.data import estimate_srf
from proposal1.daetf.baselines import evaluate_all_baselines

model = result['model'] if isinstance(result, dict) and 'model' in result else result
srf = estimate_srf(cfg.source_root, 'Train', cfg)

print('=== classical baselines (same protocol) ===')
base = evaluate_all_baselines(cfg.source_root, cfg, srf, 'Test', DEVICE, verbose=False)
for name, r in base.items():
    m = r['mean']
    print(f"  {name:<14} PSNR={m['psnr']:7.3f}  SSIM={m['ssim']:.4f}  "
          f"SAM={m['sam']:6.3f}  ERGAS={m['ergas']:8.3f}")

print('\\n=== KrylovNet ===')
ours = shared_eval(model, cfg.source_root, cfg, 'Test', DEVICE, verbose=True)
"""))

    cells.append(md("""
## 7. Against the published numbers

Read the protocol column. These are the authors' reported values; ours are
computed here under the Nikon D700 response at x4, which is the protocol they
used. Anything obtained under the old three-Gaussian response is **not**
comparable and is excluded.
"""))
    cells.append(code("""
PUBLISHED_CAVE_X4 = {
    'FeINFN (2024)':   {'psnr': 52.47, 'ssim': 0.998, 'sam': 1.91, 'ergas': 0.98},
    'BDT (2023)':      {'psnr': 52.30, 'ssim': 0.997, 'sam': 1.93, 'ergas': 1.02},
    '3DT-Net (2023)':  {'psnr': 51.38, 'ssim': 0.996, 'sam': 2.16, 'ergas': 1.14},
    'DSPNet (2023)':   {'psnr': 51.18, 'ssim': 0.997, 'sam': 2.15, 'ergas': 1.13},
    'DHIF (2022)':     {'psnr': 51.07, 'ssim': 0.997, 'sam': 2.01, 'ergas': 1.22},
    'MIMO-SST (2022)': {'psnr': 50.98, 'ssim': 0.997, 'sam': 2.23, 'ergas': 1.18},
    'CoFusion (2026)': {'psnr': 50.67, 'ssim': 0.997, 'sam': 2.15, 'ergas': 1.73},
    'PSRT (2023)':     {'psnr': 50.47, 'ssim': 0.996, 'sam': 2.19, 'ergas': 2.06},
    'SSA (2026)':      {'psnr': 45.92, 'ssim': 0.996, 'sam': 2.02, 'ergas': 1.07},
    'Fusformer (2022)':{'psnr': 44.52, 'ssim': 0.983, 'sam': 4.12, 'ergas': 1.06},
}
rows = dict(PUBLISHED_CAVE_X4)
rows.update({f'{n} (same protocol)': r['mean'] for n, r in base.items()})
rows['KrylovNet (ours)'] = ours

print(f"{'Method':<28}{'PSNR':>8}{'SSIM':>9}{'SAM':>8}{'ERGAS':>9}")
print('-' * 62)
for n, m in sorted(rows.items(), key=lambda kv: -kv[1]['psnr']):
    print(f"{n:<28}{m['psnr']:8.2f}{m.get('ssim', float('nan')):9.4f}"
          f"{m['sam']:8.2f}{m['ergas']:9.2f}")

best_pub = max(v['psnr'] for v in PUBLISHED_CAVE_X4.values())
gap = ours['psnr'] - best_pub
print(f"\\nvs best published ({best_pub:.2f} dB): {gap:+.2f} dB")
print('BEATS SOTA' if gap > 0 else f'still {abs(gap):.2f} dB behind')
"""))

    cells.append(md("## 8. Save"))
    cells.append(code("""
payload = {'config': cfg.to_dict(), 'ours': ours,
           'baselines': {n: r['mean'] for n, r in base.items()},
           'published_cave_x4': PUBLISHED_CAVE_X4,
           'protocol': 'Nikon D700 SRF, Wald x4, data_range=1.0'}
with open('krylovnet_results.json', 'w') as f:
    json.dump(payload, f, indent=1, default=float)
print('written:', [f for f in os.listdir('.') if f.endswith(('.json', '.pt', '.pth'))])
print('checkpoints:', os.listdir(cfg.out_dir) if os.path.isdir(cfg.out_dir) else [])
"""))

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    _check_complete()
    nb = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {OUT}")
    print(f"  {len(nb['cells'])} cells ({n_code} code), "
          f"{os.path.getsize(OUT) / 1024:.0f} KB")
