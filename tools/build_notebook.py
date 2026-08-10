"""Generate the self-contained Kaggle notebook from the daetf package.

The notebook embeds the package source with %%writefile cells rather than
cloning the repository, so it runs on Kaggle with no network access to GitHub
and never drifts from the package: regenerate after any code change.

    python tools/build_notebook.py
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, "proposal1", "daetf")
OUT = os.path.join(REPO, "proposal1", "notebooks", "DAETF_Net_Kaggle_GPU.ipynb")

# order matters only for readability; Python resolves imports at import time
MODULES = ["io_utils", "config", "degrade", "metrics", "modules", "model",
           "losses", "data", "engine", "experiments", "selfcheck", "__init__"]


def _lines(text: str) -> list:
    """nbformat stores source as a list of lines that KEEP their newline.
    Splitting without keepends collapses the whole cell onto one line."""
    return text.splitlines(keepends=True)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source.strip())}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(source.strip("\n"))}


def build() -> dict:
    cells = []

    cells.append(md("""
# DAETF-Net - Domain-Adaptive Equivariant Tensor Fusion Network

Hyperspectral (HSI) and multispectral (MSI) image fusion built to survive the
transfer from the dataset it was trained on to one it has never seen.

**The problem this attacks.** Re-running ten published fusion methods under one
protocol shows spatial quality holds up across datasets while *spectral* quality
collapses: SAM is 2-7 deg in-domain and 8-36 deg cross-domain, and ERGAS rises by
one to two orders of magnitude. PSNR hides this - several baselines score
*higher* PSNR on the harder dataset - which is why the headline comparison in the
earlier benchmark was misleading.

**What is new here**

| Component | Mechanism | Verified by |
|---|---|---|
| EFE | p4 group-equivariant convolutions (lifting + group conv + group pool) | `check_equivariance` ~1e-6 |
| TSSE | Tucker contraction against a learned core tensor | `check_core_used` (gradient reaches the core) |
| AF-MoE | per-pixel top-k expert routing + load balancing | usage histogram |
| FDRM | orthonormal Haar DWT, learnable per-subband shrinkage, exact IDWT | `check_wavelet` ~1e-7 |
| DAE | degradation code conditioning every block through FiLM | auxiliary regression head |
| BPU | back-projection upsampler replacing bicubic | ablation |
| SPC loss | Charbonnier + SAM + gradient + SSIM + two physics terms | ablation |

**The central idea.** The two physics terms, `||Down(Y) - LR||` and
`||SRF(Y) - MSI||`, need no ground truth. They can be evaluated on any scene
from any sensor, so the same objective that trains the model also *adapts* it at
test time on a dataset with no labels.

**Hardware.** Fits a single 16 GB GPU. Choose **GPU T4 x2** as the accelerator:
PyTorch >= 2.6 ships no kernels for the P100's `sm_60`, so a P100 raises
"no kernel image is available" on the first CUDA op regardless of the code. The
environment cell below checks this and tells you if the accelerator is wrong.

Set `QUICK` below to smoke-test the whole pipeline in a few minutes before
committing to a full run.
""".strip()))

    # ---------------------------------------------------------------- setup
    cells.append(md("## 1. Environment"))
    cells.append(code("""
import os, sys, json, time, math, warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch, numpy as np
print('python  ', sys.version.split()[0])
print('torch   ', torch.__version__)
print('numpy   ', np.__version__)
print('cuda    ', torch.cuda.is_available())

GPU_OK = False
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    arch = f'sm_{p.major}{p.minor}'
    built = list(torch.cuda.get_arch_list())
    print('gpu     ', p.name, f'{p.total_memory / 2**30:.1f} GB', arch)
    print('built   ', ' '.join(built))
    GPU_OK = arch in built
    if not GPU_OK:
        # PyTorch >= 2.6 dropped Pascal (sm_60/sm_61), so a P100 cannot run
        # the current Kaggle image no matter what the code does. Catch it here
        # rather than 50 seconds later inside a forward pass.
        print()
        print('*' * 74)
        print(f'INCOMPATIBLE GPU: this torch build ships no kernels for {arch}.')
        print(f'  {p.name} is {arch}; this build targets: {" ".join(built)}')
        print('  Any CUDA op will raise "no kernel image is available".')
        print()
        print('  FIX: Notebook menu -> Settings -> Accelerator -> "GPU T4 x2"')
        print('       (T4 is sm_75 and is supported). Then re-run all.')
        print('*' * 74)
    else:
        # fp16 is worth enabling on any supported card here; tensor cores
        # (sm_70+) make it faster still, Pascal only saves memory.
        print('amp     ', 'fp16 with tensor cores' if p.major >= 7
              else 'fp16 (memory only, no tensor cores)')
else:
    print('no GPU detected - training will be extremely slow on CPU')

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
os.chdir(WORK)
print('workdir ', os.getcwd())
"""))

    # -------------------------------------------------------------- package
    cells.append(md("""
## 2. The DAETF-Net package

Written out module by module so the notebook is fully self-contained - no
`git clone`, no internet needed. Generated from `proposal1/daetf/` by
`tools/build_notebook.py`; edit the package and regenerate rather than editing
these cells.
"""))
    cells.append(code("import os; os.makedirs('daetf', exist_ok=True)"))

    for mod in MODULES:
        path = os.path.join(PKG, f"{mod}.py")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        cells.append(code(f"%%writefile daetf/{mod}.py\n{src}"))

    cells.append(code("""
import importlib, sys
for m in [k for k in list(sys.modules) if k.startswith('daetf')]:
    del sys.modules[m]
sys.path.insert(0, os.getcwd())
import daetf
print('daetf', daetf.__version__, 'loaded from', os.path.dirname(daetf.__file__))
"""))

    # ------------------------------------------------------------ selfcheck
    cells.append(md("""
## 3. Self-checks

Each claim in the architecture is verified numerically before any training
starts, so a broken environment fails in seconds rather than three hours in.
"""))
    cells.append(code("ok = daetf.selfcheck.run_all(device='cpu')\nassert ok, 'self-checks failed - stop here'"))

    # --------------------------------------------------------------- config
    cells.append(md("""
## 4. Configuration

Dataset roots and band counts are discovered from the filesystem - nothing is
hardcoded. Attach the CAVE and Harvard datasets and this cell finds them.
If discovery misses, set `DAETF_DATA_ROOTS` or pass `source_root=` explicitly.
"""))
    cells.append(code("""
QUICK = False          # True -> a few minutes end-to-end, for validating the pipeline
DEVICE = 'cuda' if (torch.cuda.is_available() and GPU_OK) else 'cpu'
if DEVICE == 'cpu' and torch.cuda.is_available():
    raise RuntimeError(
        'The attached GPU is not supported by this torch build (see the '
        'environment cell). Switch the accelerator to "GPU T4 x2" and re-run. '
        'Set DEVICE = "cpu" manually only if you intend a CPU run.')

cfg = daetf.Config(
    scale=4,                # fixed for every method, unlike the v1 benchmark
    patch=64,
    batch=16,
    iters=20000,
    width=64,
    amp=True,               # fp16: halves memory, faster on T4
    workers=2,
    out_dir=os.path.join(os.getcwd(), 'daetf_out'),
    val_every=1000,
    log_every=100,
)
if QUICK:
    cfg.iters, cfg.batch, cfg.val_every, cfg.log_every = 300, 8, 150, 50

cfg.resolve()           # auto-discovery: finds CAVE as source, Harvard as target
print(json.dumps({k: v for k, v in cfg.to_dict().items()
                  if k in ('source_root','target_root','bands','msi_bands',
                           'scale','patch','batch','iters')}, indent=1))
"""))

    cells.append(code("""
# what the loaders actually see
for name, root in (('source', cfg.source_root), ('target', cfg.target_root)):
    if not root:
        continue
    splits = daetf.available_splits(root)
    line = [f'{name}: {root}']
    for canon in ('Train', 'Test'):
        if canon in splits:
            line.append(f'{canon}={len(daetf.find_pairs(root, canon))} scenes')
    print('  '.join(line))
"""))

    # -------------------------------------------------------------- training
    cells.append(md("""
## 5. Training

Trained on the source domain only. When a target root is present, unlabelled
target patches are drawn alongside and aligned with an MMD penalty - **no target
ground truth is ever used**, so the cross-domain evaluation below stays honest.
"""))
    cells.append(code("""
t0 = time.time()
model, history = daetf.train(cfg, device=DEVICE)
print(f'\\ntrained in {(time.time() - t0) / 60:.1f} min')
"""))

    cells.append(code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(history['iter'], history['loss'])
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('SPC loss'); ax[0].grid(alpha=.3)
ax[0].set_title('training loss')
if history['val']:
    it = [v['iter'] for v in history['val']]
    ax[1].plot(it, [v['psnr'] for v in history['val']], marker='o', label='PSNR')
    ax[1].set_xlabel('iteration'); ax[1].set_ylabel('PSNR (dB)'); ax[1].grid(alpha=.3)
    axb = ax[1].twinx(); axb.plot(it, [v['sam'] for v in history['val']],
                                  marker='s', color='tab:red', label='SAM')
    axb.set_ylabel('SAM (deg)')
    ax[1].set_title('validation')
plt.tight_layout(); plt.savefig('training_curves.png', dpi=140); plt.show()
"""))

    # ------------------------------------------------------------ evaluation
    cells.append(md("""
## 6. In-domain evaluation

Full scenes via Hann-weighted overlapping tiles. All four metrics come from one
shared implementation with a fixed `data_range=1.0` - never the per-image
maximum, which is what inflated several v1 numbers on dark scenes.
"""))
    cells.append(code("""
print('=== in-domain (source Test) ===')
src_mean, src_rows = daetf.evaluate_dataset(
    model, cfg.source_root, cfg, 'Test', DEVICE, return_rows=True)
"""))

    cells.append(md("""
## 7. Cross-domain transfer - the actual research question

Zero-shot on the target dataset, then the same model adapted per scene using
**only** the two physics terms. No target ground truth is used at any point.
"""))
    cells.append(code("""
tgt_mean = tgt_rows = tta_mean = tta_rows = None
if cfg.target_root:
    print('=== zero-shot cross-domain (target Test) ===')
    tgt_mean, tgt_rows = daetf.evaluate_dataset(
        model, cfg.target_root, cfg, 'Test', DEVICE, return_rows=True)

    print('\\n=== cross-domain + self-supervised test-time adaptation ===')
    import numpy as np
    srf = np.load('daetf_srf.npy') if os.path.exists('daetf_srf.npy') else \\
          torch.load(os.path.join(cfg.out_dir, 'daetf_final.pth'),
                     map_location='cpu', weights_only=False)['srf']
    tta_mean, tta_rows = daetf.evaluate_with_tta(
        model, cfg.target_root, cfg, srf, 'Test', DEVICE, steps=30)
else:
    print('no target dataset attached - skipping the transfer study')
"""))

    cells.append(code("""
from daetf.experiments import comparison_table, summarise_rows
entries = {'DAETF-Net (in-domain)': src_mean}
if tgt_mean: entries['DAETF-Net (zero-shot transfer)'] = tgt_mean
if tta_mean: entries['DAETF-Net (transfer + TTA)'] = tta_mean
print(comparison_table(entries))

print('\\nper-metric spread (mean, std, bootstrap 95% CI):')
for name, rows in (('in-domain', src_rows), ('transfer', tgt_rows),
                   ('transfer+TTA', tta_rows)):
    if not rows: continue
    s = summarise_rows(rows)
    print(f'  {name}:')
    for m in ('psnr', 'ssim', 'sam', 'ergas'):
        e = s[m]
        print(f'    {m:6s} {e["mean"]:9.4f} +/- {e["std"]:7.4f}   '
              f'CI [{e["ci_lo"]:.4f}, {e["ci_hi"]:.4f}]  n={e["n"]}')
"""))

    # -------------------------------------------------------------- baselines
    cells.append(md("""
## 8. Comparison against the published baselines

The numbers below were recorded by the ten baseline notebooks in `existing/`.

**Read this table with care.** Those runs each used their own protocol - scale
factors of 4, 8, 16 and 32, different normalisations, and ERGAS scale arguments
that did not always match the actual downsampling. They are reproduced here for
reference, but a like-for-like claim requires re-running each baseline under
this protocol; `existing/results/` documents the discrepancies. Rows marked
`same-protocol` are the only strictly comparable ones.
"""))
    cells.append(code("""
# recorded by the baseline notebooks in existing/ (their own protocols)
BASELINES_SOURCE = {
    'Fusformer  (x4)':  {'psnr': 50.20, 'ssim': 0.9996, 'sam': 2.35, 'ergas': 0.85},
    'DHIF-Net   (x8)':  {'psnr': 48.80, 'ssim': 0.9966, 'sam': 2.16, 'ergas': 0.51},
    'DBIN       (x16)': {'psnr': 47.14, 'ssim': 0.9939, 'sam': 2.97, 'ergas': 0.33},
    'TSFN       (x8)':  {'psnr': 46.40, 'ssim': 0.9943, 'sam': 2.75, 'ergas': 0.63},
    'UTAL       (x32)': {'psnr': 41.37, 'ssim': 0.9906, 'sam': 4.62, 'ergas': 0.27},
    'MoG-DCN    (x32)': {'psnr': 38.55, 'ssim': 0.9715, 'sam': 6.62, 'ergas': 0.38},
    'PSRT       (x32)': {'psnr': 38.24, 'ssim': 0.9647, 'sam': 7.55, 'ergas': 0.39},
    'LRU        (x4)':  {'psnr': 37.16, 'ssim': 0.9768, 'sam': 3.73, 'ergas': 4.07},
    'AMGSGAN    (x4)':  {'psnr': 37.42, 'sam': 7.41, 'ergas': 2.00},
    'IFCASformer(CASSI)': {'psnr': 35.98, 'ssim': 0.9602, 'sam': 5.15, 'ergas': 3.55},
}
BASELINES_TARGET = {
    'DHIF-Net   (x8)':  {'psnr': 70.20, 'ssim': 0.9997, 'sam': 2.62, 'ergas': 0.93},
    'PSRT       (x32)': {'psnr': 57.25, 'ssim': 0.9918, 'sam': 15.87, 'ergas': 1.81},
    'TSFN       (x8)':  {'psnr': 56.35, 'ssim': 0.9935, 'sam': 8.29, 'ergas': 4.60},
    'UTAL       (x32)': {'psnr': 48.60, 'ssim': 0.9511, 'sam': 36.46, 'ergas': 16.60},
    'IFCASformer(CASSI)': {'psnr': 44.73, 'ssim': 0.8713, 'sam': 26.86, 'ergas': 85.41},
    'LRU        (x4)':  {'psnr': 38.76, 'ssim': 0.7320, 'sam': 7.77, 'ergas': 80.04},
    'AMGSGAN    (x4)':  {'psnr': 31.86, 'sam': 5.94, 'ergas': 7.16},
    'MoG-DCN    (x32)': {'psnr': 30.11, 'ssim': 0.9892, 'sam': 14.60, 'ergas': 4.64},
    'Fusformer  (x4)':  {'psnr': 25.80, 'ssim': 0.3059, 'sam': 58.89, 'ergas': 302.39},
}

print('=== SOURCE DOMAIN ===')
print(comparison_table({**BASELINES_SOURCE,
                        'DAETF-Net (ours, x%d, same-protocol)' % cfg.scale: src_mean}))
if tgt_mean:
    ours = {'DAETF-Net (ours, zero-shot)': tgt_mean}
    if tta_mean: ours['DAETF-Net (ours, +TTA)'] = tta_mean
    print('\\n=== TARGET DOMAIN (cross-dataset transfer) ===')
    print(comparison_table({**BASELINES_TARGET, **ours}))
"""))

    cells.append(md("""
### Spectral degradation under transfer

The single number that matters for the research claim: how much worse each
method's *spectral* fidelity gets when the domain changes.
"""))
    cells.append(code("""
rows = []
for name in set(BASELINES_SOURCE) & set(BASELINES_TARGET):
    s, t = BASELINES_SOURCE[name], BASELINES_TARGET[name]
    rows.append((name, s['sam'], t['sam'], t['sam'] - s['sam'],
                 s.get('ergas'), t.get('ergas')))
if tgt_mean:
    rows.append(('DAETF-Net (ours)', src_mean['sam'], tgt_mean['sam'],
                 tgt_mean['sam'] - src_mean['sam'],
                 src_mean['ergas'], tgt_mean['ergas']))
    if tta_mean:
        rows.append(('DAETF-Net (+TTA)', src_mean['sam'], tta_mean['sam'],
                     tta_mean['sam'] - src_mean['sam'],
                     src_mean['ergas'], tta_mean['ergas']))
rows.sort(key=lambda r: r[3])
print(f'{"method":<24}{"SAM src":>9}{"SAM tgt":>9}{"delta":>9}'
      f'{"ERGAS src":>11}{"ERGAS tgt":>11}')
print('-' * 73)
for n, ss, ts, d, es, et in rows:
    print(f'{n:<24}{ss:9.2f}{ts:9.2f}{d:+9.2f}'
          f'{(es if es is not None else float("nan")):11.2f}'
          f'{(et if et is not None else float("nan")):11.2f}')
"""))

    # ------------------------------------------------------------ efficiency
    cells.append(md("""
## 9. Cost

Parameters, GFLOPs, latency and peak memory for one full scene. The earlier
benchmark reported none of these, so nothing could be judged per unit of compute.
"""))
    cells.append(code("""
prof = daetf.experiments.profile_model(model, cfg, device=DEVICE, hr=512)
for k, v in prof.items():
    print(f'  {k:14s} {v:.3f}' if isinstance(v, float) else f'  {k:14s} {v}')
"""))

    # ------------------------------------------------------------- ablation
    cells.append(md("""
## 10. Component ablation

Each variant is retrained from scratch with an identical budget, seed and data
order, and each disabled module is replaced by a **matched control arm** (plain
convolutions for the equivariant stem, bicubic for back-projection, concat-fuse
for the Tucker interaction) rather than by nothing - otherwise the ablation
measures capacity, not the mechanism.

This retrains the model once per variant. Set `RUN_ABLATION = True` and expect
roughly 8x the single-run time.
"""))
    cells.append(code("""
RUN_ABLATION = False
ablation = None
if RUN_ABLATION:
    abl_cfg = daetf.Config(**cfg.to_dict())
    abl_cfg.iters = max(2000, cfg.iters // 4)   # shorter budget, identical across variants
    ablation = daetf.experiments.run_ablation(abl_cfg, device=DEVICE)
    print('\\n' + daetf.experiments.ablation_table(ablation))
else:
    print('ablation skipped (set RUN_ABLATION = True)')
"""))

    # --------------------------------------------------------- interpretability
    cells.append(md("""
## 11. What the model learned

The MoE gate is a per-pixel distribution over experts, so it can be shown as a
map: this is the interpretability the design document claimed and v1 could not
deliver, since its gate was a single global vector per image.
"""))
    cells.append(code("""
import matplotlib.pyplot as plt
import numpy as np

if getattr(model, 'moe', None) is not None:
    pairs = daetf.find_pairs(cfg.source_root, 'Test')
    cache = daetf.SceneCache(cfg.bands, cfg.msi_bands, limit=1)
    hsi, rgb = cache.get(*pairs[0])
    h = (hsi.shape[1] // cfg.scale) * cfg.scale
    w = (hsi.shape[2] // cfg.scale) * cfg.scale
    h, w = min(h, 256), min(w, 256)
    gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(DEVICE)
    msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(DEVICE)
    lr = daetf.FixedDegradation.from_config(cfg).to(DEVICE)(gt)
    model.eval()
    with torch.no_grad():
        out = model(lr, msi)
    gate = model.moe.last_gate[0].cpu().numpy()

    n = gate.shape[0]
    fig, axes = plt.subplots(1, n + 2, figsize=(3 * (n + 2), 3))
    axes[0].imshow(np.clip(msi[0].cpu().numpy().transpose(1, 2, 0), 0, 1))
    axes[0].set_title('MSI'); axes[0].axis('off')
    err = np.abs(out['out'][0].cpu().numpy() - gt[0].cpu().numpy()).mean(0)
    im = axes[1].imshow(err, cmap='inferno'); axes[1].set_title('abs error')
    axes[1].axis('off'); plt.colorbar(im, ax=axes[1], fraction=.046)
    for i in range(n):
        axes[i + 2].imshow(gate[i], cmap='viridis', vmin=0, vmax=1)
        axes[i + 2].set_title(f'expert {i}  ({gate[i].mean():.2f})')
        axes[i + 2].axis('off')
    plt.tight_layout(); plt.savefig('moe_routing.png', dpi=140); plt.show()
    print('expert usage:', np.round(gate.mean(axis=(1, 2)), 3),
          '(uniform would be', round(1 / n, 3), ')')
"""))

    # --------------------------------------------------------------- outputs
    cells.append(md("""
## 12. Save everything

Results, per-scene tables, the environment report and the checkpoints are
written to `/kaggle/working` so the run is reproducible and the numbers can be
pulled straight into the manuscript.
"""))
    cells.append(code("""
from daetf.experiments import (environment_report, save_results, write_report,
                               comparison_table, summarise_rows)

payload = {
    'config': cfg.to_dict(),
    'environment': environment_report(),
    'params_M': model.n_params() / 1e6,
    'efficiency': prof,
    'source': {'mean': src_mean, 'rows': src_rows,
               'summary': summarise_rows(src_rows)},
}
if tgt_rows:
    payload['target_zeroshot'] = {'mean': tgt_mean, 'rows': tgt_rows,
                                  'summary': summarise_rows(tgt_rows)}
if tta_rows:
    payload['target_tta'] = {'mean': tta_mean, 'rows': tta_rows,
                             'summary': summarise_rows(tta_rows)}
if ablation:
    payload['ablation'] = [{k: v for k, v in r.items()
                            if not k.endswith('_rows')} for r in ablation]

save_results('results.json', payload)

sections = [('Environment', '\\n'.join(f'- **{k}**: {v}'
             for k, v in environment_report().items()))]
entries = {'DAETF-Net (in-domain)': src_mean}
if tgt_mean: entries['DAETF-Net (zero-shot)'] = tgt_mean
if tta_mean: entries['DAETF-Net (+TTA)'] = tta_mean
sections.append(('Results', comparison_table(entries)))
sections.append(('Cost', '\\n'.join(f'- **{k}**: {v}' for k, v in prof.items())))
if ablation:
    sections.append(('Ablation', daetf.experiments.ablation_table(ablation)))
write_report('RESULTS.md', 'DAETF-Net run report', sections)

print('written:', [f for f in os.listdir('.')
                   if f.endswith(('.json', '.md', '.png', '.pth'))])
print('checkpoints:', os.listdir(cfg.out_dir))
"""))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {OUT}")
    print(f"  {len(nb['cells'])} cells ({n_code} code), "
          f"{os.path.getsize(OUT) / 1024:.0f} KB")
