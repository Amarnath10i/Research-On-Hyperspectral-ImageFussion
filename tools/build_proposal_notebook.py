"""Generate a self-contained Kaggle notebook for any proposal.

    python tools/build_proposal_notebook.py            # all proposals
    python tools/build_proposal_notebook.py proposal3  # just one

Each notebook embeds the shared `hsifusion` library plus that proposal's own
package with %%writefile cells, so it runs on Kaggle with no clone and no
internet, and cannot drift from the source - regenerate rather than editing
cells.

Every proposal notebook runs the SAME protocol and the same classical
baselines, so the four sets of results are directly comparable.
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = os.path.join(REPO, "common", "hsifusion")

COMMON_MODULES = ["io_utils", "config", "degrade", "metrics", "losses",
                  "data", "engine", "baselines", "experiments", "__init__"]


def _lines(text: str) -> list:
    """nbformat keeps the newline on every source line; splitting without
    keepends collapses the cell onto one line."""
    return text.splitlines(keepends=True)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source.strip())}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(source.strip("\n"))}


# --------------------------------------------------------------- proposals
PROPOSALS = {
    "proposal2": {
        "pkg": "unfoldfusion",
        "title": "UnfoldFusion - deep-unfolded variational solver",
        "modules": ["model", "__init__"],
        "blurb": """
Proposal 2 of four. Where DAETF-Net (proposal 1) is a feed-forward network
*told* about the physics through two loss terms, UnfoldFusion has no free-form
backbone: it is an optimisation algorithm whose iterations are unrolled into
layers, so the observation model is structural.

Half-quadratic splitting alternates

* a **prior step** - a learned proximal denoiser, shared across stages and
  conditioned on the current penalty weight rho, and
* a **data step** - a least-squares problem enforcing `B(X)=Y_h` and `R(X)=Y_m`,
  solved by conjugate gradients in a low-rank spectral subspace estimated from
  the LR-HSI, which is what makes a 31-band solve affordable inside a forward
  pass.

The blur operator `B` is *estimated per input* rather than assumed, and only
the prior is free-form - the smallest surface for domain-specific overfitting.

**Why it should transfer:** every stage re-imposes agreement with the actual
observations. A feed-forward net that memorised CAVE statistics has nothing
forcing it to re-explain a Harvard observation; here a solution that violates
`B(X)=Y_h` is driven out at every stage regardless of which dataset it came
from.
""",
        "extra_title": "Stage count at inference",
        "extra_md": """
The denoiser weights are shared across stages and extra stages reuse the final
converged rho, so the solver depth can be changed *after* training. A
feed-forward network cannot trade compute for accuracy this way.
""",
        "extra_code": """
import copy
print(f'{"stages":>8} {"PSNR":>8} {"SSIM":>8} {"SAM":>8} {"ERGAS":>9}')
base_stages = model.stages
depth_rows = []
for s in (2, 4, 6, 8, 12):
    model.stages = s
    m = hsifusion.evaluate_dataset(model, cfg.source_root, cfg, 'Test', DEVICE,
                                   limit=4, verbose=False)
    depth_rows.append({'stages': s, **m})
    print(f'{s:>8} {m["psnr"]:>8.3f} {m["ssim"]:>8.4f} {m["sam"]:>8.3f} '
          f'{m["ergas"]:>9.3f}')
model.stages = base_stages
print(f'\\ntrained with {base_stages} stages; the row at that depth is the '
      f'reported one, the others show the accuracy/compute trade')
""",
    },
    "proposal3": {
        "pkg": "continuumfusion",
        "title": "ContinuumFusion - arbitrary-scale implicit representation",
        "modules": ["model", "__init__"],
        "blurb": """
Proposal 3 of four. Proposals 1 and 2 both produce a fixed output grid and are
welded to one scale factor; changing it means retraining.

ContinuumFusion never represents the image as a grid. It learns a continuous
function `f(x, y, lambda) -> radiance`, conditioned on local features from the
observations, and **samples** it wherever an output pixel is wanted. The scale
factor becomes a query parameter rather than an architectural constant.

* Latent features live on the **LR grid**, because that is where the spectra
  were actually measured. Spatial detail is read at query time from the
  full-resolution MSI.
* The band index is a **coordinate**, entering through a learned embedding, so
  one MLP serves every band and the representation is continuous along
  wavelength as well as space.
* A local ensemble over the four neighbouring latent cells removes the blocking
  that decoding from the single nearest cell produces.

**Why this gap matters:** the benchmark in `existing/` is unusable precisely
because its ten methods ran at x4, x8, x16 and x32. That is not sloppy
bookkeeping, it is a property of grid-based architectures. A model that handles
any factor from one set of weights makes the comparison well-posed.
""",
        "extra_title": "The claim: one model, every scale factor",
        "extra_md": """
This is the experiment the other three proposals cannot run without retraining,
and the one that makes the x4/x8/x16/x32 confusion in `existing/` addressable.
""",
        "extra_code": """
scale_rows = continuumfusion.evaluate_scales(
    model, cfg.source_root, cfg, scales=(4, 8, 16, 32), device=DEVICE,
    limit=4, verbose=False)
print(f'{"scale":>6} {"PSNR":>8} {"SSIM":>8} {"SAM":>8} {"ERGAS":>9}')
for r in scale_rows:
    print(f'x{r["scale"]:<5} {r["psnr"]:>8.3f} {r["ssim"]:>8.4f} '
          f'{r["sam"]:>8.3f} {r["ergas"]:>9.3f}')
print('\\nall rows from ONE set of weights - no retraining between them')

print('\\n--- non-integer and non-square queries ---')
import torch
pairs = hsifusion.find_pairs(cfg.source_root, 'Test')
cache = hsifusion.SceneCache(cfg.bands, cfg.msi_bands, limit=1)
hsi, rgb = cache.get(*pairs[0])
side = (min(hsi.shape[1], 256) // 32) * 32
gt = torch.from_numpy(hsi[:, :side, :side].astype('float32'))[None].to(DEVICE)
msi_q = torch.from_numpy(rgb[:, :side, :side].astype('float32'))[None].to(DEVICE)
lr_q = hsifusion.FixedDegradation(cfg.scale, cfg.blur_ksize,
                                  cfg.eval_sigma).to(DEVICE)(gt)
model.eval()
with torch.no_grad():
    for hw in [(side, side), (side // 2, side), (int(side * 1.5), side)]:
        o = model(lr_q, msi_q, out_hw=hw)['out']
        print(f'  query {str(hw):16s} -> {tuple(o.shape[-2:])}')
""",
    },
    "proposal4": {
        "pkg": "zerofusion",
        "title": "ZeroFusion - self-supervised per-scene fusion, no training set",
        "modules": ["model", "__init__"],
        "blurb": """
Proposal 4 of four, and the control arm for the whole study.

Proposals 1-3 learn from a training split and are then asked to generalise.
ZeroFusion has **no training split at all**. Given a single pair
(LR-HSI, MSI) it optimises a small network from scratch on that pair, using
only losses that need no ground truth:

    ||B(X) - Y_h||  +  ||R(X) - Y_m||  +  priors

The model is physically grounded rather than a generic denoiser: linear
spectral unmixing, `X = E @ softmax(A) * scale`, with endmembers `E`
initialised from the scene's own spectral subspace and abundances `A` predicted
from both observations. Sum-to-one and non-negativity come free from the
softmax, which rules out the spectral hallucination that shows up as large SAM.

**Why this is the most important of the four.** Every cross-domain number the
others report answers "how much does training on CAVE hurt you on Harvard?".
ZeroFusion never trains on CAVE, so it *cannot* suffer domain shift - its
Harvard score is the same kind of quantity as its CAVE score. That makes it the
reference line:

* if a trained proposal beats it cross-domain, the training genuinely
  transferred something useful;
* if not, then what that model learned on the source was worth less than
  nothing on the target, and per-scene optimisation is simply the better method.

Very few fusion papers include this comparison. It is easy to look strong
cross-domain until someone asks whether a method that ignores the training set
entirely would have done better.

**Cost:** optimisation is per scene, so training cost is zero and inference
cost is high. That trade is the honest headline and is reported explicitly.
""",
        "extra_title": "Cost per scene",
        "extra_md": """
The trade this method makes, stated plainly rather than buried.
""",
        "extra_code": """
sec = src_mean.get('seconds_per_scene', float('nan'))
print(f'training cost:        0 (there is no training phase)')
print(f'inference per scene:  {sec:.1f} s')
print(f'scenes scored:        {len(src_rows)} source'
      + (f' + {len(tgt_rows)} target' if tgt_rows else ''))
print('\\nfor contrast, proposals 1-3 pay hours of training once and then'
      '\\nfractions of a second per scene.')
""",
    },
}


def build(key: str) -> dict:
    spec = PROPOSALS[key]
    pkg = spec["pkg"]
    pkg_dir = os.path.join(REPO, key, pkg)
    is_zero = key == "proposal4"
    cells = []

    cells.append(md(f"""
# {spec['title']}

{spec['blurb'].strip()}

---

**Shared protocol.** This notebook imports the same `hsifusion` library as the
other three proposals: same degradation, same scale factor, same metric code
with a fixed `data_range=1.0`, same classical baselines, same scenes. A
difference between proposals is therefore attributable to the architecture and
not to one of them quietly evaluating differently - which is exactly how the
ten published baselines in `existing/` became incomparable.

**Hardware.** Choose **GPU T4 x2**. PyTorch >= 2.6 ships no kernels for the
P100's `sm_60`, so a P100 fails on the first CUDA op whatever the code does.
The next cell checks this and tells you if the accelerator is wrong.
"""))

    cells.append(md("## 1. Environment"))
    cells.append(code("""
import os, sys, json, time, math, warnings
warnings.filterwarnings('ignore', category=UserWarning)
import torch, numpy as np

print('python  ', sys.version.split()[0])
print('torch   ', torch.__version__)
GPU_OK = False
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    arch = f'sm_{p.major}{p.minor}'
    built = list(torch.cuda.get_arch_list())
    print('gpu     ', p.name, f'{p.total_memory / 2**30:.1f} GB', arch)
    print('built   ', ' '.join(built))
    GPU_OK = arch in built
    if not GPU_OK:
        print('\\n' + '*' * 74)
        print(f'INCOMPATIBLE GPU: this torch build has no kernels for {arch}.')
        print('  FIX: Settings -> Accelerator -> "GPU T4 x2", then re-run all.')
        print('*' * 74)
else:
    print('no GPU - this will be very slow on CPU')

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
os.chdir(WORK)
print('workdir ', os.getcwd())
"""))

    # ---- shared library ------------------------------------------------
    cells.append(md("""
## 2. Shared library (`hsifusion`)

Identical in all four proposal notebooks: protocol, data pipeline, metrics,
degradation model, classical baselines and the model-agnostic engine.
"""))
    cells.append(code("import os; os.makedirs('hsifusion', exist_ok=True)"))
    for mod in COMMON_MODULES:
        with open(os.path.join(COMMON, f"{mod}.py"), "r", encoding="utf-8") as f:
            cells.append(code(f"%%writefile hsifusion/{mod}.py\n{f.read()}"))

    # ---- proposal package ----------------------------------------------
    cells.append(md(f"""
## 3. This proposal (`{pkg}`)

The only part that differs between the four notebooks.
"""))
    cells.append(code(f"import os; os.makedirs('{pkg}', exist_ok=True)"))
    for mod in spec["modules"]:
        with open(os.path.join(pkg_dir, f"{mod}.py"), "r", encoding="utf-8") as f:
            cells.append(code(f"%%writefile {pkg}/{mod}.py\n{f.read()}"))

    cells.append(code(f"""
import sys
for m in [k for k in list(sys.modules) if k.startswith(('hsifusion', '{pkg}'))]:
    del sys.modules[m]
sys.path.insert(0, os.getcwd())
import hsifusion
import {pkg}
print('hsifusion', hsifusion.__version__, '|', '{pkg}', {pkg}.__version__)
"""))

    # ---- config ---------------------------------------------------------
    cells.append(md("""
## 4. Configuration

Dataset roots and band counts are discovered from the filesystem - nothing is
hardcoded. Attach the CAVE and Harvard datasets and this finds them.
"""))
    quick = ("cfg.steps, cfg.max_side = 120, 128" if is_zero
             else "cfg.iters, cfg.batch, cfg.val_every, cfg.log_every = 300, 8, 150, 50")
    cells.append(code(f"""
QUICK = False        # True -> a few minutes, to validate the pipeline first
DEVICE = 'cuda' if (torch.cuda.is_available() and GPU_OK) else 'cpu'
if DEVICE == 'cpu' and torch.cuda.is_available():
    raise RuntimeError('Unsupported GPU - switch the accelerator to "GPU T4 x2".')

cfg = {pkg}.Config(
    scale=4,          # identical across all four proposals
    patch=64,
    out_dir=os.path.join(os.getcwd(), '{pkg}_out'),
)
if QUICK:
    {quick}

cfg.resolve()
print(json.dumps({{k: v for k, v in cfg.to_dict().items()
                  if k in ('name','source_root','target_root','bands',
                           'msi_bands','scale','patch','batch','iters')}}, indent=1))
"""))

    # ---- train / fit ----------------------------------------------------
    if is_zero:
        cells.append(md("""
## 5. No training phase

There is nothing to train. The only preparation is measuring the spectral
response function from the data, which the physics loss needs.
"""))
        cells.append(code("""
from hsifusion.data import estimate_srf
srf = estimate_srf(cfg.source_root, 'Train', cfg)
print('SRF', srf.shape, 'column sums', srf.sum(0).round(3).tolist())
np.save('srf.npy', srf)
history = None
"""))
        cells.append(md("""
## 6. Fit and score every scene, independently

Each scene is optimised from scratch. Nothing carries over between scenes, so
there is no train/test split to respect and no domain to shift from.
"""))
        cells.append(code("""
print('=== source domain ===')
src_mean, src_rows = zerofusion.evaluate_zeroshot(
    cfg.source_root, cfg, srf, 'Test', DEVICE)

tgt_mean = tgt_rows = None
if cfg.target_root:
    print('\\n=== target domain (same method, nothing changes) ===')
    tgt_mean, tgt_rows = zerofusion.evaluate_zeroshot(
        cfg.target_root, cfg, srf, 'Test', DEVICE)
tta_mean = tta_rows = None
model = zerofusion.build_model(cfg).to(DEVICE)   # for the cost profile only
"""))
    else:
        cells.append(md("""
## 5. Training

Trained on the source domain only. The target dataset is never used with labels.
"""))
        cells.append(code(f"""
t0 = time.time()
model, history = {pkg}.train(cfg, device=DEVICE)
print(f'\\ntrained in {{(time.time() - t0) / 60:.1f}} min')

srf = torch.load(os.path.join(cfg.out_dir, f'{{cfg.name}}_final.pth'),
                 map_location='cpu', weights_only=False)['srf']
"""))
        cells.append(code("""
import matplotlib.pyplot as plt
if history and history['iter']:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(history['iter'], history['loss'])
    ax[0].set_xlabel('iteration'); ax[0].set_ylabel('loss'); ax[0].grid(alpha=.3)
    if history['val']:
        it = [v['iter'] for v in history['val']]
        ax[1].plot(it, [v['psnr'] for v in history['val']], marker='o')
        ax[1].set_xlabel('iteration'); ax[1].set_ylabel('PSNR (dB)')
        ax[1].grid(alpha=.3)
        axb = ax[1].twinx()
        axb.plot(it, [v['sam'] for v in history['val']], marker='s', color='tab:red')
        axb.set_ylabel('SAM (deg)')
    plt.tight_layout(); plt.savefig('training_curves.png', dpi=140); plt.show()
"""))
        cells.append(md("""
## 6. In-domain and cross-domain evaluation

Full scenes via Hann-weighted overlapping tiles, scored by the shared metric
module with a fixed `data_range=1.0` - never the per-image maximum, which is
what inflated several numbers in the original benchmark.
"""))
        cells.append(code(f"""
print('=== in-domain (source Test) ===')
src_mean, src_rows = hsifusion.evaluate_dataset(
    model, cfg.source_root, cfg, 'Test', DEVICE, return_rows=True)

tgt_mean = tgt_rows = tta_mean = tta_rows = None
if cfg.target_root:
    print('\\n=== zero-shot cross-domain (target Test) ===')
    tgt_mean, tgt_rows = hsifusion.evaluate_dataset(
        model, cfg.target_root, cfg, 'Test', DEVICE, return_rows=True)

    print('\\n=== cross-domain + physics-only test-time adaptation ===')
    crit = {pkg}.build_loss(cfg, srf).to(DEVICE)
    tta_mean, tta_rows = hsifusion.evaluate_with_tta(
        model, cfg.target_root, cfg, crit, 'Test', DEVICE, steps=30)
"""))

    # ---- baselines + significance ---------------------------------------
    cells.append(md("""
## 7. Same-protocol baselines and paired significance

These classical methods need no checkpoints, so they run through the identical
pipeline. They are the only rows strictly comparable to ours, and they set the
floor any learned method must clear.
"""))
    cells.append(code("""
print('=== classical baselines, source domain ===')
base_src = hsifusion.evaluate_all_baselines(cfg.source_root, cfg, srf, 'Test',
                                            DEVICE, verbose=False)
for name, r in base_src.items():
    m = r['mean']
    print(f'  {name:<14} PSNR={m["psnr"]:7.3f}  SSIM={m["ssim"]:.4f}  '
          f'SAM={m["sam"]:6.3f}  ERGAS={m["ergas"]:8.3f}')

base_tgt = None
if cfg.target_root:
    print('\\n=== classical baselines, target domain ===')
    base_tgt = hsifusion.evaluate_all_baselines(cfg.target_root, cfg, srf,
                                                'Test', DEVICE, verbose=False)
    for name, r in base_tgt.items():
        m = r['mean']
        print(f'  {name:<14} PSNR={m["psnr"]:7.3f}  SSIM={m["ssim"]:.4f}  '
              f'SAM={m["sam"]:6.3f}  ERGAS={m["ergas"]:8.3f}')
"""))
    cells.append(code("""
from hsifusion.experiments import (comparison_table, compare_methods,
                                   significance_table, summarise_rows)

entries = {k: v['mean'] for k, v in base_src.items()}
entries[f'{cfg.name} (in-domain)'] = src_mean
print('=== SOURCE DOMAIN, same protocol ===')
print(comparison_table(entries))

if tgt_mean and base_tgt:
    t = {k: v['mean'] for k, v in base_tgt.items()}
    t[f'{cfg.name} (zero-shot)'] = tgt_mean
    if tta_mean:
        t[f'{cfg.name} (+TTA)'] = tta_mean
    print('\\n=== TARGET DOMAIN, same protocol ===')
    print(comparison_table(t))

print('\\n=== paired significance vs same-protocol baselines (SAM) ===')
cmps = [compare_methods(src_rows, r['rows'], cfg.name, name)
        for name, r in base_src.items()]
print(significance_table(cmps, metric='sam'))
"""))

    # ---- proposal-specific experiment -----------------------------------
    cells.append(md(f"""
## 8. {spec['extra_title']}

{spec['extra_md'].strip()}
"""))
    cells.append(code(spec["extra_code"]))

    # ---- cost + save -----------------------------------------------------
    cells.append(md("## 9. Cost and saved results"))
    cells.append(code("""
try:
    prof = hsifusion.experiments.profile_model(model, cfg, device=DEVICE, hr=512)
    for k, v in prof.items():
        print(f'  {k:14s} {v:.3f}' if isinstance(v, float) else f'  {k:14s} {v}')
except Exception as exc:
    prof = {'note': f'profile skipped: {exc}'}
    print(prof['note'])
"""))
    cells.append(code("""
from hsifusion.experiments import environment_report, save_results, write_report

payload = {
    'proposal': cfg.name,
    'config': cfg.to_dict(),
    'environment': environment_report(),
    'efficiency': prof,
    'source': {'mean': src_mean, 'rows': src_rows,
               'summary': summarise_rows(src_rows)},
    'baselines_same_protocol': {
        'source': {k: v['mean'] for k, v in base_src.items()},
        'target': ({k: v['mean'] for k, v in base_tgt.items()} if base_tgt else None)},
    'significance_vs_same_protocol': cmps,
}
if tgt_rows:
    payload['target_zeroshot'] = {'mean': tgt_mean, 'rows': tgt_rows,
                                  'summary': summarise_rows(tgt_rows)}
if tta_rows:
    payload['target_tta'] = {'mean': tta_mean, 'rows': tta_rows,
                             'summary': summarise_rows(tta_rows)}

save_results(f'{cfg.name}_results.json', payload)
sections = [('Environment', '\\n'.join(f'- **{k}**: {v}'
             for k, v in environment_report().items())),
            ('Results', comparison_table(entries))]
write_report(f'{cfg.name}_RESULTS.md', f'{cfg.name} run report', sections)
print('written:', [f for f in os.listdir('.') if f.endswith(('.json', '.md', '.png'))])
"""))

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"},
                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    keys = sys.argv[1:] or list(PROPOSALS)
    for key in keys:
        nb = build(key)
        out_dir = os.path.join(REPO, key, "notebooks")
        os.makedirs(out_dir, exist_ok=True)
        name = PROPOSALS[key]["pkg"]
        out = os.path.join(out_dir, f"{name}_Kaggle_GPU.ipynb")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"{key}: {len(nb['cells'])} cells ({n_code} code), "
              f"{os.path.getsize(out) / 1024:.0f} KB -> {out}")


if __name__ == "__main__":
    main()
