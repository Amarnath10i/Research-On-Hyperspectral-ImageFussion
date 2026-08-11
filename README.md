# Hyperspectral-Multispectral Image Fusion

A benchmark of ten published HSI-MSI fusion methods, and **four new
architectures** attacking the failure that benchmark exposed by four genuinely
different mechanisms.

```
common/hsifusion/    shared protocol, data, metrics, engine, classical baselines
existing/            the ten benchmarked methods (notebooks, papers, results)
proposal1/  DAETF-Net        equivariant tensor + MoE, physics in the LOSS
proposal2/  UnfoldFusion     unrolled variational solver, physics in the ARCHITECTURE
proposal3/  ContinuumFusion  implicit representation, ARBITRARY SCALE FACTOR
proposal4/  ZeroFusion       self-supervised per-scene, NO TRAINING SET
literature_survey/   Crossref -> Unpaywall -> annotated Excel pipeline
tools/               notebook generators, Kaggle push/repair/re-run automation
```

Each `proposalN/` contains `<package>/`, `notebooks/`, `docs/`, `results/`.

---

## The finding that drives all of this

Re-running ten fusion methods across CAVE and Harvard shows the usual headline
metric points the wrong way.

Five of nine methods score **higher PSNR** on the harder dataset — per-image
maximum normalisation inflates it on dark scenes. What actually breaks is
**spectral fidelity**: seven of nine lose 4–57 degrees of SAM, and ERGAS rises
by up to two orders of magnitude.

| Method | SAM CAVE → Harvard | ERGAS CAVE → Harvard |
|---|---|---|
| Fusformer | 2.35 → **58.89** | 0.85 → **302.39** |
| UTAL | 4.62 → **36.46** | 0.27 → 16.60 |
| IFCASformer | 5.15 → **26.86** | 3.55 → **85.41** |
| PSRT | 7.55 → 15.87 | 0.39 → 1.81 |
| DHIF-Net | 2.16 → 2.62 | 0.51 → 0.93 |

Full table and caveats: [`existing/results/BENCHMARK.md`](existing/results/BENCHMARK.md).

**Stated up front:** those ten notebooks each used their own protocol — scale
factors of ×4, ×8, ×16 and ×32, different normalisations, and one method on a
different dataset and task. The numbers are real but **not comparable across
methods**. No cross-method ranking here is settled until every baseline is
re-run under one protocol.

---

## The four proposals

All target spectral fidelity under domain shift, by different mechanisms.

| | Mechanism | Owns which gap | Params |
|---|---|---|---|
| **P1** [DAETF-Net](proposal1/docs/ARCHITECTURE.md) | p4-equivariant CNN + Tucker + per-pixel MoE + wavelet; physics in the loss | geometric robustness, interpretable routing | 2.06 M |
| **P2** [UnfoldFusion](proposal2/docs/ARCHITECTURE.md) | half-quadratic splitting unrolled; CG data step in a low-rank spectral subspace | interpretability; smallest overfitting surface | 0.09 M |
| **P3** [ContinuumFusion](proposal3/docs/ARCHITECTURE.md) | continuous field `f(x,y,λ)`, decoded per queried coordinate | **arbitrary scale factor** from one model | ~0.1 M |
| **P4** [ZeroFusion](proposal4/docs/ARCHITECTURE.md) | per-scene unmixing fitted from the observations alone | **immune to domain shift by construction** | 0.03 M |

**P4 is the control arm.** It never trains on a source domain, so it cannot
suffer domain shift. If P1–P3 cannot beat it cross-domain, what they learned on
the source was worth less than nothing on the target — and per-scene
optimisation is simply the better method. Few fusion papers include this test.

### Every claim is checked numerically, not asserted

| Claim | Check | Result |
|---|---|---|
| P1 EFE is p4-equivariant | `‖rot90(f(x)) − f(rot90(x))‖` | 7.6e-06 |
| P1 wavelet is invertible | `‖IDWT(DWT(x)) − x‖` | 2.4e-07 |
| P1 Tucker core is used | gradient reaches the core | pass |
| P2 `Bᵀ` is the true adjoint | `⟨Bx,y⟩ = ⟨x,Bᵀy⟩` | 1.0e-06 |
| P2 CG solves the data step | residual over 16 steps | 1.3e+02 → 1.4e-05 |
| P3 scale is a query parameter | ×2/×3/×4/×5/×8, one model | pass |
| P4 uses no ground truth | loss term audit | pass |
| SRF recovered from data | vs known synthetic response | 2.3e-08 |

Testing caught bugs that would have invalidated results while still training
happily — a padding mismatch that broke P2's adjoint and would have reduced the
solver to a stack of denoisers, and a forward-pass `clamp` that froze P4's
optimisation entirely. Both are documented in the respective architecture docs.

---

## Running it

### On Kaggle

1. Upload the proposal's notebook from `proposalN/notebooks/`
2. Attach the CAVE and Harvard datasets
3. **Settings → Accelerator → "GPU T4 x2"** — set this in the UI. PyTorch ≥ 2.6
   ships no kernels for the P100's `sm_60`, so a P100 fails on the first CUDA op.
   The push API accepts an `accelerator` field but Kaggle ignores it and will
   silently reset your choice, so do not push to a kernel between setting it and
   running.
4. Set `QUICK = True` first (a few minutes) to validate, then `False`

Notebooks are self-contained — no clone, no internet.

### Locally

```bash
pip install -r requirements.txt
export PYTHONPATH=common:proposal1:proposal2:proposal3:proposal4

python -c "import daetf; daetf.selfcheck.run_all()"     # verify P1's mechanisms

python - <<'PY'
import hsifusion, unfoldfusion
cfg = unfoldfusion.Config().resolve()      # discovers datasets, infers bands
model, hist = unfoldfusion.train(cfg)
hsifusion.evaluate_dataset(model, cfg.source_root, cfg)
PY
```

Datasets are discovered under `/kaggle/input`, `./data` or the cwd, at any
nesting depth. Override with `DAETF_DATA_ROOTS` or `Config(source_root=...)`.
Nothing is hardcoded — band counts are read from the files.

### Regenerating notebooks

```bash
python tools/build_notebook.py                  # proposal 1
python tools/build_proposal_notebook.py         # proposals 2-4
```

Edit the packages and regenerate; never edit notebook cells directly.

---

## Why one shared library

`common/hsifusion` holds the protocol, data pipeline, metric implementation,
degradation model, classical baselines and a model-agnostic engine. Any model
following `forward(lr_hsi, msi) -> {"out": tensor}` trains and scores
identically.

`BaseConfig` carries the protocol fields; each proposal subclasses it and adds
only its architecture. So a difference between two proposals is attributable to
the architecture — not to one of them evaluating at a different scale factor.
That is precisely how the ten published baselines became incomparable, and the
structure exists to stop it happening again.

Every proposal is also scored against the same three classical baselines
(bicubic, GSA, a coupled subspace estimator) which need no checkpoints and so
run through the identical pipeline. They are the only strictly comparable rows
available today, and the floor any learned method must clear.

---

## Status

Implementations, evaluation harness, and automation are complete and tested.

**No results on real CAVE or Harvard data exist yet.** Every number in this
repository from the four proposals comes from a synthetic test harness used to
verify correctness. Until the training runs complete and the ten baselines are
re-run under one protocol, this repository makes **no claim** about beating the
state of the art.
