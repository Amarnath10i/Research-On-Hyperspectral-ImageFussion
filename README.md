# Hyperspectral-Multispectral Image Fusion

Two things live here: a **benchmark** of ten published HSI-MSI fusion methods,
and **DAETF-Net**, a new method built to fix the failure that benchmark exposed.

```
existing/            the ten benchmarked methods
  notebooks/         one notebook per method per dataset, named for the method
  papers/            paper PDFs (untracked) + INDEX.md mapping paper -> notebook
  results/           BENCHMARK.md - what was measured, and why it is not comparable
proposal1/           DAETF-Net
  daetf/             the implementation (installable package)
  notebooks/         DAETF_Net_Kaggle_P100.ipynb - self-contained, runs on Kaggle
  docs/              ARCHITECTURE.md, GAP_ANALYSIS.md
  results/           run outputs
literature_survey/   Crossref -> Unpaywall -> annotated Excel pipeline
tools/               notebook generator, Kaggle push/repair/re-run automation
```

---

## The finding that drives this work

Re-running ten fusion methods across CAVE and Harvard shows that the usual
headline metric points the wrong way.

Five of nine methods score **higher PSNR** on the harder dataset — per-image
maximum normalisation inflates it on dark scenes. What actually breaks is
**spectral fidelity**: seven of nine lose 4-57 degrees of SAM, and ERGAS rises by
up to two orders of magnitude.

| Method | SAM CAVE → Harvard | ERGAS CAVE → Harvard |
|---|---|---|
| Fusformer | 2.35 → **58.89** | 0.85 → **302.39** |
| UTAL | 4.62 → **36.46** | 0.27 → 16.60 |
| IFCASformer | 5.15 → **26.86** | 3.55 → **85.41** |
| PSRT | 7.55 → 15.87 | 0.39 → 1.81 |
| DHIF-Net | 2.16 → 2.62 | 0.51 → 0.93 |

Full table and the protocol caveats: [`existing/results/BENCHMARK.md`](existing/results/BENCHMARK.md).

**Caveat, stated up front:** those ten notebooks each used their own protocol —
scale factors of ×4, ×8, ×16 and ×32, different normalisations, mismatched ERGAS
scale arguments, and one method evaluated on a different dataset and task. The
numbers are real but **not comparable across methods**, and no cross-method
ranking here should be treated as settled until every baseline is re-run under
one protocol.

---

## DAETF-Net

Targets spectral fidelity under domain shift rather than another decimal of
in-domain PSNR.

| Component | Mechanism | Verified by |
|---|---|---|
| EFE | p4 group-equivariant convolutions | `max|rot90(f(x)) − f(rot90(x))| = 7.6e-06` |
| TSSE | Tucker contraction against a learned core | gradient reaches the core |
| AF-MoE | per-pixel top-k routing + load balancing | routing maps |
| FDRM | Haar DWT, learnable shrinkage, exact IDWT | `max|IDWT(DWT(x)) − x| = 2.4e-07` |
| DAE | degradation code conditioning via FiLM | auxiliary regression head |
| BPU | back-projection upsampler replacing bicubic | ablation |

**The central idea.** The loss carries two physics terms, `‖Down(ŷ) − LR‖` and
`‖SRF(ŷ) − MSI‖`, that need no ground truth. They hold on any scene from any
sensor, so the objective that trains the model can also **adapt it at test time
on an unlabelled dataset**. The SRF is recovered from the data by least squares
rather than hand-picked (recovery error 2.3e-08 on synthetic data with a known
response).

2.06 M parameters. Runs on a single P100 (16 GB).

Design and honest limitations: [`proposal1/docs/ARCHITECTURE.md`](proposal1/docs/ARCHITECTURE.md).

---

## Running it

### On Kaggle (intended path)

1. Upload [`proposal1/notebooks/DAETF_Net_Kaggle_P100.ipynb`](proposal1/notebooks/DAETF_Net_Kaggle_P100.ipynb)
2. Attach the CAVE and Harvard datasets, enable the GPU
3. Run all — the notebook is self-contained (no clone, no internet needed)

Set `QUICK = True` in the config cell to validate the whole pipeline in a few
minutes before committing to a full run.

### Locally

```bash
pip install -r requirements.txt
cd proposal1

python -c "import daetf; daetf.selfcheck.run_all()"      # verify the mechanisms

python - <<'PY'
import daetf
cfg = daetf.Config().resolve()          # discovers datasets, infers band counts
model, hist = daetf.train(cfg)
daetf.evaluate_dataset(model, cfg.source_root, cfg)
PY
```

Datasets are found automatically under `/kaggle/input/*`, `./data/*` or the
current directory. Override with `DAETF_DATA_ROOTS` (os.pathsep separated) or
`Config(source_root=...)`. Nothing is hardcoded — band counts are read from the
files.

### Automating Kaggle runs

```bash
python tools/kaggle_autorun.py \
  --notebook proposal1/notebooks/DAETF_Net_Kaggle_P100.ipynb \
  --dataset nikeshreddypatlolla/cave-dataset-2 \
  --dataset nikeshreddypatlolla/harvard-hsi-2
```

Pushes, polls, and on failure pulls the log, matches it to a repair rule
(missing module, CUDA OOM, fp16 overflow, DataLoader worker crash, dataset not
found, run-time limit), patches the notebook and re-runs. Every attempt is
archived for auditing.

Credentials come from `KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`
— never from a file in this repo.

### Regenerating the notebook

```bash
python tools/build_notebook.py
```

The notebook is generated from `proposal1/daetf/`, so edit the package and
regenerate rather than editing cells.

---

## Reproducibility

- Every architectural claim has a numerical self-check that runs in seconds
  before training starts (`daetf/selfcheck.py`)
- One metric implementation for all methods and datasets, fixed `data_range=1.0`
  (`daetf/metrics.py`)
- Per-scene results are retained, so comparisons are paired: Wilcoxon
  signed-rank + Cohen's *d*, bootstrap 95% CIs (`daetf/experiments.py`)
- Ablations use matched control arms, not deletions, so they measure mechanisms
  rather than lost capacity
- Config, seed, environment and cost (params/GFLOPs/latency/peak memory) are
  written into every results file

---

## Literature survey

```bash
python literature_survey/lit_search.py       # Crossref harvest -> lit_raw.json
python literature_survey/check_oa.py         # Unpaywall OA check -> lit_candidates.json
python literature_survey/lit_multimodal.py   # HSI+LiDAR track
python literature_survey/build_lit_excel.py  # annotated 4-sheet workbook
```

Filters to six high-impact IEEE journals, verifies open-access status per DOI,
and annotates ~43 papers with method family, core mechanism and relevance.
Requires `openpyxl`.

---

## Status

The implementation, evaluation harness and automation are done and tested. The
full training run and the like-for-like baseline re-runs are not — until those
land, this repository makes no claim about beating the state of the art.
