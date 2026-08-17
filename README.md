# Hyperspectral–Multispectral Image Fusion — Q1 research program

The Q1 program was restructured around **identifiability and uncertainty**,
after the literature audit showed the "another fusion CNN" route is crowded
out (range/null decomposition: DDNM, NSDD, LCDM-BAOAB; arbitrary resolution:
CLoRF, IR&ArF; blind joint estimation: PGU-Net, U2K, DTDNML, ...).  Four
papers remain, each a scientific question plus a mathematical object, ranked
by risk:

| | Paper | Question | Object | Headline metric |
|---|---|---|---|---|
| **P2** (priority 1) | Spectral complexity | Can the intrinsic spectral dimensionality of a fusion problem be identified from its observations? | observation-identifiable rank `r_id = rank(R^T U_r)` | `\|r̂ − r_id\|` |
| **P1** (priority 2) | Admissible ambiguity | What does a fusion method know, and what does it invent? | learned admissible manifold `M_θ ⊂ N(A)` | hallucination `H = ‖P_N(X̂−X)‖/(‖P_N X‖+ε)` |
| **P3** | Sensor-independent field | Can one scene field be fused for *any* sensor? | continuous field `F(x,y,λ)` with sensor operators `O_s` | hold-out-sensor gap `Δ_sensor = E_unseen − E_seen` |
| **P4** (capstone) | Joint identifiability | When is the fusion problem identifiable *at all*? | identifiability phase diagram (I → W → N) | regime, `r_id_hat/r`, `‖P_N X‖/‖X‖` |

**Common spine:** uncertainty / ambiguity quantification — `U(x,y,λ)` (P1),
`r_id` (P2), `U_sensor` (P3), `I(X,D,R,E,A)` (P4).  Evaluation philosophy:
observation consistency, spectral fidelity (SAM/ERGAS per `PROTOCOL_AUDIT.md`),
generalisation gaps, identifiability — never a single inflated headline.

## Current state: all four non-neural scaffolds complete and self-checked

The program's falsifiable foundations are implemented, verified (ALL PASS),
committed and pushed.  Each scaffold contains no networks — every claim is
against the operator algebra, so failures are caught before any training.

| Scaffold | What it proves | Selfcheck |
|---|---|---|
| `proposal2/rankest` + `metrics` + `theory` | `r_id` is recoverable from observations: exact `\|Δr\|=0` for r=3..30; monotone in SNR (12→5) and MSI bands; SRF-overlap tracking 12→12→11→6; noise within 3–5% | ALL PASS |
| `proposal1/ambiguity` | joint `A=[D;R]` projector; consistency is an identity (8.7e-6); `H` monotone 1.0/0.5/0.0 with `E_obs` invariant (spread 4e-6); ambiguity map predicts null-ignorant failure (corr 1.0) | ALL PASS |
| `proposal3/field` | one field, many sensors: linear operators 2.6e-7; zero-shot `Δ_sensor ≈ −3e-8`; nearest-band baseline 3.8e-1 (five orders worse); under-specified family `E_unseen 0.95` | ALL PASS |
| `proposal4/identifiability` | phase diagram I/W/N; `r_id_hat` monotone in SNR & M; `null_frac` monotone; the two identifiability views anti-correlate (−0.70) | ALL PASS |

The learned stage of each paper (network `M_θ`, adaptive rank model, neural
field, joint recovery) is built on these scaffolds and runs on Kaggle (GPU).

## Repository layout

```
common/hsifusion/      shared protocol, data, metrics, engine, classical baselines
existing/              the ten benchmarked methods (notebooks, papers, results)
proposal1/  ambiguity/     P1 paper scaffold (A=[D;R], obs/amb, hallucination, U)
            daetf/ slt/    legacy architectures (selfchecked, superseded)
proposal2/  rankest/       P2 paper foundation (generator, r_id selfcheck, results)
            metrics/       identifiable-rank + rank-error metrics
            theory/        rank definitions, estimator, assumptions
            experiments/   4 sweep scripts (python -m)
            krylovnet/ unfoldfusion/   legacy architectures
proposal3/  field/         P3 paper scaffold (SceneField, sensors, delta_sensor)
            continuumfusion/ nsp/      legacy architectures
proposal4/  identifiability/  P4 capstone scaffold (phase-diagram simulator)
            zerofusion/ graphdip/      legacy architectures
proposal5/  manifoldflow/     learned null-space flow (P1 learned-stage groundwork)
            spectralflow/     legacy + RangeNullProjector (the shared operator core)
proposal6/  consistentflow/   legacy architecture
literature_survey/   Crossref -> Unpaywall -> annotated Excel pipeline
tools/               notebook generators, Kaggle push/repair/re-run automation
PROTOCOL_AUDIT.md    evaluation protocol (SAM/ERGAS headline, paired Wilcoxon,
                     bootstrap CIs, >=3 seeds, SOTA_REFERENCE table)
```

Each `proposalN/` README describes its present-work package and legacy ones.
Every package self-checks its own claims numerically.

## Running it

```powershell
# Windows PowerShell (Linux/macOS: replace ';' with ':' in PYTHONPATH)
$env:PYTHONPATH="common;proposal1;proposal2;proposal3;proposal4;proposal5"

# verify each paper scaffold
python -c "import proposal2.rankest as r; r._selfcheck.run_all()"       # P2
python -c "import proposal1.ambiguity as a; a._selfcheck.run_all()"     # P1
python -c "import proposal3.field as f; f._selfcheck.run_all()"         # P3
python -c "import proposal4.identifiability as i; i._selfcheck.run_all()"  # P4

# artefacts
python -m proposal2.experiments.synthetic_rank_sweep   # P2 tables
python -m proposal2.experiments.noise_sweep
python -m proposal2.experiments.band_count_sweep
python -m proposal2.experiments.srf_sweep
python -m proposal4.identifiability.phasediagram       # P4 phase diagram
```

Legacy architecture selfchecks still run the same way (e.g. `python -c
"import daetf; daetf.selfcheck.run_all()"`).

### Kaggle (learned stage, GPU)

Upload the proposal notebook, attach CAVE + Harvard, set **Accelerator → GPU
T4 x2** in the UI (the push API ignores the accelerator field), run `QUICK =
True` first.  Notebooks are self-contained.

## Why one shared library

`common/hsifusion` holds the protocol, data pipeline, metric implementations,
degradation model and classical baselines.  Any model following
`forward(lr_hsi, msi) -> {"out": tensor}` trains and scores identically;
`BaseConfig` carries the protocol fields so differences between methods are
attributable to the method, not to different scale factors — the exact mistake
that made the ten published baselines incomparable.

## Status

The four paper scaffolds and all legacy implementations are complete and
self-checked.  **No results on real CAVE/Harvard data exist yet** — the learned
stage (network `M_θ`, adaptive rank model, neural field, joint recovery) runs
on Kaggle, and every published claim is re-verified under the shared protocol
before any state-of-the-art statement is made.  Until then this repository
makes no claim about beating the state of the art.

Background finding that motivates the program: re-running ten fusion methods
across CAVE/Harvard shows spectral fidelity, not PSNR, is what breaks under
domain shift (see `existing/results/BENCHMARK.md`).