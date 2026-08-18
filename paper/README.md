# Research Paper — Unrolled Krylov Fusion with Spectral-Rank Control

This folder contains the manuscript and supporting artifacts for the
cross-sensor HSI-MSI fusion study.

## Files

| File | Purpose |
|---|---|
| `research_paper.md` | Full manuscript draft: abstract, intro, related work, KrylovNet, theorems Thm 1–5, benchmark protocol, experiments (tables filled from the Kaggle run), honest SOTA section, proofs, reproducibility appendix |
| `verify_theorems.py` | Numeric verification of Thm 1 (r_id recovery), Thm 2 (rank-fusion error lower bound), Thm 4 (phase transition M*(r) monotone). All PASS. |

## Experiments (results land in `../results_all_datasets.json`)

The experiments run in the Kaggle notebook `MultiDataset_Fusion_Study.ipynb`
(kernel `amarnathmadaka/q1-results-cave-harvard-v2`, version 15):

1. **In-domain, all datasets**: train KrylovNet (2000 iters, 2.3k params)
   on CAVE / Harvard / Chikusei / Pavia train splits; test on each dataset's
   test split. Baselines (Bicubic / GSA / Subspace-LS) on the same pairs.
2. **Cross-domain zero-shot**: CAVE-trained model → Harvard Test and
   Harvard-trained model → CAVE Test (both 31-band, 400–700 nm).
3. **r_id analysis**: per-scene observation-identifiable rank (Thm 1 estimator).
4. **Ambiguity audit**: H score (Thm 3) vs SAM error across methods.

## Protocol (fully specified, reproducible)

- LR-HSI = Gaussian blur (σ=1.2, 9×9) + x4 decimation of the HR-HSI.
- HR-MSI = 3-band Gaussian SRF (centers 0.30/0.55/0.78, width 0.10,
  columns sum to 1) applied to the HR-HSI.
- Metrics: PSNR/SSIM/SAM/ERGAS at data_range=1.0.

## How to reproduce

1. Run `MultiDataset_Fusion_Study.ipynb` on Kaggle with the four datasets
   attached (CAVE, Harvard-hsi-2, Chikusei, PaviaU).
2. Results are saved to `results_<dataset>.json`,
   `results_cross_domain.json`, `results_all_datasets.json`.
3. Fill the placeholder tables in §7 of the manuscript from these files.

## Honesty rule

Published SOTA numbers are shown **only as context** with an explicit
protocol warning. Our numbers are compared **only** against same-protocol
baselines. We do not claim a PSNR record; the contribution is the theory
(identifiability, ambiguity, sensor-shift) plus a reproducible
cross-sensor benchmark.