# Research Paper — Unrolled Krylov Fusion with Spectral-Rank Control

This folder contains the manuscript and supporting artifacts for the
cross-sensor HSI-MSI fusion study.

## Files

| File | Purpose |
|---|---|
| `research_paper.md` | Full manuscript draft: abstract, intro, related work, KrylovNet, theorems Thm 1–5, benchmark protocol, experiments (tables filled from the Kaggle run), honest SOTA section, proofs, reproducibility appendix |
| `verify_theorems.py` | Numeric verification of Thm 1 (r_id recovery), Thm 2 (rank-fusion error lower bound), Thm 4 (phase transition M*(r) monotone). All PASS. |
| `fill_tables.py` | Prints manuscript tables from the run's JSON outputs (also `kaggle_runs/multidataset/out/results_*.json` for the completed run). |

## Experiments (results land in `../results_all_datasets.json`)

The experiments run in the Kaggle notebook `MultiDataset_Fusion_Study.ipynb`
(kernel `amarnathmadaka/multidataset-hsi-msi-fusion-study`, version 1,
COMPLETE — re-run of the same code previously at
`amarnathmadaka/q1-results-cave-harvard-v2` v16):

1. **In-domain, all datasets**: train KrylovNet (2000 iters, 2.3k params)
   on CAVE / Harvard / Chikusei / Pavia train splits; test on each dataset's
   test split. Baselines (Bicubic / GSA / Subspace-LS) on the same pairs.
   → CAVE 40.85, Harvard 70.78, Chikusei 43.69, PaviaU 34.48 dB (PSNR),
   beating all baselines on all datasets.
2. **SOTA push (§8.2)**: KrylovNet-P (1.38M params, learned proximal
   prior, EMA, physics loss) trained on CAVE ×4 under the published
   protocol (Wald + Nikon D700 SRF). Notebook:
   `proposal2/notebooks/krylovnet_SOTA_CAVE_Nikon.ipynb` (Kaggle,
   checkpoint-and-resume via the `amarnathmadaka/krylovnet-cp` dataset).
   Target: FeINFN 52.47 dB. Status: in training (GPU quota refresh
   2026-08-22).
2. **Cross-domain zero-shot**: CAVE-trained model → Harvard Test and
   Harvard-trained model → CAVE Test (both 31-band, 400–700 nm).
   → 70.72 / 40.85 dB; no sensor shift (shared SRF) ⇒ no drop (Thm 5).
3. **r_id analysis**: per-scene observation-identifiable rank (Thm 1 estimator).
   → CAVE/Harvard 0.0, Chikusei/PaviaU 2.0 (≤ M = 3 bands, Thm 4).
4. **Ambiguity audit**: H score (Thm 3) vs SAM error across methods.
   → KrylovNet H = 0.19-0.28 (< 1, under-fills null space); Bicubic/GSA
   H ≈ 1.06-1.51 (> 1, over-fill); lowest H ⇒ lowest SAM on 4/4 datasets.
5. **Phase transition (P4)**: r̂_id(M) monotone and capped by M on all
   datasets.
6. **Sensor shift (P3)**: sensor EMD = 0 (same simulated SRF),
   scene spectral EMD (CAVE↔Harvard) = 0.116.

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
baselines. The one exception is §8.2: CAVE ×4 under the Nikon D700
protocol is the single setting where a direct head-to-head with the
published record (FeINFN 52.47 / BDT 52.30) is possible, and
KrylovNet-P is run under exactly that protocol. The contribution is the
theory (identifiability, ambiguity, sensor-shift), the 2026 protocol
census (`../SOTA_COMPARISON.md`), and a falsifiable protocol-matched
PSNR attempt.