# Benchmark of existing HSI-MSI fusion methods

Results recorded by the notebooks in [`existing/notebooks/`](../notebooks/),
read from their saved cell outputs.

## Read this first

**These numbers are not comparable across methods.** Each notebook was written
independently and they disagree on nearly every protocol choice. They are
reported here because they are what was actually measured, and because the
disagreement is itself the finding that motivates
[proposal1](../../proposal1/).

Specific incompatibilities:

| Issue | Detail |
|---|---|
| Scale factor | ×4 (Fusformer, LRU, AMGSGAN), ×8 (TSFN, DHIF-Net), ×16 (DBIN), ×32 (MoG-DCN, PSRT, UTAL) |
| Normalisation | some divide by the per-image maximum, some do not; MoG-DCN-Harvard uses `data_range=gt.max()` for PSNR |
| ERGAS scale | hardcoded per notebook, not always matching the actual downsampling |
| Test set | IFCASformer runs a CASSI mask task on `scene01…scene27` from a different dataset (`cave-dataset-3`), not the 12-scene split |
| Sample count | AMGSGAN evaluates **one** 128×128 image after resizing; others use 12 or 20 full scenes |
| Weights | LRU and AMGSGAN train in-notebook; the rest load released checkpoints |

## CAVE (12 test scenes, 512×512×31 — except where noted)

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Scale | Note |
|---|---|---|---|---|---|---|
| Fusformer | 50.20 | 0.9996 | 2.35 | 0.85 | ×4 | tiled 64×64 inference |
| DHIF-Net | 48.80 | 0.9966 | 2.16 | 0.51 | ×8 | |
| DBIN | 47.14 | 0.9939 | 2.97 | 0.33 | ×16 | TensorFlow, via hif-benchmarking |
| TSFN | 46.40 | 0.9943 | 2.75 | 0.63 | ×8 | TensorFlow |
| UTAL | 41.37 | 0.9906 | 4.62 | 0.27 | ×32 | |
| MoG-DCN | 38.55 | 0.9715 | 6.62 | 0.38 | ×32 | |
| PSRT | 38.24 | 0.9647 | 7.55 | 0.39 | ×32 | |
| AMGSGAN | 37.42 | — | 7.41 | 2.00 | ×4 | **1 test image**, trained in-notebook |
| LRU | 37.16 | 0.9768 | 3.73 | 4.07 | ×4 | trained in-notebook, 200 epochs |
| IFCASformer | 35.98 | 0.9602 | 5.15 | 3.55 | CASSI | 10 scenes, different dataset |

## Harvard (20 test scenes, 1040×1392×31)

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Scale |
|---|---|---|---|---|---|
| DHIF-Net | 70.20 | 0.9997 | 2.62 | 0.93 | ×8 |
| PSRT | 57.25 | 0.9918 | 15.87 | 1.81 | ×32 |
| TSFN | 56.35 | 0.9935 | 8.29 | 4.60 | ×8 |
| UTAL | 48.60 | 0.9511 | 36.46 | 16.60 | ×32 |
| IFCASformer | 44.73 | 0.8713 | 26.86 | 85.41 | CASSI |
| LRU | 38.76 | 0.7320 | 7.77 | 80.04 | ×4 |
| AMGSGAN | 31.86 | — | 5.94 | 7.16 | ×4 |
| MoG-DCN | 30.11 | 0.9892 | 14.60 | 4.64 | ×32 |
| Fusformer | 25.80 | 0.3059 | 58.89 | 302.39 | ×4 |

DBIN's Harvard run completed but its averaged output was truncated in the saved
cell; per-image values are around 46-56 dB PSNR with SAM 7-26.

## The actual finding

The original README claimed a universal CAVE→Harvard collapse and cited
Fusformer's PSNR drop (50.20 → 25.80). **The data does not support the general
claim as stated.** Five of nine methods score *higher* PSNR on Harvard than on
CAVE — DHIF-Net 48.8 → 70.2, PSRT 38.2 → 57.2, TSFN 46.4 → 56.4, UTAL 41.4 →
48.6. Per-image maximum normalisation on Harvard's darker scenes inflates PSNR.

What genuinely degrades is **spectral fidelity**:

| Method | SAM CAVE | SAM Harvard | Δ | ERGAS CAVE | ERGAS Harvard |
|---|---|---|---|---|---|
| DHIF-Net | 2.16 | 2.62 | +0.46 | 0.51 | 0.93 |
| AMGSGAN | 7.41 | 5.94 | −1.47 | 2.00 | 7.16 |
| LRU | 3.73 | 7.77 | +4.04 | 4.07 | 80.04 |
| TSFN | 2.75 | 8.29 | +5.54 | 0.63 | 4.60 |
| MoG-DCN | 6.62 | 14.60 | +7.98 | 0.38 | 4.64 |
| PSRT | 7.55 | 15.87 | +8.32 | 0.39 | 1.81 |
| IFCASformer | 5.15 | 26.86 | +21.71 | 3.55 | 85.41 |
| UTAL | 4.62 | 36.46 | +31.84 | 0.27 | 16.60 |
| Fusformer | 2.35 | 58.89 | +56.54 | 0.85 | 302.39 |

Seven of nine methods lose 4-57 degrees of spectral accuracy, and ERGAS rises by
up to two orders of magnitude, while PSNR often improves. **PSNR is the wrong
headline metric for this problem**, which is why proposal1 optimises SAM
directly and reports all four metrics with a fixed data range.

## Reproducing

Each notebook runs on Kaggle with a T4/P100, the relevant dataset attached, and
the method's checkpoint as a model source. Dataset and model slugs are listed in
[`../papers/INDEX.md`](../papers/INDEX.md).

To make these numbers genuinely comparable, each method needs re-running under
the single protocol in
[`proposal1/docs/ARCHITECTURE.md` §5](../../proposal1/docs/ARCHITECTURE.md#5-evaluation-protocol):
one scale factor, one metric implementation with fixed `data_range`, one
degradation, full scenes. That work is not yet done, and no cross-method claim
in this repository should be treated as settled until it is.
