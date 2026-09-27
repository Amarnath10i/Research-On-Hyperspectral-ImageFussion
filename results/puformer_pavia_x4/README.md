# PUFormer on Pavia Centre ×4: trained from scratch, exact border physics, 2 GPUs, one notebook

Kaggle kernel [`puformer-pavia-x4`](https://www.kaggle.com/code/amarnath10chinu/puformer-pavia-x4),
2026-09-26/27, 2 × Tesla T4, one 12-hour session that trains, tests and runs the robustness study
(10.19 h of training, 93,850 iterations). The notebook embeds its code (sha256 `0d5e9027…`, identical to
`methods/puformer` at commit `5e5d772`). The analysis files come from the CPU kernel
[`puformer-pavia-analysis`](https://www.kaggle.com/code/amarnath10chinu/puformer-pavia-analysis), which reads
this run's saved test predictions.

## Protocol

TIP'26 takes its Pavia Centre data from the benchmark files released with PSRT. Those files are only on
Baidu Netdisk, which we could not reach, so we rebuilt the benchmark from the original scene to the same
specification (Kaggle dataset `mlxlx0000/paviadata`, `Pavia.mat`, 1096 × 715 × 102):

| Item | Setting |
|---|---|
| HR-HSI | bands 11–102 (92 bands, 473–860 nm if the 102 bands are evenly spaced over 430–860 nm), ÷ 8,000 DN (their maximum) |
| LR-HSI | Gaussian 7 × 7, σ = 2, ×4 decimation, reflect border |
| HR-MSI | IKONOS, 4 bands (blue 445–516, green 506–595, red 632–698, NIR 757–853 nm, 50 % edges) |
| Test | the four 256 × 256 tiles of the top-left 512 × 512 region (image 1 = rows 0–256 / cols 0–256, 2 = rows 0–256 / cols 256–512, 3 = rows 256–512 / cols 0–256, 4 = rows 256–512 / cols 256–512) |
| Validation | 63 patches of 64 × 64 on a 32-pixel grid, rows 528–656 |
| Training | random 64 × 64 crops (dihedral augmentation) from rows 672–1096 and from rows 0–512 × cols 528–715 (97 non-overlapping patch equivalents) |
| Selection | validation PSNR only |

16-pixel gaps separate the regions. As a check on the rebuilt data, our GSA scores 34.78 dB against the
35.35 dB that TIP'26 reports for GSA, so the rebuilt data are not easier.

| File | Contents |
|---|---|
| `puformer/results.json` | Final test (single pass and 2-fold self-ensemble), per image, Q2n / SCC, consistency, GSA and bicubic |
| `puformer/history.json` | Validation (selection) and test (logged only) metrics every 1,000 iterations |
| `puformer/gaps.json` | Blur / SRF mismatch with operator swap, noise, observation consistency |
| `test/results.json` | The same weights re-scored by `test.py` (identical numbers) |
| `paper_analysis/` | Band-wise errors and the Q2n analysis (`visual.npz` is kept locally) |
| `sam_analysis/` | Where the spectral-angle error comes from |
| `tip26_table4_pavia.csv` | TIP'26 Table IV, Pavia Centre rows (reported values) |
| `kaggle_run.log` | Full notebook log |

## Result against TIP'26 Table IV (Pavia Centre ×4)

Weights: best-validation EMA (val PSNR 48.579, iteration 43,000). Selection used validation only.

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Q2n ↑ | CC ↑ | SCC ↑ | RMSE (DN) ↓ |
|---|---|---|---|---|---|---|---|---|
| OTPNet | 45.8081 | 0.9967 | 1.6573 | 0.9465 | **0.9987** | 0.9980 | 0.9968 | 41.51 |
| SMGU | 47.1239 | 0.9972 | 1.4690 | 0.8547 | 0.9981 | 0.9984 | 0.9974 | 37.10 |
| CLSNet | 46.5277 | 0.9969 | 1.5708 | 0.8412 | 0.9983 | 0.9983 | 0.9975 | 36.21 |
| Two-Stage Diffusion (TIP'26, previous best) | 47.1597 | 0.9972 | **1.4542** | 0.8262 | 0.9980 | 0.9985 | 0.9976 | 35.71 |
| **PUFormer, single pass** | 50.6299 | **0.9974** | 1.8955 | 0.6206 | 0.9751 | **0.9994** | **0.9995** | 23.68 |
| **PUFormer, self-ensemble** | **50.6915** | **0.9974** | 1.8833 | **0.6170** | 0.9753 | **0.9994** | **0.9995** | **23.52** |

Rank out of 18 (17 TIP'26 rows + ours): **1st on PSNR, SSIM (PSRT definition), ERGAS, CC, SCC and RMSE**;
13th on SAM and 16th on Q2n. PSNR is 3.53 dB above the best reported value, RMSE 34.1 % lower and ERGAS
25.3 % lower. With the stricter SSIM (data range = each image's maximum) PUFormer scores 0.9970, below the
reported 0.9972. The implied PSNR peak of the TIP'26 rows (RMSE × 10^(PSNR/20)) is 7.7k–8.4k DN, median
8.07k, so their PSNR and ours (peak 8,000 DN) are on the same scale.

Per test image (self-ensemble):

| Image | PSNR | SSIM (strict) | SSIM (PSRT) | SAM | ERGAS | Q2n | RMSE (DN) |
|---|---|---|---|---|---|---|---|
| 1 | 50.48 | 0.9977 | 0.9977 | 1.546 | 0.555 | 0.9987 | 23.95 |
| 2 | 52.30 | 0.9949 | 0.9964 | 2.722 | 0.813 | 0.9044 | 19.40 |
| 3 | 49.53 | 0.9979 | 0.9979 | 1.677 | 0.557 | 0.9991 | 26.70 |
| 4 | 50.45 | 0.9975 | 0.9976 | 1.588 | 0.543 | 0.9991 | 24.01 |
| Mean | 50.69 | 0.9970 | 0.9974 | 1.883 | 0.617 | 0.9753 | 23.52 |

Both lost metrics come from image 2. On images 1, 3 and 4, Q2n is 0.9987–0.9991, at or above the best
reported value, and SAM is 1.55–1.68°.

### Why SAM and Q2n trail: water in image 2 (`sam_analysis/sam.json`, `paper_analysis/q2n.json`)

* **Image 2 contains the river Ticino.** 63 % of its pixels have NDWI > 0 (green 560 nm vs NIR 800 nm),
  against 11 %, 7 % and 13 % in images 1, 3 and 4.
* **Water spectra are noise-dominated.** The per-band noise estimated from the ground truth alone
  (robust spectral second differences) makes an angle of 3.8° with the noise-free spectrum in the darkest
  10 % of the pixels and 0.23° in the brightest 10 %; averaged over the test images this floor is 1.01°
  (1.65° on image 2). Neither observation carries that noise, so no method can reproduce it.
* **PUFormer's SAM is 2.95° over water and 1.56° over all other pixels** (per image without water:
  1.42°, 1.41°, 1.67°, 1.64°). The darkest 10 % of the pixels account for 34 % of its mean SAM. GSA shows
  the same pattern (6.98° over water, 4.70° elsewhere).
* **Q2n:** image 2 scores 0.904, the others 0.9987–0.9991. Image 2 carries 97 % of the summed deficit
  Σ(1 − Q2n), and the worst 10 % of the 32 × 32 blocks carry 96 % of it: nearly flat water blocks, where
  Q2n divides the error by a tiny local variance.
* No test pixel has an all-zero spectrum, so the SAM convention for zero-norm pixels does not matter here.

TIP'26 does not say whether its Pavia test images contain water, so SAM and Q2n cannot be compared
directly on this dataset.

## Robustness (test PSNR / SAM)

| Condition | GSA | PUFormer, training operator | PUFormer, true operator swapped in |
|---|---|---|---|
| Nominal | 34.78 / 5.23 | 50.63 / 1.90 | – |
| Blur σ = 1.5 | 35.59 / 4.99 | 43.80 / 2.75 | **50.56 / 1.90** |
| Blur σ = 2.5 | 34.36 / 5.36 | 47.30 / 2.14 | **50.59 / 1.90** |
| Blur σ = 3.0 | 34.13 / 5.43 | 44.87 / 2.37 | **50.52 / 1.91** |
| SRF shift −8 nm | 34.58 / 5.36 | 49.46 / 1.99 | **49.75 / 2.03** |
| SRF shift +8 nm | 34.79 / 5.20 | 49.62 / 1.99 | **49.82 / 2.02** |
| Noise 40 dB SNR | 34.71 | **47.55** | – |
| Noise 35 dB SNR | 34.57 | **44.36** | – |
| Noise 30 dB SNR | 34.16 | **40.18** | – |

With the true blur swapped in, the model stays within 0.11 dB of its nominal PSNR for σ between 1.5 and
3.0 (the training operator loses 3.3–6.8 dB). Unlike on Chikusei, the swap also raises the PSNR for a
shifted spectral response (+0.2–0.3 dB), although SAM rises slightly (+0.03–0.04°). Observation
consistency of the self-ensemble: 79.9 dB (LR-HSI) and 77.6 dB (HR-MSI).

## Caveats

Our test images follow the protocol as TIP'26 describes it, but they are not its pixels, and for this
dataset the observations are our own simulation, because the PSRT files could not be downloaded. Every
reported row except ours is copied from TIP'26. RMSE in DN is the most comparable metric; SAM and Q2n are
the least, because they depend strongly on whether the test images contain water.
