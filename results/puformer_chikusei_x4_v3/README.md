# PUFormer v3 on Chikusei ×4: fine-tune with exact border physics, SAM + SSIM loss, 2 GPUs

Kaggle kernel [`puformer-chikusei-x4-v3`](https://www.kaggle.com/code/amarnath10chinu/puformer-chikusei-x4-v3),
2026-09-25/26, 2 × Tesla T4, 11.03 h. The notebook embeds its code (sha256 `e0e35e17…`, built from
`8003f49` plus the v3 changes in `methods/puformer`). Protocol: TIP'26 (WorldView-2 8-band SRF,
Gaussian 7×7 σ = 2, ×4, 8 test tiles of 256²), identical to the first run.

| File | Contents |
|---|---|
| `puformer/results.json` | Final test (single pass and 2-fold self-ensemble), per image, Q2n / SCC, consistency |
| `puformer/history.json` | Validation (selection) and test (logged only) metrics every 2,000 iterations |
| `puformer/gaps.json` | Blur / SRF mismatch with operator swap, noise, observation consistency |
| `first_run_reeval/results.json` | The first run's 57.99 dB checkpoint re-scored with the full v3 metric set |
| `kaggle_run.log` | Full notebook log |

## Result against TIP'26 Table IV (Chikusei ×4)

Weights: best-validation EMA (val PSNR 58.722, iteration 44,000). Selection used validation only.

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Q2n ↑ | CC ↑ | SCC ↑ | RMSE (DN) ↓ |
|---|---|---|---|---|---|---|---|---|
| SMGU (PR'25) | 54.5244 | 0.9977 | 0.8719 | 1.6405 | 0.9952 | 0.9939 | 0.9875 | 31.29 |
| CLSNet | 55.2007 | 0.9979 | 0.7836 | 1.6318 | 0.9943 | 0.9941 | 0.9869 | 29.06 |
| Two-Stage Diffusion (TIP'26, previous best) | 56.1871 | 0.9982 | 0.7203 | 1.5158 | **0.9969** | 0.9948 | 0.9891 | 25.73 |
| PUFormer first run (re-scored) | 57.9944 | 0.9989 | 0.7281 | 1.3560 | 0.9929 | 0.9966 | 0.9991 | 19.35 |
| **PUFormer v3, single pass** | 58.1467 | **0.9989** | 0.7171 | 1.3367 | 0.9930 | 0.9967 | 0.9992 | 19.06 |
| **PUFormer v3, self-ensemble** | **58.2016** | **0.9989** | **0.7126** | **1.3304** | 0.9931 | **0.9967** | **0.9992** | **18.94** |

SSIM column: PSRT's `cal_ssim.py` (data range 1 = the dataset peak), the definition consistent with
the fixed-peak PSNR that TIP'26 uses. With our stricter definition (data range = each image's own
max), v3 scores 0.9970 (single pass 0.9969), below TIP'26's 0.9982. TIP'26 does not say which it used.

Rank out of 18 (17 TIP'26 rows + ours), self-ensemble and single pass alike: **1st on PSNR, SSIM
(PSRT definition), SAM, ERGAS, CC, SCC and RMSE; 10th on Q2n**. `methods/puformer/compare_tip26.py`
regenerates the full 18-row table.

Against the first run, v3 gains +0.15 dB PSNR (single pass), −0.011° SAM, −1.4 % ERGAS and
−1.5 % RMSE. Observation consistency rises from 76.7 to 85.5 dB (LR-HSI) and from 75.0 to 89.0 dB
(HR-MSI), because the physics step now uses the exact simulation operator.

## Q2n: why it is the one metric we do not beat

Q2n (Vivone's `q2n.m`, 32×32 blocks, ported line by line and checked against a literal port of
`onion_mult2D.m`) normalises every band of every block by the ground truth's own standard deviation
in that block. On our test tiles, the deficit is set by bands that no method can reconstruct well.

* **83 % of the Q2n deficit sits in bands 0–7 (363–404 nm).** They lie below WorldView-2's first MS
  band (5 % edge at 396 nm), so the HR-MSI carries no detail for them. If those 8 bands were exact,
  Q2n would be 0.9988; if bands 104–127 were exact as well, 0.9998.
* Their fine detail is essentially unpredictable. An oracle least-squares injection of the MSI detail,
  fitted on the test ground truth itself, reduces the band-0 error only from 61.6 DN (bicubic) to 57.5
  DN. PUFormer reaches 52.7 DN, and beats that oracle in every band (band 4: 24.6 vs 49.1 DN). The
  residual is white, with no stripe structure.
* Under this implementation, Q2n barely moves with fidelity above 50 dB. Degrading our own
  prediction gives 0.9902 at 52.1 dB, 0.9921 at 55.5 dB and 0.9929 at 58.0 dB. TIP'26 lists 0.9969 at
  56.19 dB, and U2Net 0.9928 at 50.55 dB. Their Q2n was therefore most likely computed differently, or
  on tiles with fewer flat UV-dominated blocks. We report ours as computed.
* A coarser intensity scale before the `uint16()` step of q2n.m lowers Q2n (0.9863 at ×255), so
  quantization does not explain the gap either.

The strict SSIM gap has the same root: dark tiles 4–7 score 0.995 under data range = tile max.

## Training

| | first run | v3 |
|---|---|---|
| Start | from scratch | first run's best EMA (57.99 dB) |
| Physics border | zero padding | reflect, the exact simulation operator + exact adjoint |
| Loss | L1 + 0.1 L1 (stages) | + 0.05 SAM (radians) + 0.1 (1 − SSIM) |
| GPUs / batch | 1 × 8 | 2 × 8 (DDP), 39 samples/s |
| Iterations | 88,946 (9.5 h) | 94,450 (10.9 h), cosine to the deadline, peak lr 1.5e-4 |
| Host memory | – | flat at 2.4 GB RSS for the whole run |

The warm start scores 49.5 dB val / 54.2 dB test at iteration 0 under the new reflect physics,
because its prior had learned to undo the old zero-padding border error. It recovers within 2,000
iterations (58.60 val / 58.06 test). Validation peaks at 58.722 (iteration 44,000); test stays at
58.13–58.15 dB from iteration 26,000 onwards. More iterations of this recipe would not help.

## Why the v2 run was killed

v2 (`dc89aa8`, same account) trained with `nn.DataParallel` over both T4s and scored every
evaluation in float64 on the CPU (295 s for the bicubic reference alone). It printed nothing for
6,200 s after the bicubic line, saved no checkpoint, and was then killed (SIGKILL). So it never
reached iteration 2,000, which v3 reaches in about 14 minutes on the same hardware. v3 removes DataParallel and uses DDP with a speed
probe and a one-GPU fallback. It also evaluates on the GPU, logs speed and memory every 500
iterations, and stops at a deadline inside Kaggle's 12 h limit.

## Robustness (`puformer/gaps.json`, PSNR in dB)

| Test condition | GSA | first run fixed | first run swap | v3 fixed | v3 swap |
|---|---|---|---|---|---|
| Nominal | 43.20 | 58.00 | – | 58.15 | – |
| Blur σ = 1.5 | 44.05 | 53.46 | 55.17 | 50.68 | **57.97** |
| Blur σ = 2.5 | 42.77 | 55.85 | 57.13 | 54.26 | **58.05** |
| Blur σ = 3.0 | 42.54 | 54.05 | 56.14 | 51.69 | **57.89** |
| SRF shift −8 nm | 42.86 | 52.03 | 52.46 | **53.01** | 52.54 |
| SRF shift +8 nm | 43.29 | 51.88 | 52.36 | **53.03** | 52.47 |
| Noise 40 / 35 / 30 dB | 43.04 / 42.71 / 41.84 | 53.34 / 49.55 / 45.11 | – | 53.03 / 49.11 / 44.62 | – |

With exact physics, swapping in the true blur now recovers almost all of the nominal quality: within
0.26 dB for σ from 1.5 to 3.0 (before: 1.3–2.1 dB short). Without the swap, v3 relies more on the
nominal blur than the first run did. SRF mismatch still needs more than an operator swap, and noise
robustness is unchanged.

Correction: the first run's consistency numbers (LR-HSI 42.46 dB, GSA 40.36 dB) measured D X̂ with
zero padding against LR-HSI simulated with reflect padding, so they were dominated by that border
mismatch. With the matching operator, the first run's checkpoint scores 76.7 dB, and GSA 45.4 dB.

## Per-image test results (v3, self-ensemble)

| Tile | PSNR | SSIM (strict) | SSIM (PSRT) | SAM | ERGAS | Q2n |
|---|---|---|---|---|---|---|
| 0 | 59.15 | 0.9994 | 0.9994 | 0.644 | 1.550 | 0.9957 |
| 1 | 60.45 | 0.9985 | 0.9994 | 0.643 | 0.972 | 0.9967 |
| 2 | 60.97 | 0.9987 | 0.9995 | 0.597 | 1.093 | 0.9962 |
| 3 | 57.10 | 0.9981 | 0.9988 | 0.862 | 1.334 | 0.9917 |
| 4 | 57.29 | 0.9951 | 0.9986 | 0.726 | 1.305 | 0.9929 |
| 5 | 57.51 | 0.9957 | 0.9987 | 0.669 | 1.357 | 0.9913 |
| 6 | 56.24 | 0.9953 | 0.9984 | 0.830 | 1.449 | 0.9913 |
| 7 | 56.91 | 0.9947 | 0.9986 | 0.729 | 1.584 | 0.9889 |

Caveats from the first run still apply: our test tiles follow the protocol as TIP'26 describes it but
may not be their exact pixels, and every reported row except ours is copied from their table.
