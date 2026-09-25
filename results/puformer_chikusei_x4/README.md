# PUFormer on Chikusei ×4: first full run

Kaggle kernel [`puformer-chikusei-x4`](https://www.kaggle.com/code/amarnathmadaka/puformer-chikusei-x4),
run on 2026-09-24 on a Tesla T4. It used commit `548c3df`, which is in the old repository
(remote `old-origin`), and ran code from the `chikusei_sota/` directory. That code now lives in
`methods/puformer/`. Protocol: TIP'26 (WorldView-2 8-band SRF, Gaussian blur 7×7 σ=2,
×4 decimation, 8 test tiles of 256×256). The full protocol is in
[methods/puformer/README.md](../../methods/puformer/README.md).

| File | Contents |
|---|---|
| `puformer/results.json` | Final test metrics, per-image metrics, bicubic reference, arguments |
| `puformer/history.json` | Validation metrics every 2,000 iterations |
| `puformer/gaps.json` | Blur, SRF and noise robustness, operator swap, observation consistency |
| `ssrnet/` | SSRNet anchor, trained and tested under the same protocol |
| `tip26_table4_comparison.csv` | All Chikusei rows of TIP'26 Table IV, plus PUFormer |
| `kaggle_run.log` | Full notebook log |

## Training

| | PUFormer | SSRNet |
|---|---|---|
| Parameters | 10.13 M | 0.44 M |
| Iterations | 88,946 (9.5 h wall-clock budget) | 30,000 |
| Batch / patch | 8 × 64² | 16 × 64² |
| LR schedule | AdamW 3e-4, 1k warmup, cosine to the time budget | AdamW 1e-3, cosine |
| Loss | L1 + 0.1·L1 on intermediate stages | L1 |
| Weights tested | best-validation EMA (val PSNR 58.60) | best-validation EMA |
| Inference | 0.62 s per 256² tile | – |

Patches are drawn at random, so there are no epochs. The training area is 1632×2048 pixels,
about 816 non-overlapping 64² patches, so one epoch-equivalent is about 100 iterations at batch 8.
The run is therefore about 870 epoch-equivalents. Validation PSNR stopped improving at about
80k iterations, once the learning rate reached its floor.

## Comparison with TIP'26 Table IV (Chikusei ×4)

All rows except PUFormer are copied from Table IV of the TIP'26 paper (Chen et al., IEEE TIP
vol. 35, 2026, p. 8987). Best result per column in **bold**.

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | CC ↑ | RMSE (DN) ↓ |
|---|---|---|---|---|---|---|
| GSA | 47.5769 | 0.9933 | 1.7216 | 2.1667 | 0.9894 | 70.54 |
| CNMF | 44.5291 | 0.9891 | 1.7259 | 2.8676 | 0.9828 | 101.64 |
| FUSE | 41.1550 | 0.9787 | 2.5814 | 3.4710 | 0.9736 | 148.10 |
| AMSF | 49.3827 | 0.9944 | 1.4852 | 2.3943 | 0.9879 | 56.78 |
| U2Net | 50.5455 | 0.9956 | 1.2389 | 1.8740 | 0.9920 | 49.97 |
| PSRT | 50.6722 | 0.9952 | 1.2733 | 2.0107 | 0.9906 | 49.00 |
| DCT | 52.1998 | 0.9977 | 0.9239 | 1.7448 | 0.9935 | 41.45 |
| SSDT | 52.4788 | 0.9975 | 0.8768 | 1.7992 | 0.9928 | 40.00 |
| OTPNet | 52.8243 | 0.9969 | 1.0201 | 1.9332 | 0.9916 | 38.49 |
| DPFormer | 53.1632 | 0.9970 | 0.9483 | 1.8795 | 0.9923 | 37.18 |
| MIMFormer | 53.3578 | 0.9973 | 0.9094 | 1.7611 | 0.9931 | 35.90 |
| DDIF | 53.4421 | 0.9972 | 0.9387 | 1.8087 | 0.9929 | 35.58 |
| ASMNet | 53.4528 | 0.9972 | 0.9472 | 1.9153 | 0.9917 | 35.89 |
| MIMOSST | 53.6073 | 0.9972 | 0.9315 | 1.8119 | 0.9926 | 34.93 |
| SMGU | 54.5244 | 0.9977 | 0.8719 | 1.6405 | 0.9939 | 31.29 |
| CLSNet | 55.2007 | 0.9979 | 0.7836 | 1.6318 | 0.9941 | 29.06 |
| Two-Stage Diffusion (TIP'26, previous SOTA) | 56.1871 | **0.9982** | **0.7203** | 1.5158 | 0.9948 | 25.73 |
| **PUFormer (ours)** | **57.9943** | 0.9969 | 0.7281 | **1.3561** | **0.9966** | **19.35** |

Our rank out of 18: **1st** on PSNR (+1.81 dB over TIP'26), ERGAS (−10.5 %), CC and RMSE
(−24.8 %); **2nd** on SAM (+0.008°); about **11th** on SSIM.

Our own runs under the same protocol:

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | CC ↑ | RMSE (DN) ↓ |
|---|---|---|---|---|---|---|
| Bicubic | 33.6333 | 0.7404 | 4.4061 | 9.0115 | 0.8408 | 321.81 |
| GSA (training-free) | 43.1966 | 0.9778 | 2.7300 | 3.2791 | 0.9905 | 107.69 |
| SSRNet | 49.9841 | 0.9922 | 1.7606 | 2.4684 | 0.9911 | 48.83 |
| PUFormer | 57.9943 | 0.9969 | 0.7281 | 1.3561 | 0.9966 | 19.35 |

### Is the PSNR comparable?

The TIP'26 PSNR uses a fixed, dataset-level peak. For every Chikusei row of Table IV,
`RMSE · 10^(PSNR/20)` gives a nearly constant peak of **16.6k–17.1k DN**, from GSA to their
own model (last column of the CSV). A per-image peak (PSRT-style) would not give a constant.
Our headline PSNR uses the global max of our crop (15,133 DN). Recomputed with their peak of
about 16.7k DN, PUFormer would score about 58.7 dB. RMSE in DN does not depend on the peak at
all, and there PUFormer leads by 25 %.

Our other PSNR definitions, for completeness: PSRT-style (per-image peak) 52.56 dB, band-wise
mean with peak 1 65.29 dB.

### Caveats

1. **Test tiles.** Our 8 test tiles follow the protocol as the paper describes it: the top 256
   rows of the central 2048² crop. They may not be the exact pixels TIP'26 used.
2. **SSIM definition.** The paper does not say how SSIM is computed. Ours is band-wise
   Gaussian SSIM with data range = the image max, which tends to score lower than other
   implementations. Part of the SSIM gap may come from that.
3. **GSA calibration.** Our GSA scores 43.20 dB, and TIP'26 reports 47.58 dB. Our setup is
   therefore not easier for GSA. GSA implementations differ, though, so this is only a rough check.

## Per-image test results (PUFormer)

| Tile | PSNR | PSNR (PSRT) | SSIM | SAM | ERGAS | RMSE (DN) |
|---|---|---|---|---|---|---|
| 0 | 58.77 | 58.77 | 0.999 | 0.666 | 1.602 | 17.44 |
| 1 | 60.06 | 54.22 | 0.998 | 0.667 | 0.999 | 15.03 |
| 2 | 60.60 | 55.67 | 0.999 | 0.619 | 1.137 | 14.13 |
| 3 | 56.96 | 54.34 | 0.998 | 0.879 | 1.350 | 21.47 |
| 4 | 57.20 | 49.38 | 0.995 | 0.734 | 1.313 | 20.88 |
| 5 | 57.39 | 50.55 | 0.996 | 0.679 | 1.371 | 20.44 |
| 6 | 56.15 | 49.58 | 0.995 | 0.841 | 1.449 | 23.58 |
| 7 | 56.83 | 47.96 | 0.995 | 0.740 | 1.627 | 21.81 |

## Robustness and operator swap (`gaps.json`)

The model is trained on σ=2 and the nominal SRF, then tested on mismatched degradations.
"Fixed" keeps the training operator inside the network. "Swap" gives the network the true
test-time operator, with no retraining. PSNR in dB.

| Test condition | GSA | PUFormer fixed | PUFormer swap |
|---|---|---|---|
| Nominal | 43.20 | 58.00 | – |
| Blur σ = 1.5 | 44.05 | 53.46 | **55.17** (+1.71) |
| Blur σ = 2.5 | 42.77 | 55.85 | **57.13** (+1.28) |
| Blur σ = 3.0 | 42.54 | 54.05 | **56.14** (+2.09) |
| SRF shift −8 nm | 42.86 | 52.03 | 52.46 (+0.43) |
| SRF shift +8 nm | 43.29 | 51.88 | 52.36 (+0.48) |
| Noise 40 dB SNR | 43.04 | 53.34 | – |
| Noise 35 dB SNR | 42.71 | 49.55 | – |
| Noise 30 dB SNR | 41.84 | 45.11 | – |

Observation consistency at nominal settings: PUFormer reproduces the HR-MSI at 74.95 dB and
the LR-HSI at 42.46 dB. GSA gets 45.00 and 40.36 dB.

Findings:
* Swapping in the true blur recovers 1.3–2.1 dB. Swapping the SRF gains only about 0.45 dB,
  and ERGAS gets worse, so SRF mismatch needs more than an operator swap.
* The model was trained without noise and loses 12.9 dB at 30 dB SNR. It still beats GSA,
  but noise-aware training is an open item.

## Next steps

1. Recompute SSIM on the saved predictions with other common definitions (data range 1,
   skimage/PSRT style) to see how much of the SSIM gap is definitional.
2. Add a SAM + SSIM loss term and ×8 self-ensemble at test time.
3. Ablate the starting estimate: bicubic, iterative back-projection, or a learned
   MSI-guided back-projection upsampler (DBPN-style).
