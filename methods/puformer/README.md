# PUFormer: Physics-Unfolded Transformer for Chikusei HSI–MSI Fusion (×4)

Goal: beat the best **published** Chikusei ×4 result, 56.19 dB PSNR from TIP 2026
(Two-Stage Conditional Diffusion), under that paper's protocol. Literature review:
[docs/CHIKUSEI_SOTA_SURVEY.md](../../docs/CHIKUSEI_SOTA_SURVEY.md). It covers only peer-reviewed, subscription IEEE papers.

## Protocol (TIP'26, Table II / Sec. IV-B)

| Item | Setting |
|---|---|
| Data | Chikusei, central 2048×2048×128 crop, divided by its max (15133 DN) |
| LR-HSI | Gaussian blur 7×7, σ = 2, ×4 decimation (reflect padding per image) |
| HR-MSI | WorldView-2 8-band SRF (DigitalGlobe 5 %-edge table) |
| Test / val / train | 8 × 256² tiles (rows 0–256) / 64 × 64² patches (rows 272–400) / random 64² crops from rows 416–2048 |
| Model selection | validation PSNR only; test is scored once |
| Headline PSNR | 10·log10(1/MSE), i.e. peak = the dataset's global max. TIP'26's RMSE column is in DN, and RMSE·10^(PSNR/20) gives a constant 16.6k–17.1k DN peak across all their Chikusei rows (verified). PSRT-style and band-wise PSNR are reported as well. |

## Method

```
X0 = bicubic(Y_H)
for k = 1..K:
    Z_k = X_{k-1} − η_H·Dᵀ(D X − Y_H) − η_M·Rᵀ(R X − Y_M)       exact physics step (fp32)
    X_k = Z_k + Prior_k([Z_k, Y_M, Dᵀ(D Z_k − Y_H)], memory_{k−1})
```
* `D` and `R` are the exact forward operators, and `Dᵀ` and `Rᵀ` are their exact adjoints
  (inner-product test error < 1e-14).
* The prior is a 3-level U-Net of Restormer blocks. Its transposed attention attends over
  spectral feature channels. Its output layer starts at zero, so training begins from the
  physics solution.
* Decoder features pass from each stage to the next (cross-stage memory).
* Loss is L1 on the final output plus 0.1 × L1 on intermediate stages. Training uses EMA weights
  and fp16 autocast in the prior only.

## Gaps in the surveyed papers that this work addresses

| Gap in the literature | What we do |
|---|---|
| Networks are a black box, and nobody checks whether the fused image reproduces its own inputs | We report **observation-consistency PSNR** `PSNR(D X̂, Y_H)` and `PSNR(R X̂, Y_M)`; the unfolding steps enforce it |
| Everyone trains and tests with one fixed blur and SRF | We test **blur σ ∈ {1.5, 2.5, 3}** and **SRF shift ±8 nm**. PUFormer can take the true operator at test time (**operator swap**) with no retraining |
| Robustness to noise is rarely measured (only HDGMamba does) | We test Gaussian input noise at 40 / 35 / 30 dB SNR |
| PSNR definitions differ, so the same method spans 24–56 dB across papers | We report three PSNR definitions plus RMSE in DN. A training-free GSA is included for calibration |

## Results (Chikusei ×4, 8 test tiles)

Kaggle run [`puformer-chikusei-x4`](https://www.kaggle.com/code/amarnathmadaka/puformer-chikusei-x4),
2026-09-24, one Tesla T4: 10.13 M parameters, 88,946 iterations in 9.5 h. Full outputs, the
complete TIP'26 Table IV comparison, per-image results and the robustness study are in
[results/puformer_chikusei_x4](../../results/puformer_chikusei_x4/).

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | CC ↑ | RMSE (DN) ↓ |
|---|---|---|---|---|---|---|
| Two-Stage Diffusion (TIP'26, reported) | 56.1871 | **0.9982** | **0.7203** | 1.5158 | 0.9948 | 25.73 |
| CLSNet (reported in TIP'26) | 55.2007 | 0.9979 | 0.7836 | 1.6318 | 0.9941 | 29.06 |
| SMGU (reported in TIP'26) | 54.5244 | 0.9977 | 0.8719 | 1.6405 | 0.9939 | 31.29 |
| SSRNet (our run) | 49.9841 | 0.9922 | 1.7606 | 2.4684 | 0.9911 | 48.83 |
| GSA (our run) | 43.1966 | 0.9778 | 2.7300 | 3.2791 | 0.9905 | 107.69 |
| Bicubic (our run) | 33.6333 | 0.7404 | 4.4061 | 9.0115 | 0.8408 | 321.81 |
| **PUFormer (ours)** | **57.9943** | 0.9969 | 0.7281 | **1.3561** | **0.9966** | **19.35** |

Against all 17 methods in TIP'26 Table IV, PUFormer is 1st on PSNR (+1.81 dB), ERGAS, CC and
RMSE, 2nd on SAM (0.008° behind), and about 11th on SSIM. The TIP'26 PSNR uses a fixed dataset
peak: `RMSE·10^(PSNR/20)` is 16.6k–17.1k DN for every one of their rows, so it matches our
headline definition. Other PSNR definitions for PUFormer: PSRT-style (per-image peak) 52.56,
band-wise with peak 1 65.29.

## Reproduce

```bash
cd methods/puformer
python train.py --mat /path/to/HyperspecVNIR_Chikusei_20140729.mat --model puformer --hours 9.5 --out out/puformer
python eval_gaps.py --mat /path/to/... --ckpt out/puformer/best_ema.pt --out out/puformer
# Kaggle: python kaggle/build_notebook.py --commit <sha> ; then push kaggle/ with the Kaggle API
```

### Training options

| Flag | Meaning |
|---|---|
| `--epochs N` | Train for N epochs. An epoch is one pass-equivalent over the training area: 800 patches of 64², i.e. 800/bs iterations. Overrides `--iters` |
| `--eval_epochs E` | Every E epochs: score validation (used for model selection) and test (logged only), and save `last.pt` |
| `--snap_epochs S` | Also keep an EMA snapshot `ema_ep#####.pt` every S epochs |
| `--hours H` | Stop at H hours of wall-clock time, save, and exit cleanly |
| `--resume path/last.pt` | Continue an earlier session (Kaggle: mount the previous kernel's output). The cosine schedule then counts iterations only |
| `--init_ckpt best_ema.pt` | Warm start (fine-tune) from earlier weights |
| `--w_sam`, `--w_ssim` | Add a SAM (radians) and a (1 − SSIM) loss term to L1 |
| `--gpus` | GPUs to use, split with DataParallel (default: all visible) |

The final test is reported twice: plain, and with a 2-fold self-ensemble (identity + transpose).
Flips are not used because they would shift the ×4 sampling grid.
