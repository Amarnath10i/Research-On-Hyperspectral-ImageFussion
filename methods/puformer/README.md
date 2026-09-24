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
| Headline PSNR | 10·log10(1/MSE), i.e. peak = the dataset's global max. TIP'26's RMSE column is in DN, and 20·log10(max/RMSE) reproduces their PSNR column. PSRT-style and band-wise PSNR are reported as well. |

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

*Pending: Kaggle run
[`puformer-chikusei-x4`](https://www.kaggle.com/code/amarnathmadaka/puformer-chikusei-x4).*

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | RMSE (DN) ↓ |
|---|---|---|---|---|---|
| Two-Stage Diffusion (TIP'26, reported) | 56.1871 | 0.9982 | 0.7203 | 1.5158 | 25.73 |
| CLSNet (reported in TIP'26) | 55.2007 | 0.9979 | 0.7836 | 1.6318 | 29.06 |
| GSA (our run) | 44.8* | 0.978 | 2.73 | 3.28 | ~106 |
| Bicubic (our run) | – | 0.740 | 4.41 | 9.01 | – |
| **PUFormer (ours)** | *pending* | | | | |

\* GSA and bicubic figures come from an earlier local metric check and will be refreshed from the
Kaggle run.

## Reproduce

```bash
cd methods/puformer
python train.py --mat /path/to/HyperspecVNIR_Chikusei_20140729.mat --model puformer --hours 9.5 --out out/puformer
python eval_gaps.py --mat /path/to/... --ckpt out/puformer/best_ema.pt --out out/puformer
# Kaggle: python kaggle/build_notebook.py --commit <sha> ; then push kaggle/ with the Kaggle API
```
