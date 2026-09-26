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
  (inner-product test error < 1e-14). Since v3, `D` uses reflect padding, so it is exactly the
  operator that simulates the LR-HSI (`D X = Y_H` for the true image), and `Dᵀ` folds the reflected
  border back (`--pad zeros` reproduces the first run).
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

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Q2n ↑ | CC ↑ | SCC ↑ | RMSE (DN) ↓ |
|---|---|---|---|---|---|---|---|---|
| Two-Stage Diffusion (TIP'26, reported) | 56.1871 | 0.9982 | 0.7203 | 1.5158 | **0.9969** | 0.9948 | 0.9891 | 25.73 |
| CLSNet (reported in TIP'26) | 55.2007 | 0.9979 | 0.7836 | 1.6318 | 0.9943 | 0.9941 | 0.9869 | 29.06 |
| SSRNet (our run) | 49.9841 | – | 1.7606 | 2.4684 | – | 0.9911 | – | 48.83 |
| Bicubic (our run) | 33.6333 | 0.8500 | 4.4061 | 9.0115 | – | 0.8408 | 0.7648 | 321.81 |
| PUFormer first run | 57.9944 | 0.9989 | 0.7281 | 1.3560 | 0.9929 | 0.9966 | 0.9991 | 19.35 |
| **PUFormer v3** (self-ensemble) | **58.2016** | **0.9989** | **0.7126** | **1.3304** | 0.9931 | **0.9967** | **0.9992** | **18.94** |

SSIM is PSRT's `cal_ssim.py` (data range 1), consistent with the fixed-peak PSNR of TIP'26. With data
range = each image's own max, v3 scores 0.9970. Q2n and SCC follow Vivone's toolbox
(`q2n.m`, `SCC.m`). Against all 17 methods of TIP'26 Table IV, v3 is 1st on PSNR, SSIM, SAM, ERGAS,
CC, SCC and RMSE and 10th on Q2n. The Q2n deficit sits in bands 0–7 (363–404 nm), which the
WorldView-2 MSI does not cover and whose fine detail is noise-like. Single pass without
self-ensemble: 58.1467 dB, SAM 0.7171, also 1st on the same seven metrics.

* First run: [results/puformer_chikusei_x4](../../results/puformer_chikusei_x4/). From scratch,
  1 × T4, 88,946 iterations, zero-padded physics.
* v3: [results/puformer_chikusei_x4_v3](../../results/puformer_chikusei_x4_v3/). Kaggle
  [`puformer-chikusei-x4-v3`](https://www.kaggle.com/code/amarnath10chinu/puformer-chikusei-x4-v3):
  fine-tuned from the first run with reflect (exact) physics, + SAM and SSIM loss, 2 × T4 DDP,
  94,450 iterations. It also contains the Q2n analysis, the robustness study with operator swap
  (blur mismatch now recovers to within 0.26 dB of nominal), and why the v2 run was killed.

## Reproduce

```bash
cd methods/puformer
python selfcheck.py                                   # adjoints, Q2n / SCC ports, forward pass (CPU, ~10 s)
python train.py --mat /path/to/HyperspecVNIR_Chikusei_20140729.mat --model puformer --hours 9.5 --out out/puformer
# v3 recipe: fine-tune on 2 GPUs (--bs is per GPU)
torchrun --standalone --nproc_per_node 2 train.py --mat ... --init_ckpt first_run/best_ema.pt --pad reflect     --bs 8 --lr 1.5e-4 --w_sam 0.05 --w_ssim 0.1 --hours 10.8 --out out/v3
python test.py --mat ... --ckpt out/v3/best_ema.pt --pad reflect --out out/v3_test     # full test report
python eval_gaps.py --mat ... --ckpt out/v3/best_ema.pt --pad reflect --out out/v3
python compare_tip26.py out/v3/results.json                                            # rank vs TIP'26
# Kaggle: python kaggle/build_notebook.py --slug <slug> --hours 11 ; kaggle kernels push -p kaggle
```

The Kaggle notebook embeds this folder's code (with its sha256), builds the data cache, re-scores
the warm-start checkpoint, probes DDP against one GPU, trains until a deadline inside the 12 h limit
(resuming on one GPU if training dies), then runs the final test and the robustness study.

### Training options

| Flag | Meaning |
|---|---|
| `--pad reflect/zeros` | Border mode of `D`/`Dᵀ` inside the network (reflect = the simulation operator; default) |
| `--epochs N` | Train for N epochs. An epoch is one pass-equivalent over the training area: 800 patches of 64², i.e. 800/(bs · GPUs) iterations. Overrides `--iters` |
| `--eval_every I` / `--eval_epochs E` | Score validation (used for model selection) and test (logged only), and save `last.pt` |
| `--snap_epochs S` | Also keep an EMA snapshot `ema_ep#####.pt` every S epochs |
| `--hours H` / `--deadline T` | Stop cleanly after H hours / at unix time T. The cosine then spans the iterations that fit, re-estimated from the measured speed |
| `--resume path/last.pt` | Continue an earlier session (Kaggle: mount the previous kernel's output). The cosine then counts iterations only (`--sched time` to keep it time-based) |
| `--init_ckpt best_ema.pt` | Warm start (fine-tune) from earlier weights; scored before training (`--eval_init`) |
| `--w_sam`, `--w_ssim` | Add a SAM (radians) and a (1 − SSIM) loss term to L1 |
| `--probe N` | Speed test: time N iterations, print samples/s, exit |
| multi-GPU | launch with `torchrun --nproc_per_node G` (DistributedDataParallel; `--bs` is per GPU) |

The final test is reported twice: plain, and with a 2-fold self-ensemble (identity + transpose).
Flips are not used because they would shift the ×4 sampling grid.
