# PUFormer: unfolded HSI–MSI fusion with exact observation operators

PUFormer fuses a low-resolution hyperspectral image (LR-HSI) and a high-resolution multispectral image (HR-MSI)
into a high-resolution hyperspectral image. It alternates data steps with learned priors, and its data steps use
the **exact** observation operators: the blur, decimation and spectral response that produced the data, with the
border rule of the simulation. Results and the comparison with ten retrained networks are in the
[repository README](../../README.md); the paper is in [paper/puformer](../../paper/puformer/).

## Method

```
X0 = upsample(Y_H)
for k = 1..K:                                                  (K = 3)
    Z_k = data step on X_{k-1} with the exact D (blur + decimation) and R (spectral response)
    X_k = Z_k + Prior_k([Z_k, Y_M, s² Dᵀ(D Z_k − Y_H)], memory_{k−1})
```

* **Exact operators.** `D` uses the simulation's reflect border and `Dᵀ`, `Rᵀ` are its exact adjoints (inner-product
  test error < 1e-14), so `D X = Y_H` and `R X = Y_M` hold for the true image.
* **Closed-form data step.** With a separable blur, `D` and `R` act on the three axes of the cube separately. In
  the product eigenbasis of the three factors the exact proximal data step is a per-component division
  (`--dstep kron`), and the fusion null space, the part of the image neither observation measures, is explicit:
  a fraction (1 − M/B)(1 − 1/s²) of the unknowns. `--dstep kronlearn` learns per-component gains in (1/2, 2) times
  the exact ones, which keeps the ground truth a fixed point and the step non-expansive. `grad` (one gradient step)
  and `cg` (conjugate gradient) are the alternatives for non-separable operators.
* **Prior.** A 3-level U-Net (widths 48, 96, 192) of Restormer blocks with transposed attention (`--prior unet`),
  or the eigen-operator network (`--prior eon`), which mixes globally in the operator's eigenbasis with
  eigenvalue-conditioned filters and has no down-sampling. The output layers start at zero, so training begins
  from the physics solution. Decoder features pass from stage to stage (cross-stage memory).
* **Operator family.** Training draws a new operator for every batch (`--rand_*` flags below). With `--omc 1` the
  step sizes and the prior (FiLM) are conditioned on the blur width, the spectral shift and the noise level
  estimated from each observation. At test time the operators are given or estimated from the pair by
  minimising ‖D(Y_M) − R(Y_H)‖² (`cond_eval.py`).
* **Loss.** L1 on the output + 0.1 × L1 on the intermediate stages + 0.05 × SAM + 0.1 × (1 − SSIM); AdamW, EMA weights.

### Optional modules

Each is zero-initialized, so a model with the module starts as the model without it. A module joins the
default configuration only if it does not lower any benchmark metric.

| Flag | Module |
|---|---|
| `--nsp 1` | Null-space-gated prior: the prior's correction acts fully in the null space and with a learned weight in the observed subspace |
| `--ntp 1` | Null-space transfer: the LR-HSI enters the U-Net bottleneck and a per-pixel spectral branch carries the MSI's fine detail to the other bands |
| `--cwd 1` | Confidence-weighted data step: small networks down-weight measurements that disagree locally with the estimate (changes, clouds, stripes); gives change maps |
| `--unc 1` | Uncertainty head: predicts the size of the error left in the null space |
| `--sa 1` | Sensor-independent input: the HR-MSI enters as R⁺Y_M in the hyperspectral bands, so one network serves any MSI sensor |
| `--reg 1` | Wider operator family: anisotropic blur and sub-pixel offsets of the LR-HSI grid in the observation code |
| `--mres 1` | HR-MSI residual (B/M) Rᵀ(R Z − Y_M) as an extra prior input |
| `--nomsi 1`, `--nohsi 1` | Input ablations: LR-HSI only (super-resolution) or HR-MSI only (spectral super-resolution), same network and data step |

Ablation switches of the base model: `--physics 0`, `--res_input 0`, `--memory 0`, `--mixer window|conv`,
`--op learned`, `--pad zeros`, `--stages K`.

## Protocol (Wald)

| Dataset | Bands | HR-MSI | Test images |
|---|---|---|---|
| Chikusei | 128 (363–1018 nm), divided by its max | WorldView-2, 8 bands | 8 × 256² |
| Pavia Centre | 92 (bands 11–102) | IKONOS, 4 bands | 4 × 256² |
| CAVE / Harvard | 31 | Nikon RGB | 12 / 20 scenes |
| NEON (real) | spectrometer + co-registered camera RGB | fitted response | reduced and full resolution |

The LR-HSI is a Gaussian blur, (2s − 1) × (2s − 1) with σ = s/2 (7 × 7, σ = 2 at ×4), followed by ×s decimation, each
crop simulated on its own with a reflect border. Model selection uses validation PSNR only; the test set is scored
once. PSNR uses the dataset's peak; SSIM, SAM, ERGAS, Q2n, CC, SCC and RMSE in DN are reported as well.

## Scripts

| Script | Purpose |
|---|---|
| `train.py` | Train and test one model (`--model puformer` or a retrained baseline, see `hsifuse/external.py`) |
| `test.py`, `eval_gaps.py` | Full test report; observation consistency and test-time operator swap |
| `cond_eval.py` | Changed blur, SRF, noise, anisotropic/defocus blur, grid offsets, misregistration, corrupted inputs; operators estimated from the pair; change-map AUROC |
| `family_eval.py` | One checkpoint on other MSI sensors and scale factors |
| `unc_eval.py` | Predicted against real null-space error |
| `null_analysis.py` | Splits each method's error into the observed subspace and the null space |
| `msi_value.py` | What the HR-MSI adds: energy shares of the observed and null parts and the PSNR fixed by the measurements alone |
| `blind_eval.py`, `neon_fullres.py`, `complexity.py` | Estimated operators, real full-resolution data, parameters and run time |
| `selfcheck.py`, `kron_check.py`, `reg_check.py`, `sa_check.py`, `arch_check.py`, `arch2_check.py`, `arch3_check.py`, `ext_smoke.py` | Numerical checks of the operators, the closed form, the modules and the baseline adapters |

## Running

All training, tests and checks run as Kaggle kernels; the local machine only edits code and pushes kernels.

```bash
cd methods/puformer
# one job (the jobs of the paper are defined in kaggle/jobs.py)
python train.py --dataset chikusei --mat /kaggle/input/chikusei --scale 4 --model puformer --hours 4 --out out/puformer
# family training with the estimated operator and conditioning
python train.py ... --rand_sigma 0.5,2.0 --rand_shift 12 --rand_snr 30,50 --noise_prob 0.5 --omc 1
python cond_eval.py --dataset chikusei --mat ... --ckpt out/puformer/best_ema.pt --model puformer --omc 1
# a 2 x T4 session from kaggle/jobs.py, and its results into results/q1/<job>/
cd kaggle && python build_jobs.py --session puf-a10 --user <user> && kaggle kernels push -p sessions/puf-a10
python fetch_results.py --kernel <user>/puf-a10 --small
```

`fetch_results.py` merges a new `cond_eval.json` into an existing one by condition, so a cancelled or partial
evaluation never removes results.

### Training options

| Flag | Meaning |
|---|---|
| `--dataset`, `--scale` | Dataset and resolution ratio (the blur follows the scale) |
| `--hours H` / `--deadline T` | Stop cleanly after H hours / at unix time T; the cosine schedule spans the iterations that fit |
| `--rand_sigma lo,hi` | Blur width per batch, uniform in [lo, hi] × the nominal width |
| `--rand_shift x` | SRF shift per batch, uniform in [−x, x] nm |
| `--rand_snr lo,hi` | Gaussian noise on both inputs, SNR uniform in [lo, hi] dB |
| `--rand_aniso 1`, `--rand_offset x` | Independent column blur width; sub-pixel offsets of the LR-HSI grid |
| `--rand_srf 1` | A random multispectral sensor per batch (3–10 bands) |
| `--rand_scale 4,8` | Resolution ratio per batch |
| `--rand_corrupt p` | Corrupted observations with probability p (local changes, stripes, local misregistration, band noise) |
| `--w_sam`, `--w_ssim`, `--w_deep`, `--w_unc` | Loss weights |
| `--init_ckpt`, `--resume` | Warm start / continue an earlier session |
| multi-GPU | `torchrun --nproc_per_node G train.py ...` (`--bs` is per GPU) |

The final test is reported twice: plain, and with a 2-fold self-ensemble (identity + transpose); flips would shift
the decimation grid.
