# Research on Hyperspectral Image Fusion

Deep learning and theory for **hyperspectral–multispectral image fusion (HSI–MSI fusion)**: from a
low-resolution hyperspectral image (LR-HSI) and a high-resolution multispectral image (HR-MSI),
reconstruct a high-resolution hyperspectral image (HR-HSI).

The repository has two tracks:

1. **ExactFuse (active).** [ExactFuse](methods/puformer/) (code name PUFormer, run with
   `--dstep kronlearn --mres 1 --sa 1`) is an unfolded fusion network whose data steps use the exact observation
   operators: the true blur, decimation and spectral response, with the border rule of the simulation. With a
   separable blur these operators give a closed-form data step and an explicit fusion null space (the part of the
   image that neither observation measures, 88% of the unknowns on Chikusei ×4). ExactFuse takes the closed-form
   step with learned spectral gains, gives its prior inputs in the hyperspectral bands only (so one network can
   serve any multispectral sensor), is trained over a family of blurs, spectral responses and noise levels and can
   estimate the operators from the two observations at test time. Against eleven published networks retrained
   under one protocol it is the most accurate on every dataset and scale factor tested. The earlier version with a
   gradient data step and the HR-MSI as prior input is called the *base model* below.
2. **Theory track.** Identifiability and ambiguity of the fusion problem: admissible ambiguity
   (P1), identifiable spectral rank (P2), sensor-shift bound (P3), phase transition (P4), and the
   methods built on them (KrylovNet, NullFusion). Details in
   [docs/THEORY_AND_RESULTS.md](docs/THEORY_AND_RESULTS.md).

---

## Equal-budget benchmark (×4)

Every learned method is retrained by us from its authors' code under one pipeline: the same data, patches,
batch, EMA weights and validation-only model selection, and the same 4 h on one Tesla T4 (SSR-NET 1.5 h).
Test PSNR (dB) and SAM (degrees), single pass:

| Method | Chikusei PSNR ↑ | SAM ↓ | Pavia Centre PSNR ↑ | SAM ↓ |
|---|---|---|---|---|
| **ExactFuse (ours)** | **58.05** | **0.724** | **50.63** | **1.892** |
| MIMO-SST (TGRS 2024) | 57.27 | 0.780 | 49.69 | 1.961 |
| OTPNet (AAAI 2025) | 57.06 | 0.786 | 50.08 | 1.951 |
| MIMFormer (JSTARS 2024) | 56.96 | 0.801 | 49.63 | 1.982 |
| CLSNet (2026) | 56.65 | 0.811 | 49.50 | 1.993 |
| SSR-NET (TGRS 2021) | 55.55 | 0.954 | 48.45 | 2.443 |
| SMGU-Net (PR 2025) | 55.14 | 0.949 | 48.03 | 2.331 |
| DHIF-Net (TCI 2022) | 54.39 | 1.020 | 47.67 | 2.295 |
| PSRT (TGRS 2023) | 53.99 | 1.057 | 47.87 | 2.212 |
| CaFormer (2026) | 53.56 | 1.054 | 48.43 | 2.145 |
| U2Net (ACM MM 2023) | 53.35 | 1.076 | 47.29 | 2.327 |
| DCT (Information Fusion 2024) | 53.18 | 1.133 | 46.94 | 2.442 |

ExactFuse is best on PSNR, SSIM, SAM and ERGAS on both datasets, 0.79 / 0.55 dB above the best retrained network.
Over three seeds its PSNR varies by at most 0.03 dB. Pooled over the 76 test images of the eight settings it is
more accurate than the strongest network of each setting on 75 of them (one-sided Wilcoxon signed-rank test,
p = 1.9 × 10⁻¹⁴). Its error in the observed subspace, the part that contradicts the inputs, is 27× (Chikusei) and
9× (Pavia Centre) smaller than that of the best retrained network on this measure (MIMO-SST), and it reproduces the
LR-HSI 17 dB more accurately than the strongest baseline. U2Net, whose attention spans all pixels, is tested in
overlapping 128 × 128 tiles.

Other datasets and scale factors (same protocol and budget; –: not run; Harvard ×8 and Pavia ×16 are queued):

| Method | CAVE ×4 | Harvard ×4 | CAVE ×8 | Chikusei ×8 | Pavia ×8 | Chikusei ×16 |
|---|---|---|---|---|---|---|
| **ExactFuse** | **52.46** | **50.03** | **50.39** | **56.95** | **50.05** | **56.11** |
| MIMFormer | 50.93 | 49.72 | 48.10 | 55.57 | 49.09 | 54.21 |
| MIMO-SST | 50.39 | 49.82 | – | 55.88 | 48.90 | – |
| OTPNet | – | – | – | 56.11 | 49.51 | – |
| DHIF-Net | 49.82 | 49.57 | – | – | – | – |
| SMGU-Net | 48.54 | 49.34 | – | – | – | – |
| PSRT | 48.51 | 49.29 | 45.21 | 53.05 | 47.13 | – |
| U2Net | 48.04 | 48.70 | – | – | – | – |
| CaFormer | 47.91 | 49.09 | – | – | – | – |
| SSR-NET | 46.51 | 49.18 | 44.25 | 54.57 | 47.71 | 53.98 |
| DCT | 46.36 | 48.92 | 43.17 | 51.77 | 45.51 | – |

Where the margin comes from: a single U-Net prior without data steps already reaches 57.67 / 50.39 dB, above
every retrained network, and the exact data steps add 0.38 / 0.23 dB. The recipe is not the whole story: with the
same training recipe for both, ExactFuse leads MIMFormer by 0.53 / 0.40 dB (our recipe) and 0.72 / 0.44 dB (the
baselines'). With twice the budget (8 h) on Chikusei, ExactFuse reaches 58.18 dB and MIMFormer 57.37 dB:
the gap narrows from 1.10 to 0.81 dB but remains, although ExactFuse completes 63,850 iterations against 186,800
(the base model on Pavia Centre: 1.00 → 0.69 dB). ExactFuse has 10.45 M
parameters. The whole pipeline takes 0.39 s for a 256 × 256 Chikusei pair on a T4 (operator estimate 0.08 s,
eigendecompositions 0.07 s, cached per image size, network 0.24 s, 0.54 GB); the closed-form data step is
cheaper than a conjugate-gradient step (0.24 vs 0.33 s with 5 iterations), and ExactFuse is about 2.5× slower than
MIMFormer at every image size (5.3 vs 1.9 s at 1024 × 1024). All metrics per job are in
[results/q1/](results/q1/); the jobs are defined in [methods/puformer/kaggle/jobs.py](methods/puformer/kaggle/jobs.py).

## What the HR-MSI adds

Why fuse instead of super-resolving the LR-HSI alone? In the eigenbasis of the exact operators the LR-HSI sees only
the coarse part of every band, so single-image super-resolution must invent all fine detail. The HR-MSI measures
that detail in only 3–8 spectral directions, but detail is shared across bands, so these directions carry most of
its energy ([msi_value.py](methods/puformer/msi_value.py), no network involved):

| ×4 | MSI's share of the fine-detail energy | Minimum-norm image from the LR-HSI alone | From both observations |
|---|---|---|---|
| Chikusei | 84% | 33.5 dB | 41.3 dB |
| Pavia Centre | 77% | 27.2 dB | 33.6 dB |
| CAVE | 95% | 31.1 dB | 44.1 dB |
| Harvard | 95% | 34.1 dB | 47.7 dB |

The minimum-norm image keeps exactly what the observations measure and is zero elsewhere; the gain from the HR-MSI
is the same at ×8 and ×16. Results: [results/q1/msi-value-*](results/q1/).

## One network for a family of observation models

Trained over random blur widths (0.5–2 σ₀), spectral-response shifts (±12 nm) and noise (30–50 dB SNR), with the
operators given to the data steps or estimated from the two observations, the network keeps its accuracy when the
observation model changes (mean PSNR in dB):

| Model | Chikusei: training operator | Noise 40–30 dB | Blur + SRF + noise | Pavia: training operator | Noise | Blur + SRF + noise |
|---|---|---|---|---|---|---|
| MIMFormer, nominal training | 56.96 | 50.33 | 48.98 | 49.63 | 45.24 | 44.15 |
| MIMFormer, randomized training | 56.50 | 51.67 | 51.30 | 49.37 | 45.87 | 45.22 |
| ExactFuse, nominal training, estimated operator | 58.05 | 47.60 | 46.71 | 50.63 | 43.15 | 42.32 |
| **ExactFuse, randomized + estimated operator** | **57.77** | **52.64** | **52.59** | **50.43** | **46.55** | **46.53** |

Within the family the blur width is recovered to within 0.05 px and the spectral shift exactly on the search grid.
Outside the family the method fails like the others: a half-pixel misregistration of the HR-MSI costs every
method more than 15 dB. Registering the HR-MSI with the offset estimated from the pair before fusion recovers part
of the loss (Pavia Centre, 1 px: 27.0 → 43.9 dB; GSA with the same registration 34.3 dB). With an anisotropic blur and 35 dB noise added, ExactFuse trained over the wider family (noise, anisotropic blur, grid offsets) reaches 45.72 dB on Chikusei against 45.49 dB for MIMFormer.

**One network for any sensor.** ExactFuse trained only on random multispectral responses (3–10 bands of random
position and width) and tested on seven real sensors it never saw is 9.9–13.5 dB (Chikusei) and 10.5–15.6 dB
(Pavia Centre) above GSA and 3.2–11.3 dB (Chikusei) and 3.6–7.2 dB (Pavia Centre) above MIMFormer trained on the
same random responses with the same input; on WorldView-2 it is 1.79 dB below ExactFuse trained for that sensor
and 0.7 dB below MIMFormer trained for it. Trained for a three-band RGB camera, ExactFuse reaches 47.15 dB on Chikusei against
46.03 dB for MIMFormer.

**One network for several scale factors.** ExactFuse trained at ×4 and ×8 together is at most 0.15 dB below the
networks trained for each scale and above the best retrained network of every scale; at ×16, which it never saw,
it leads that network by 1.76 dB (Chikusei, 55.97 dB) and 1.99 dB (Pavia Centre, 49.38 dB).

## Comparison with published numbers (Chikusei ×4)

| | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Q2n ↑ | CC ↑ | SCC ↑ | RMSE (DN) ↓ | Source |
|---|---|---|---|---|---|---|---|---|---|
| **ExactFuse, 8 h (ours)** | **58.21** | **0.9989** | **0.713** | **1.326** | 0.9931 | **0.9967** | **0.9992** | **18.91** | [results](results/q1/long-chikusei-x4-ng/) |
| Two-Stage Cond. Diffusion (IEEE TIP 2026) | 56.19 | 0.9982 | 0.720 | 1.516 | **0.9969** | 0.9948 | 0.9891 | 25.73 | [survey](docs/CHIKUSEI_SOTA_SURVEY.md) |
| CLSNet (2026) | 55.20 | 0.9979 | 0.784 | 1.632 | 0.9943 | 0.9941 | 0.9869 | 29.06 | reported in TIP 2026 |
| SMGU-Net (Pattern Recognition 2025) | 54.52 | 0.9977 | 0.872 | 1.641 | 0.9952 | 0.9939 | 0.9875 | 31.29 | reported in TIP 2026 |

ExactFuse trained for 8 h (one run, 2-fold self-ensemble) is 1st of 18 against the full TIP'26 Table IV on seven of
the eight metrics and 10th on Q2n; in a single pass it reaches 58.18 dB, above the base model trained for 20.4 h
(58.15 dB). On Pavia Centre ×4 the long-trained base model reaches 50.69 dB against 47.16 dB reported
([results](results/puformer_pavia_x4/)). These tables give context, not a ranking: the reported numbers come
from other implementations of the protocol, and our retrained baselines exceed their reported values by up to
3.6 dB. SSIM here is PSRT's definition (data range 1); the Q2n deficit comes from the ultraviolet bands 0–7, which
the WorldView-2 MSI does not cover.

## Theory

With a separable blur and the true border rule, the data step has a closed form and the fusion null space is
explicit. The operators are identifiable from the observation pair over continuous parameters: two sensor
configurations are indistinguishable exactly when the scene lies in the null space of
D_θ R_φ* − R_φ D_θ*, which happens only for degenerate scenes (spatially constant ones for the blur, ones with
equal bands for the response shift), and the estimate moves by at most ‖n‖ / σ_min(G) under observation noise
n, where G is the Jacobian of the identifiability residual. An operator error moves the reconstruction by an
amount bounded independently of the learned weights, so the noise is carried through the operator estimate to
the reconstruction. On the test images G has full rank for two blur widths, two grid offsets and the response
shift (condition number 1.28 on Chikusei after column scaling), and the predicted error of the offsets matches
the measured one (0.0013 px at 30 dB SNR); the blur estimate has a small second-order noise bias.
Analysis: [methods/puformer/operator_analysis.py](methods/puformer/operator_analysis.py).

**Real data.** On NEON (airborne spectrometer and RGB camera) ExactFuse is the most accurate at reduced
resolution (38.70 vs 38.45 dB for MIMFormer) and has the best no-reference score at full resolution (HQNR 0.920
vs 0.892; spectral distortion 0.021 vs 0.064), with a slightly higher spatial distortion than MIMFormer and
DHIF-Net (0.060 vs 0.047 and 0.039).

## Novelty check

[paper/puformer/novelty_check.py](paper/puformer/novelty_check.py) compares the architecture with 19 prior
methods on 16 architectural facets. For ExactFuse as adopted, 1 facet appears in no prior method (the border rule
and adjoint of the simulation in the closed-form step) and 4 in no prior fusion method (true border rule, learned
filters in the operator's eigen domain, training over an operator family, observation-model conditioning); the
nearest method overlaps 42% (USRNet, single-image super-resolution) and the nearest fusion method 23% (IR&ArF).
The novelty is the analysis and its use in one network, not a new building block.
SPECTER-2 similarity of the abstracts places the paper among typical papers of the field, with no near-duplicate.
The facets are scored by us from the papers; results are in [results/q1/novelty/](results/q1/novelty/).

## Ablation study (×4)

Each row changes one module of the full model and is retrained from scratch with the same 4 h budget.
Δ is the PSNR change (dB) against the full model; rows (a) and (d) are the mean over three seeds
(± standard deviation) on both datasets.

The full model's loss is L1 on the final output + 0.1 × L1 on every intermediate stage (deep supervision)
+ 0.05 × SAM + 0.1 × (1 − SSIM); rows (l) and (m) change it.

| Row | Change | Pavia Centre ×4 | Chikusei ×4 |
|---|---|---|---|
| (a) | Full model | 50.62 ± 0.02 dB | 57.95 ± 0.004 dB |
| (b) | No data-consistency step, no residual input | −0.17 | −0.30 |
| (c) | No residual input | −0.01 | +0.06 |
| (d) | Zero border instead of the exact operator | −0.08 (± 0.03) | −0.22 (± 0.05) |
| (e) | Blur kernel and spectral response learned | −0.06 | −0.22 |
| (f) | No cross-stage memory | +0.05 | +0.01 |
| (g) | One stage (K = 1, 3.3 M parameters) | −0.02 | −0.20 |
| (h) | Two stages | +0.02 | −0.02 |
| (i) | Four stages | +0.06 | +0.01 |
| (j) | Convolution instead of transposed attention | −0.02 | +0.07 |
| (k) | 8×8 window attention instead | +0.01 | +0.03 |
| (l) | Loss: L1 + deep supervision only (no SAM / SSIM) | +0.03 | +0.02 |
| (m) | Loss: no deep supervision | −0.12 | +0.04 |

Blur changed at test time (σ = 1.5, 2.5, 3.0 instead of 2), worst test PSNR when the true blur is given
to the model:

| Operator | Pavia Centre ×4 | Chikusei ×4 |
|---|---|---|
| Exact (full model) | 50.50 | 57.70 |
| Zero border | 49.64 | 54.38 |
| Learned (cannot be replaced) | 49.33 | 54.28 |

What it shows: the accuracy comes from the exact data-consistency step. Removing the data step costs the
most; the zero-border and learned operators lose 0.06–0.22 dB at the training blur and 0.9–3.4 dB when the
blur changes, while the exact model loses at most 0.25 dB. Every zero-border seed is below every full-model
seed on both datasets. Cross-stage memory, the type of attention and the SAM / SSIM loss terms make no
measurable difference; deep supervision helps on Pavia Centre (−0.12 dB without it) but not on Chikusei.
A single stage is free on Pavia Centre but costs 0.2 dB on Chikusei. On Chikusei several simplifications
are slightly better than the full model, so its prior is not optimal there.

---

## Repository layout

```
.
├── methods/            Our methods, one self-contained package each
│   ├── puformer/       ★ PUFormer: unfolded fusion with exact observation operators (active)
│   ├── nullfusion/     Null-space conditional fusion (exact data consistency)
│   ├── krylovnet/      Unrolled preconditioned GMRES + identifiable-rank theory (P2)
│   ├── daetf/          Admissible-ambiguity manifold (P1)
│   ├── continuumfusion/ Sensor-independent continuous scene field (P3)
│   ├── zerofusion/     Identifiability phase diagram (P4)
│   ├── spectralflow/   Null-space generative models
│   ├── dacf/           Degradation-adaptive conditional flow
│   ├── ason/           Adaptive spectral operator network
│   └── consistentflow/ (superseded)
├── common/hsifusion/   Shared library: datasets, degradation, SRF, metrics, losses, engine
├── experiments/
│   ├── scripts/        Training scripts (train_*.py)
│   ├── notebooks/      Kaggle / Colab notebooks
│   └── kaggle/         Kaggle kernel folders
├── results/            Result JSONs and comparison tables (weights are not tracked)
├── docs/               Survey, protocol audit, datasets, architecture notes, theory write-up
├── paper/              Manuscript draft, bibliography, table and theorem scripts
├── tools/              Notebook builders and Kaggle automation
└── tests/              Quick model and forward-pass checks
```

Every folder under `methods/` has its own README describing the method, its status and how to run it.

---

## Quick start

```bash
pip install -r requirements.txt            # torch, numpy, scipy, h5py, ...

# PUFormer on Chikusei (needs HyperspecVNIR_Chikusei_20140729.mat)
cd methods/puformer
python train.py --mat /path/to/HyperspecVNIR_Chikusei_20140729.mat --model puformer --hours 9.5 --out out/puformer
python eval_gaps.py --mat /path/to/HyperspecVNIR_Chikusei_20140729.mat --ckpt out/puformer/best_ema.pt --out out/puformer

# CPU smoke test (synthetic data, about 1 minute)
python train.py --mat x --smoke --iters 4 --eval_every 2 --width 16 --stages 2 --bs 2 --amp 0 --out /tmp/smoke
```

Theory-track self-checks (run from the repo root):

```bash
export PYTHONPATH=common:methods        # PowerShell: $env:PYTHONPATH="common;methods"
python -c "import krylovnet.krylovnet as k; k.selfcheck.run_all()"
python -c "import nullfusion.nullfusion as n; n.selfcheck.run_all()"
```

GPU training runs on Kaggle (T4). Each method's README gives its notebook, and
`methods/puformer/kaggle/build_notebook.py` builds a notebook pinned to a git commit.

## Data

| Dataset | Bands | Source |
|---|---|---|
| Chikusei | 128 (363–1018 nm), 2517×2335 | Yokoya & Iwasaki, 2016 (Kaggle mirror `mingliu123/chikusei`) |
| Houston 2018 | 48 | IEEE GRSS Data Fusion Contest |
| CAVE / Harvard / Pavia | 31 / 31 / 102 | public benchmarks |

See [docs/DATASETS.md](docs/DATASETS.md). Datasets, model weights and paper PDFs are not stored in
the repository (see `.gitignore`).

## Documentation

| Document | Contents |
|---|---|
| [docs/KEY_PAPERS.md](docs/KEY_PAPERS.md) | The 17 papers most relevant to PUFormer: protocol, retrained baselines, closest methods and theory |
| [docs/CHIKUSEI_SOTA_SURVEY.md](docs/CHIKUSEI_SOTA_SURVEY.md) | Chikusei SOTA from peer-reviewed IEEE papers, protocols, metric definitions |
| [docs/THEORY_AND_RESULTS.md](docs/THEORY_AND_RESULTS.md) | Theorems P1–P4, KrylovNet / NullFusion results, protocol and statistics |
| [docs/SOTA_COMPARISON.md](docs/SOTA_COMPARISON.md) | CAVE / Houston SOTA tables |
| [docs/PROTOCOL_AUDIT.md](docs/PROTOCOL_AUDIT.md) | Audit of evaluation protocols |
| [docs/PROPOSAL_POSITIONING.md](docs/PROPOSAL_POSITIONING.md) | How the proposals relate to prior work |
| [paper/](paper/) | Manuscript draft and bibliography |

## Citation

```bibtex
@misc{amarnath2026hsifusion,
  title  = {Research on Hyperspectral Image Fusion: Physics-Unfolded Fusion and Identifiability Theory},
  author = {Amarnath M},
  year   = {2026},
  url    = {https://github.com/Amarnath10i/Research-On-Hyperspectral-ImageFussion}
}
```

## Paper

The PUFormer manuscript is in [paper/puformer/](paper/puformer/): the journal version
([main.pdf](paper/puformer/main.pdf), [supplementary.pdf](paper/puformer/supplementary.pdf)) and an 8-page
conference version ([cvpr_main.pdf](paper/puformer/cvpr_main.pdf),
[cvpr_supp.pdf](paper/puformer/cvpr_supp.pdf)). Both are built on Kaggle from the committed result files; the
sources and build notes are in [paper/puformer/README.md](paper/puformer/README.md).

## License

Research use. The datasets belong to their respective providers.
