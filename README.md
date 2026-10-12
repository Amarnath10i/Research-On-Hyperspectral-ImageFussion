# Research on Hyperspectral Image Fusion

Deep learning and theory for **hyperspectral–multispectral image fusion (HSI–MSI fusion)**: from a
low-resolution hyperspectral image (LR-HSI) and a high-resolution multispectral image (HR-MSI),
reconstruct a high-resolution hyperspectral image (HR-HSI).

The repository has two tracks:

1. **Chikusei SOTA track (active).** [PUFormer](methods/puformer/), a physics-unfolded
   transformer that targets the best published Chikusei ×4 result (56.19 dB, IEEE TIP 2026).
   Current: 58.20 dB (v3), 1st on 7 of the 8 metrics in TIP'26 Table IV; on Pavia Centre ×4,
   50.69 dB (+3.53 dB), 1st on 6 of 8.
   It adds evaluations that published papers skip: consistency with the input observations,
   robustness to blur/SRF mismatch with a test-time operator swap, and robustness to noise.
2. **Theory track.** Identifiability and ambiguity of the fusion problem: admissible ambiguity
   (P1), identifiable spectral rank (P2), sensor-shift bound (P3), phase transition (P4), and the
   methods built on them (KrylovNet, NullFusion). Details in
   [docs/THEORY_AND_RESULTS.md](docs/THEORY_AND_RESULTS.md).

---

## Chikusei ×4: where things stand

| | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Q2n ↑ | CC ↑ | SCC ↑ | RMSE (DN) ↓ | Source |
|---|---|---|---|---|---|---|---|---|---|
| **PUFormer v3 (ours)** | **58.20** | **0.9989** | **0.713** | **1.330** | 0.9931 | **0.9967** | **0.9992** | **18.94** | [results](results/puformer_chikusei_x4_v3/) |
| PUFormer first run (ours) | 57.99 | 0.9989 | 0.728 | 1.356 | 0.9929 | 0.9966 | 0.9991 | 19.35 | [results](results/puformer_chikusei_x4/) |
| Two-Stage Cond. Diffusion (IEEE TIP 2026), previous published SOTA | 56.19 | 0.9982 | 0.720 | 1.516 | **0.9969** | 0.9948 | 0.9891 | 25.73 | [survey](docs/CHIKUSEI_SOTA_SURVEY.md) |
| CLSNet (2026) | 55.20 | 0.9979 | 0.784 | 1.632 | 0.9943 | 0.9941 | 0.9869 | 29.06 | reported in TIP 2026 |
| SMGU-Net (Pattern Recognition 2025) | 54.52 | 0.9977 | 0.872 | 1.641 | 0.9952 | 0.9939 | 0.9875 | 31.29 | reported in TIP 2026 |

PUFormer v3 (10.13 M parameters, fine-tuned 94,450 iterations on 2 × T4, 2-fold self-ensemble) is
1st of 18 against the full TIP'26 Table IV on PSNR, SSIM, SAM, ERGAS, CC, SCC and RMSE, and 10th
on Q2n. SSIM here is PSRT's definition (data range 1); with data range = each image's max it is
0.9970. The Q2n deficit comes from the ultraviolet bands 0–7, which the WorldView-2 MSI does not
cover; see [results/puformer_chikusei_x4_v3](results/puformer_chikusei_x4_v3/) for the analysis,
the robustness study and the caveats.

Protocol: WorldView-2 8-band MSI, Gaussian 7×7 blur with σ = 2, ×4 downsampling, 8 test tiles of
256×256. The survey covers only peer-reviewed, subscription IEEE papers:
[docs/CHIKUSEI_SOTA_SURVEY.md](docs/CHIKUSEI_SOTA_SURVEY.md).

## Pavia Centre ×4

| | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | Q2n ↑ | CC ↑ | SCC ↑ | RMSE (DN) ↓ | Source |
|---|---|---|---|---|---|---|---|---|---|
| **PUFormer (ours)** | **50.69** | **0.9974** | 1.883 | **0.617** | 0.9753 | **0.9994** | **0.9995** | **23.52** | [results](results/puformer_pavia_x4/) |
| Two-Stage Cond. Diffusion (IEEE TIP 2026) | 47.16 | 0.9972 | **1.454** | 0.826 | 0.9980 | 0.9985 | 0.9976 | 35.71 | TIP'26 Table IV |
| OTPNet (best reported Q2n) | 45.81 | 0.9967 | 1.657 | 0.947 | **0.9987** | 0.9980 | 0.9968 | 41.51 | reported in TIP 2026 |

Trained from scratch in one Kaggle session (9.99 M parameters, 93,850 iterations on 2 × T4, 2-fold
self-ensemble): 1st of 18 on PSNR (+3.53 dB), SSIM (PSRT definition), ERGAS, CC, SCC and RMSE (−34 %);
13th on SAM and 16th on Q2n. The Q2n shortfall comes from one of the four test images, which contains
the river Ticino; on the other three, Q2n is 0.9987–0.9991, at or above the best reported value, but SAM
is still 1.55–1.68°, above the best reported 1.454°. Protocol: bands 11–102
(92), IKONOS 4-band MSI, Gaussian 7×7 blur with σ = 2, ×4, rebuilt from the original scene because the
PSRT benchmark files that TIP'26 uses could not be downloaded. Details and caveats:
[results/puformer_pavia_x4](results/puformer_pavia_x4/).

## Equal-budget benchmark (×4)

Every learned method is retrained by us from its authors' code under one pipeline: the same data,
patches, batch, EMA weights and validation-only model selection, and the same 4 h on one Tesla T4.
Test PSNR (dB), single pass:

| Method | Chikusei ×4 | Pavia Centre ×4 |
|---|---|---|
| **PUFormer (ours)** | **57.95** | **50.63** |
| MIMFormer (JSTARS 2024) | 56.96 | 49.63 |
| SSR-NET (TGRS 2021) | 55.55 | 48.45 |
| DHIF-Net (unfolding, TCI 2022) | 54.39 | 47.67 |
| PSRT (TGRS 2023) | 53.99 | 47.87 |
| DCT (Information Fusion 2024) | 53.18 | 46.94 |

PUFormer also has the lowest SAM, ERGAS and RMSE on both datasets. With twice the budget (8 h) on Pavia
Centre, PUFormer reaches 50.67 dB and MIMFormer 49.97 dB: the gap narrows from 1.00 to 0.69 dB but remains.
The fixed budget favours fast networks, which is why the small SSR-NET ranks third. All metrics per job are in
[results/q1/](results/q1/); the jobs are defined in [methods/puformer/kaggle/jobs.py](methods/puformer/kaggle/jobs.py).

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
│   ├── puformer/       ★ Physics-Unfolded Transformer, Chikusei x4 SOTA attempt (active)
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

The PUFormer manuscript (IEEE journal format), with the Chikusei and Pavia Centre results, is in
[paper/puformer/main.pdf](paper/puformer/main.pdf); its sources and build notes are in
[paper/puformer/](paper/puformer/).

## License

Research use. The datasets belong to their respective providers.
