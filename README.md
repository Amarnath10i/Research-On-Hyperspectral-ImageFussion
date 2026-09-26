# Research on Hyperspectral Image Fusion

Deep learning and theory for **hyperspectral–multispectral image fusion (HSI–MSI fusion)**: from a
low-resolution hyperspectral image (LR-HSI) and a high-resolution multispectral image (HR-MSI),
reconstruct a high-resolution hyperspectral image (HR-HSI).

The repository has two tracks:

1. **Chikusei SOTA track (active).** [PUFormer](methods/puformer/), a physics-unfolded
   transformer that targets the best published Chikusei ×4 result (56.19 dB, IEEE TIP 2026).
   Current: 58.20 dB (v3), 1st on 7 of the 8 metrics in TIP'26 Table IV.
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
├── baselines/          Third-party SOTA code used for comparison (BDT, DSPNet, 3DT-Net, MoG-DCN, SSRNet, ...)
├── experiments/
│   ├── scripts/        Training scripts (train_*.py)
│   ├── notebooks/      Kaggle / Colab notebooks
│   └── kaggle/         Kaggle kernel folders
├── results/            Result JSONs and comparison tables (weights are not tracked)
├── docs/               Survey, protocol audit, datasets, architecture notes, theory write-up
├── paper/              Manuscript draft, bibliography, table and theorem scripts
├── tools/              Notebook builders and Kaggle automation
├── tests/              Quick model and forward-pass checks
└── archive/            Superseded notebooks and one-off scripts
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

## License

Research use. Third-party code in `baselines/` keeps its original license (see each subfolder).
The datasets belong to their respective providers.
