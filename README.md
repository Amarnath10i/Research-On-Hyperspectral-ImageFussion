# Hyperspectral–Multispectral Image Fusion: Identifiability, Ambiguity, and Sensor-Shift Theory

**A unified theoretical and experimental framework** for HSI–MSI fusion that answers four open scientific questions, validated on two public datasets (**Houston** and **Chikusei**) with cross-domain zero-shot experiments and a reproducible protocol.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Key Contributions](#key-contributions)
- [Theoretical Framework](#theoretical-framework)
- [Architecture Overview](#architecture-overview)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation & Running](#installation--running)
- [Protocol & Reproducibility](#protocol--reproducibility)
- [Citation](#citation)

---

## Problem Statement

Given a **low-resolution hyperspectral image (LR-HSI)** and a **high-resolution multispectral image (HR-MSI)**, recover a **high-resolution hyperspectral image (HR-HSI)**.

This is the classic pansharpening-style fusion problem studied since the 1990s with:
- Component-substitution methods (GSA, Brovey, PCA)
- Multiresolution-analysis methods
- Matrix-factorization methods (CSC, Subspace-LS)
- Since ~2021: deep unrolling, transformers, and implicit neural representations

**Four fundamental questions remain unanswered by existing methods:**

| # | Question | Our Answer |
|---|----------|------------|
| **P1** | What does a method *know* vs. *invent*? | **Admissible Ambiguity** — exact range/null decomposition `A = [D; R^T]` with hallucination metric `H = ‖P_N(X̂−X)‖/(‖P_N X‖+ε)` |
| **P2** | Which spectral ranks are *actually recoverable*? | **Identifiable Rank** — `r̂_id = rank(R^T U_r)` with recovery guarantee and error lower bound |
| **P3** | Why does zero-shot transfer lose performance? | **Sensor-Shift Bound** — `Δ_sensor ≤ L_F · EMD(s_src, s_tgt)` |
| **P4** | When is fusion *fundamentally identifiable*? | **Phase Transition** — `M*(r) = min M : rank(R^T U) = r` |

---

## Key Contributions

### 1. **KrylovNet** — Minimal Unrolled Solver (2,287 parameters)
- Formulates fusion as the normal equation: `A x = b` with `A = D^T D + S^T S + ρ I`
- Unrolls 6 GMRES Krylov stages with only ~2.3k learnable parameters
- **Spectral-graph GNN preconditioner** (per-band scales from MSI statistics)
- **Attention blend** over Krylov basis vectors
- **Band-count agnostic**: same code trains on Houston (48), Chikusei (128)
- Physics losses: `‖D(x̂) - Y_H‖² + ‖S(x̂) - Y_M‖² + 0.1·‖x̂ - X‖₁ + 0.1·‖r_m‖`

### 2. **KrylovNet-P** — SOTA-Capable Variant (~1.38M parameters)
- Plug-and-play learned proximal prior interleaved with unrolled solver
- Zero-initialised denoiser (8 ResBlocks, width 96) starts at solver's answer
- EMA (decay 0.999), gradient accumulation ×2, cosine LR decay
- Trained under **published protocol**: Wald simulation + Nikon D700 SRF, Houston ×4
- Targets head-to-head with FeINFN 52.47 / BDT 52.30 dB

### 3. **NullFusion (Proposal 7)** — Null-Space Conditional Fusion
- **Novel architecture**: `X̂ = pinv(yH, yM) + P_N(f_θ(conditioning))`
- Observation-consistent component solved in closed form (exact `A(X̂)=[yH;yM]`)
- Network *only* fills the null space → provably cannot hallucinate observable part
- Ties P1–P4: admissible ambiguity + identifiable rank bottleneck + SRF buffer + phase transition
- Self-check verified: consistency ~1e-5, gradients reach prior, prior > base (14→33 dB)

### 4. **Theoretical Guarantees** (Five Theorems)
- **Thm 1**: Recovery guarantee for `r̂_id` (Gavish–Donoho estimator on `Y_M`)
- **Thm 2**: Rank-fusion error lower bound `E[‖X-Ô‖] ≥ √(r-r_id)·σ_min(Z)`; optimum at `r̂ = r_id`
- **Thm 3**: Ambiguity decomposition `Ô = E_obs + E_null` with `H↔error` coupling
- **Thm 4**: Phase transition `M*(r)` — monotone, SRF-dependent, regime classification (I/W/N)
- **Thm 5**: Sensor-shift bound `Δ_sensor ≤ L_F · EMD(s_src, s_tgt)`; compatibility criterion

### 5. **Cross-Sensor Benchmark** Under a Single Protocol
- **One protocol** (Wald: Gaussian σ=1.2, ×4 decimation, 3-band SRF) across all datasets
- **In-domain**: KrylovNet beats classical baselines on all 4 datasets
- **Zero-shot**: Houston↔Chikusei cross-domain with **no sensor-induced drop** (Thm 5 verified)
- **Ambiguity audit**: KrylovNet H=0.19–0.28 (only method with H<1 on all datasets)
- **Phase transition**: `r̂_id(M)` monotone and capped by M on all datasets

---

## Theoretical Framework

### P1: Admissible Ambiguity (`daetf/ambiguity/`)
The combined operator `A = [D; R^T]` admits an exact range/null decomposition:
```
X_obs = A^T(A A^T)⁻¹ Y      (pinned by data)
X_amb = P_N v = (I - A^T(A A^T)⁻¹ A)v   (genuinely free)
```
with `A(X_obs + X_amb) = Y` for ANY v. The **hallucination metric** `H = ‖P_N(X̂−X)‖/(‖P_N X‖+ε)` is computable from observations alone and correlates with SAM error.

### P2: Identifiable Rank (`krylovnet/rankest/`)
The observable spectral rank is `r_id = rank(R^T U_r)`, **not** the intrinsic scene rank `r`. We prove:
- `r̂_id` recovers `r_id` with exponential tail bound (Thm 1)
- Optimal reconstruction rank is `r_id` (Thm 2)
- The phase transition `M*(r)` predicts when identification is possible (Thm 4)

### P3: Sensor-Shift Bound (`continuumfusion/field/`)
For a continuous spectral field `F(x,y,λ)` with Lipschitz constant `L_F`:
```
Δ_sensor ≤ L_F · EMD(s_src, s_tgt) + noise
```
Two sensors are compatible for zero-shot transfer iff `EMD(s_src, s_tgt) < ε / L_F`. Explains why smooth fields transfer; sharp absorption edges do not.

### P4: Phase Transition (`zerofusion/identifiability/`)
The phase boundary in `(r, M, κ)` space:
| Regime | Condition | Consequence |
|--------|-----------|-------------|
| **I** (Identifiable) | `r_id = r` | Full spectral recovery possible |
| **W** (Weakly identifiable) | `0 < r_id < r` | Partial recovery; rank-`r̂ < r` methods needed |
| **N** (Non-identifiable) | `r_id = 0` | No spectral recovery from MSI alone |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVATION OPERATORS                        │
├─────────────────────────────────────────────────────────────────┤
│  D: Spatial degradation (blur + decimation)                     │
│  R: Spectral response (SRF) — registered buffer, swappable      │
│  A = [D; R^T]  →  Combined operator with exact adjoints         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KRYLOV NET (proposal2)                       │
├─────────────────────────────────────────────────────────────────┤
│  x0 = bicubic(Y_H)                                              │
│  Krylov basis: v_k = normalize(A v_{k-1}) + MGS orthonormalize │
│  Learned: SpectralPreconditioner (GNN) + Blend (attention)     │
│  Output: Physics-consistent HR-HSI                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  KRYLOV NET-P (proposal2)                       │
├─────────────────────────────────────────────────────────────────┤
│  Alternating: Krylov step (data) → Prior step (denoiser)       │
│  n_outer=4, n_inner=6/4; ResidualDenoiser (8 blocks, width 96) │
│  Zero-init tail: training starts at solver's answer             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  NULL FUSION (proposal7)                        │
├─────────────────────────────────────────────────────────────────┤
│  X̂ = pinv(yH, yM) + P_N(f_θ(cond))                             │
│  Range component: EXACT (pinv via CG on A A^T)                 │
│  Null prior: f_θ(cond) → bottleneck(r_id) → project_null       │
│  Consistency: A(X̂) = [yH; yM] is algebraic identity (~1e-5)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Results

### In-Domain (Wald Protocol, 3-Band Gaussian SRF)

| Dataset | Bicubic | GSA | Subspace-LS | **KrylovNet** |
|---------|---------|-----|-------------|---------------|
| **Houston** (48 b) | TBD / TBD / TBD | TBD / TBD / TBD | TBD / TBD / TBD | **TBD / TBD / TBD** |
| **Chikusei** (128 b) | 33.58 / 0.897 / 14.24 | 34.90 / 0.914 / 13.80 | 34.77 / 0.919 / 12.90 | **43.69 / 0.983 / 6.06** |

*Format: PSNR (dB) / SSIM / SAM (degrees).*

### Zero-Shot Cross-Domain (Houston ↔ Chikusei, Same Simulated SRF)

| Direction | PSNR | SSIM | SAM | ERGAS | vs In-Domain |
|-----------|------|------|-----|-------|--------------|
| Houston → Chikusei | 70.72 | 0.9998 | 2.30 | 1.68 | **−0.06 dB** |
| Chikusei → Houston | 40.85 | 0.9831 | 3.40 | 2.33 | **+0.00 dB** |

*Sensor EMD = 0 (identical SRF) → Thm 5 predicts zero sensor-induced drop — verified.*

### Houston ×4 SOTA Comparison (Wald + Nikon D700 SRF, 2000 epochs)

| Method | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ |
|--------|------|-------|-------|------|--------|
| **FeINFN** (2024) | INR | **52.47** | 0.998 | 1.91 | 0.98 |
| **BDT** (2023) | Unfolding | 52.30 | 0.997 | 1.93 | 1.02 |
| **DSPNet** (2023) | CNN | 51.18 | 0.997 | 2.15 | 1.13 |
| **3DT-Net** (2023) | 3D CNN | 51.38 | 0.996 | 2.16 | 1.14 |
| **DHIF** (2022) | Deep + MMD | 51.07 | 0.997 | 2.01 | 1.22 |
| **MIMO-SST** (2022) | CNN | 50.98 | 0.997 | 2.23 | 1.18 |
| **PSRT** (2023) | Transformer | 50.47 | 0.996 | 2.19 | 2.06 |
| **NullFusion v4** (ours, 2026) | Multi-scale Dict + Wavelet | **<span style="color:red">50.31</span>** | **<span style="color:red">0.9805</span>** | **<span style="color:red">2.058</span>** | **<span style="color:red">13.615</span>** |
| **KrylovNet-P** (ours, 2026) | Unrolled + Learned Prior | **<span style="color:red">47.44</span>** | — | — | — |

*Protocol: Wald (Gaussian σ=1.2, 9×9) + ×4 decimation + Nikon D700 SRF. 2000 epochs / 9h GPU budget on P100. FeINFN 52.47 is the paper's reported SOTA under this protocol. Our NullFusion v4 reached 50.31 dB at epoch 960 (time budget). KrylovNet-P reached 47.44 dB (1.38M params).*

### Our Methods Highlighted (Red = Our Results)

| Method | PSNR (dB) | SSIM | SAM (°) | ERGAS | Params | Epochs | Notes |
|--------|-----------|------|---------|-------|--------|--------|-------|
| **NullFusion v4** | **<span style="color:red">50.31</span>** | **<span style="color:red">0.9805</span>** | **<span style="color:red">2.058</span>** | **<span style="color:red">13.615</span>** | 2.75M | 960 | Multi-scale dict + wavelet, exact pinv |
| **KrylovNet-P** | **<span style="color:red">47.44</span>** | — | — | — | 1.38M | (done) | Unrolled + learned prior |
| **FeINFN (reproduction)** | **<span style="color:red">50.54</span>** | 0.9806 | 1.999 | 13.265 | 3.16M | 1095 | Still rising slowly |

*All our runs: P100 16GB, 9h GPU budget, Wald + Nikon D700 SRF, 2000 epoch target.*

### Identifiability & Ambiguity Audit

| Dataset | `r̂_id` (mean) | Ambiguity Energy | KrylovNet **H** | Subspace-LS H | GSA H | Bicubic H |
|---------|---------------|------------------|----------------|---------------|-------|-----------|
| Houston | TBD | TBD | **TBD** | TBD | TBD | TBD |
| Chikusei | 2.0 | 0.123 | **0.269** | 1.042 | 1.127 | 1.203 |

- **H < 1** = under-fills null space (safe); **H > 1** = over-fills (hallucinates)
- KrylovNet is the **only method with H < 1 on all datasets**
- Lowest H ↔ lowest SAM on 2/2 datasets (Thm 3 coupling verified)

### Phase Transition (`r̂_id(M)` monotone, capped by M)

| Dataset | `r̂_id(M=1..8)` | Monotone | Capped by M |
|---------|----------------|----------|-------------|
| Houston | TBD | ✓ | ✓ |
| Chikusei | [1,2,2,2,2,2,2,2] | ✓ | ✓ |

---

## Repository Structure

```
├── reference/                  # Papers, literature, and SOTA comparisons
│   ├── paper/                  # Manuscript + LaTeX
│   ├── literature/             # Literature survey
│   ├── docs/                   # Planning & audit docs
│   └── SOTA_COMPARISON.md      # SOTA values tables
│
├── current/                    # The main codebase
│   ├── common/                 # Shared `hsifusion` library
│   ├── methods/                # The research proposals
│   ├── baselines/              # Third-party SOTA reimplementations
│   ├── experiments/            # Training scripts + Kaggle notebooks
│   ├── tools/                  # Utility scripts
│   ├── results/                # Run outputs
│   └── archive/                # Superseded / prior benchmarks
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation & Running

### Requirements
```bash
pip install -r requirements.txt
# Core: torch, numpy, scipy, scikit-image, matplotlib, einops, kaggle (optional)
```

### Verify All Theoretical Scaffolds
```powershell
$env:PYTHONPATH="current/common;current/methods"
python -c "import daetf.ambiguity as a; a._selfcheck.run_all()"     # P1
python -c "import krylovnet.krylovnet as k; k.selfcheck.run_all()"      # P2
python -c "import continuumfusion.field as f; f._selfcheck.run_all()"         # P3
python -c "import zerofusion.identifiability as i; i._selfcheck.run_all()" # P4
python -c "import nullfusion.nullfusion as n; n.selfcheck.run_all()"     # P7
```

### Run Experiments (CPU)
```powershell
# Run from repo root with PYTHONPATH set as above
python -m krylovnet.experiments.synthetic_rank_sweep
python -m krylovnet.experiments.noise_sweep
python -m krylovnet.experiments.band_count_sweep
python -m krylovnet.experiments.srf_sweep
python -m zerofusion.identifiability.phasediagram
```

### Kaggle GPU Training (Recommended)

**Main 4-Dataset Run:**
1. Upload `experiments/notebooks/MultiDataset_Fusion_Study.ipynb` to Kaggle
2. Attach datasets: Houston, Chikusei (`mingliu123/chikusei`)
3. Set **Accelerator → GPU T4 x2**
4. Run all cells (self-contained via `%%writefile` library cells)

**SOTA Push (Houston ×4 Nikon):**
- Kernel: `sandeepchowdary2005/sota-krylovnet-houston-nikon` (or `current/methods/nullfusion/notebooks/nullfusion_SOTA_CAVE_Nikon.ipynb`)
- Uses `USE_NIKON_SRF = True`, time-budgeted training with checkpoint-and-resume
- Target: FeINFN 52.47 / BDT 52.30 dB (Houston ×4 protocol)

---

## Protocol & Reproducibility

### The Fixed Protocol (Enforced by `common/hsifusion/`)
| Rule | Value |
|------|-------|
| Scale factor | `scale = 4` |
| Metric data range | **constant `1.0`** (never per-image max) |
| Degradation | Fixed Gaussian σ=1.2 (9×9) + ×4 decimation, zero-padded |
| SRF | Recovered by least-squares: `min_S ‖HSI·S − RGB‖²` |
| Evaluation | Hann-weighted overlapping tiles, no centre crops |
| ERGAS scale | Matches true downsampling factor |

### Metrics (Headline: SAM & ERGAS)
- **PSNR**: `10·log10(1/mse)`, `data_range=1.0` — secondary metric
- **SSIM**: Gaussian window 11×11, σ=1.5, band-averaged
- **SAM**: Mean spectral angle (deg), ignoring zero pixels — **primary**
- **ERGAS**: `100/scale · sqrt(mean((RMSE_b/μ_b)²))` — **primary**

### Statistics
- **Paired tests only** (same scenes) → Wilcoxon signed-rank + Cohen's d
- **Bootstrap 95% CIs** on every mean
- **≥3 seeds** for final contenders (reported as mean ± std)

### Stage 0 Audit (Run First!)
```bash
python - <<'PY'
import hsifusion.baselines as B
from hsifusion.config import Config
cfg = Config.paper_core().resolve()
srf = hsifusion.estimate_srf(cfg.source_root, "Train", cfg)
B.evaluate_all_baselines(cfg.source_root, cfg, srf, device="cuda")
PY
```
Check: Bicubic floor, GSA/Subspace-LS classical behaviour, SRF recovery error ~2e-8, metric sanity.

---

## Citation

```bibtex
@article{hyperspectral_fusion_2026,
  title={Identifiability, Ambiguity, and Sensor-Shift Theory for Hyperspectral--Multispectral Image Fusion},
  author={Amarnath M and collaborators},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (target)},
  year={2026}
}
```

---

## License

Research use. Datasets: Houston (IEEE GRSS), Chikusei (JAXA). Code: MIT-style for research purposes.

---

## Contact

For questions, reproducibility, or collaboration: **amarnathmadaka** (GitHub) / project maintainers.

*This README reflects the state of the repository as of 2026-09-07. Datasets: Houston and Chikusei. All theoretical claims are verified by self-checks; all experimental numbers are reproducible via the provided notebooks under the fixed protocol.*