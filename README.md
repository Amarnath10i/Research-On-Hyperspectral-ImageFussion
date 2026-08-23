# Hyperspectral–Multispectral Image Fusion: Identifiability, Ambiguity, and Sensor-Shift Theory

**A unified theoretical and experimental framework** for HSI–MSI fusion that answers four open scientific questions, validated on four public datasets (CAVE, Harvard, Chikusei, PaviaU) with cross-domain zero-shot experiments and a reproducible protocol.

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
- **Band-count agnostic**: same code trains on CAVE (31), Harvard (31), Chikusei (128), PaviaU (103)
- Physics losses: `‖D(x̂) - Y_H‖² + ‖S(x̂) - Y_M‖² + 0.1·‖x̂ - X‖₁ + 0.1·‖r_m‖`

### 2. **KrylovNet-P** — SOTA-Capable Variant (~1.38M parameters)
- Plug-and-play learned proximal prior interleaved with unrolled solver
- Zero-initialised denoiser (8 ResBlocks, width 96) starts at solver's answer
- EMA (decay 0.999), gradient accumulation ×2, cosine LR decay
- Trained under **published protocol**: Wald simulation + Nikon D700 SRF, CAVE ×4
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
- **Zero-shot**: CAVE↔Harvard cross-domain with **no sensor-induced drop** (Thm 5 verified)
- **Ambiguity audit**: KrylovNet H=0.19–0.28 (only method with H<1 on all datasets)
- **Phase transition**: `r̂_id(M)` monotone and capped by M on all datasets

---

## Theoretical Framework

### P1: Admissible Ambiguity (`proposal1/ambiguity/`)
The combined operator `A = [D; R^T]` admits an exact range/null decomposition:
```
X_obs = A^T(A A^T)⁻¹ Y      (pinned by data)
X_amb = P_N v = (I - A^T(A A^T)⁻¹ A)v   (genuinely free)
```
with `A(X_obs + X_amb) = Y` for ANY v. The **hallucination metric** `H = ‖P_N(X̂−X)‖/(‖P_N X‖+ε)` is computable from observations alone and correlates with SAM error.

### P2: Identifiable Rank (`proposal2/rankest/`)
The observable spectral rank is `r_id = rank(R^T U_r)`, **not** the intrinsic scene rank `r`. We prove:
- `r̂_id` recovers `r_id` with exponential tail bound (Thm 1)
- Optimal reconstruction rank is `r_id` (Thm 2)
- The phase transition `M*(r)` predicts when identification is possible (Thm 4)

### P3: Sensor-Shift Bound (`proposal3/field/`)
For a continuous spectral field `F(x,y,λ)` with Lipschitz constant `L_F`:
```
Δ_sensor ≤ L_F · EMD(s_src, s_tgt) + noise
```
Two sensors are compatible for zero-shot transfer iff `EMD(s_src, s_tgt) < ε / L_F`. Explains why smooth fields transfer; sharp absorption edges do not.

### P4: Phase Transition (`proposal4/identifiability/`)
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
| **CAVE** (31 b) | 29.93 / 0.888 / 4.89 | 34.38 / 0.924 / 7.09 | 33.55 / 0.946 / 4.68 | **40.85 / 0.983 / 3.39** |
| **Harvard** (31 b) | 60.31 / 0.997 / 2.59 | 66.46 / 0.999 / 2.75 | 64.23 / 0.999 / 2.41 | **70.78 / 1.000 / 2.30** |
| **Chikusei** (128 b) | 33.58 / 0.897 / 14.24 | 34.90 / 0.914 / 13.80 | 34.77 / 0.919 / 12.90 | **43.69 / 0.983 / 6.06** |
| **PaviaU** (103 b) | 25.40 / 0.713 / 15.05 | 25.99 / 0.743 / 14.24 | 26.70 / 0.784 / 12.78 | **34.48 / 0.952 / 4.45** |

*Format: PSNR (dB) / SSIM / SAM (degrees). KrylovNet +11/+6/+9/+8 dB over best baseline on CAVE/Chikusei/PaviaU.*

### Zero-Shot Cross-Domain (CAVE ↔ Harvard, Same Simulated SRF)

| Direction | PSNR | SSIM | SAM | ERGAS | vs In-Domain |
|-----------|------|------|-----|-------|--------------|
| CAVE → Harvard | 70.72 | 0.9998 | 2.30 | 1.68 | **−0.06 dB** |
| Harvard → CAVE | 40.85 | 0.9831 | 3.40 | 2.33 | **+0.00 dB** |

*Sensor EMD = 0 (identical SRF) → Thm 5 predicts zero sensor-induced drop — verified.*

### Identifiability & Ambiguity Audit

| Dataset | `r̂_id` (mean) | Ambiguity Energy | KrylovNet **H** | Subspace-LS H | GSA H | Bicubic H |
|---------|---------------|------------------|----------------|---------------|-------|-----------|
| CAVE | 0.0 | 0.182 | **0.190** | 0.764 | 0.606 | 1.295 |
| Harvard | 0.0 | 0.168 | **0.229** | 0.639 | 0.542 | 1.056 |
| Chikusei | 2.0 | 0.123 | **0.269** | 1.042 | 1.127 | 1.203 |
| PaviaU | 2.0 | 0.192 | **0.281** | 1.265 | 1.424 | 1.510 |

- **H < 1** = under-fills null space (safe); **H > 1** = over-fills (hallucinates)
- KrylovNet is the **only method with H < 1 on all datasets**
- Lowest H ↔ lowest SAM on 4/4 datasets (Thm 3 coupling verified)

### Phase Transition (`r̂_id(M)` monotone, capped by M)

| Dataset | `r̂_id(M=1..8)` | Monotone | Capped by M |
|---------|----------------|----------|-------------|
| CAVE | [0,0,0,0,0,0,0,0] | ✓ | ✓ |
| Harvard | [0,0,0,0,0,0,0,0] | ✓ | ✓ |
| Chikusei | [1,2,2,2,2,2,2,2] | ✓ | ✓ |
| PaviaU | [1,1,2,2,2,2,2,2] | ✓ | ✓ |

---

## Repository Structure

```
├── common/hsifusion/           # Shared library (baselines, data, metrics, SRF, engine)
│   ├── baselines.py            # Bicubic, GSA, Subspace-LS
│   ├── data.py                 # Dataset loading, SRF estimation, train/test splits
│   ├── degrade.py              # FixedDegradation (evaluation operator)
│   ├── engine.py               # Training loop, tiled inference, checkpointing
│   ├── losses.py               # Physics + fidelity losses (SAM, SSIM, L1)
│   ├── metrics.py              # PSNR, SSIM, SAM, ERGAS with fixed data_range=1.0
│   ├── srf.py                  # Nikon D700 SRF, Gaussian SRF, SRF estimation
│   ├── config.py               # Shared Config classes
│   └── checkpoint.py           # Resume/load/save with EMA
│
├── proposal1/ambiguity/        # P1: Admissible Ambiguity
│   ├── operator.py             # CombinedOperator A=[D;R] with range/null projectors
│   ├── selfcheck.py            # Verifies A(X_obs+X_amb)=Y identity (~1e-5)
│   └── docs/ARCHITECTURE.md
│
├── proposal2/                  # P2: Identifiable Rank + KrylovNet
│   ├── rankest/                # r_id estimator, recovery guarantee (Thm 1)
│   ├── krylovnet/              # KrylovNet / KrylovNet-P (2.3k / 1.38M params)
│   │   ├── model.py            # Unrolled GMRES + GNN preconditioner + PnP prior
│   │   ├── solver.py           # GMRES, Richardson, Blend, FusionOperator
│   │   ├── engine.py           # Training/eval with EMA, checkpoint resume
│   │   ├── config.py           # Config with all hyperparameters
│   │   ├── selfcheck.py        # Verifies solver + prior properties
│   │   └── notebooks/          # SOTA push notebook (CAVE ×4 Nikon)
│   ├── experiments/            # Sweeps: noise, band count, SRF, synthetic rank
│   └── theory/                 # Proofs for Thm 1, 2, 3, 4
│
├── proposal3/                  # P3: Sensor-Shift Bound
│   ├── field/                  # Continuous spectral field + neural field
│   │   ├── field.py            # SceneField, Sensor operators, EMD
│   │   ├── neural_field.py     # SIREN-style continuous field F_θ(x,y,λ)
│   │   ├── sensors.py          # SRF distributions, EMD computation
│   │   ├── theorem.md          # Δ_sensor ≤ L_F · EMD proof
│   │   └── selfcheck.py        # Zero-shot transfer on smooth fields
│   └── continuumfusion/        # INR-based SOTA attempt (FeINFN-like)
│
├── proposal4/identifiability/  # P4: Phase Transition
│   ├── simulator.py            # Synthetic scenes across (r, M) grid
│   ├── phasediagram.py         # Empirical vs analytical M*(r) boundary
│   ├── theorem.md              # Phase transition proof (M*(r) monotone)
│   └── selfcheck.py            # Verifies monotone r_id, capped by M
│
├── proposal5/                  # P5: SpectralFlow / ManifoldFlow
│   ├── spectralflow/           # DDIM sampler + null-space projection
│   ├── manifoldflow/           # Manifold-constrained diffusion
│   └── docs/ARCHITECTURE.md
│
├── proposal6/consistentflow/   # P6: Consistency-constrained flow
│   └── sampler.py              # Langevin + projection sampler
│
├── proposal7/nullfusion/       # P7: NullFusion (Q1 method)
│   ├── model.py                # NullFusionNet: exact pinv + null prior
│   ├── train_sota.py           # Full SOTA training pipeline
│   ├── selfcheck.py            # Verifies exact consistency, gradients
│   ├── notebooks/              # Kaggle notebook (CAVE ×4 Nikon)
│   └── docs/ARCHITECTURE.md
│
├── existing/                   # 10 benchmarked methods (reproducibility)
├── literature_survey/          # Crossref → Unpaywall → annotated pipeline
├── paper/                      # research_paper.md (full manuscript)
├── review/                     # Internal review notes
├── tools/                      # Utility scripts
│
├── MultiDataset_Fusion_Study.ipynb  # Main experiment notebook (all 4 datasets)
├── README.md                   # This file
├── SOTA_COMPARISON.md          # Protocol-audited SOTA table + 2026 sweep
├── PROTOCOL_AUDIT.md           # One-protocol rules, metrics, statistics
├── PROPOSAL_POSITIONING.md     # How the 7 proposals connect
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
$env:PYTHONPATH="common;proposal1;proposal2;proposal3;proposal4;proposal5;proposal7"
python -c "import proposal1.ambiguity as a; a._selfcheck.run_all()"     # P1
python -c "import proposal2.krylovnet as k; k.selfcheck.run_all()"      # P2
python -c "import proposal3.field as f; f._selfcheck.run_all()"         # P3
python -c "import proposal4.identifiability as i; i._selfcheck.run_all()" # P4
python -c "import proposal7.nullfusion as n; n.selfcheck.run_all()"     # P7
```

### Run Experiments (CPU)
```powershell
python -m proposal2.experiments.synthetic_rank_sweep
python -m proposal2.experiments.noise_sweep
python -m proposal2.experiments.band_count_sweep
python -m proposal2.experiments.srf_sweep
python -m proposal4.identifiability.phasediagram
```

### Kaggle GPU Training (Recommended)

**Main 4-Dataset Run:**
1. Upload `MultiDataset_Fusion_Study.ipynb` to Kaggle
2. Attach datasets: CAVE (`liptee/cave`), Harvard (`nikeshreddypatlolla/harvard-hsi-2`), Chikusei (`mingliu123/chikusei`), PaviaU (`syamkakarla/pavia-university-hsi`)
3. Set **Accelerator → GPU T4 x2**
4. Run all cells (self-contained via `%%writefile` library cells)

**SOTA Push (CAVE ×4 Nikon):**
- Kernel: `sandeepchowdary2005/sota-krylovnet-cave-nikon` (or `proposal7/notebooks/nullfusion_SOTA_CAVE_Nikon.ipynb`)
- Uses `USE_NIKON_SRF = True`, time-budgeted training with checkpoint-and-resume
- Target: FeINFN 52.47 / BDT 52.30 dB

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

Research use. Datasets: CAVE (Columbia), Harvard (Harvard), Chikusei (JAXA), PaviaU (University of Pavia). Code: MIT-style for research purposes.

---

## Contact

For questions, reproducibility, or collaboration: **amarnathmadaka** (GitHub) / project maintainers.

*This README reflects the state of the repository as of 2026-08-23. All theoretical claims are verified by self-checks; all experimental numbers are reproducible via the provided notebooks under the fixed protocol.*