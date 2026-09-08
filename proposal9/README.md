# Proposal 9: DACF — Degradation-Adaptive Conditional Flow

## Overview

DACF estimates degradation characteristics directly from observed LR-HSI and MSI at test time, and uses this estimate to condition a normalizing flow network — no retraining, no per-scene optimization, single forward pass.

## Core Idea

> The LR-HSI and MSI observations implicitly encode the degradation characteristics. A lightweight encoder extracts a "degradation code", and a flow network conditioned on this code adapts its fusion behavior.

## Architecture

```
LR-HSI ──┐
          ├──► DegradationEncoder ──► code [B, 64]
MSI ─────┘
                    │
                    ▼
          ConditionalFlow (8 coupling layers with FiLM)
          z_0 = LR-HSI → z_K = HR-HSI
                    │
                    ▼
          NullProjector: D(output) ≈ LR-HSI
```

## Training

**Phase 1 — Flow Matching (offline):**
- Train on PaviaU/Chikusei patches with randomized degradation (blur σ ∈ [0.5, 3.0], noise ∈ [0, 0.05])
- Loss: flow matching + cycle consistency + SRF consistency + spectral fidelity

**Phase 2 — Self-Supervised Adaptation (test time):**
- On new observations, update only the degradation encoder
- Enforce: D(fused) ≈ LR-HSI, S(fused) ≈ MSI
- ~100 iterations, seconds not minutes

## Datasets

| Dataset | Bands | Spatial | Scenes | Kaggle Slug |
|---------|-------|---------|--------|-------------|
| PaviaU | 103 | 610×340 | 1 | `syamkakarla/pavia-university-hsi` |
| Chikusei | 128 | 2335×2517 | 1 | `mingliu123/chikusei` |

## Files

- `dacf/__init__.py` — package exports
- `dacf/config.py` — Config dataclass
- `dacf/encoder.py` — DegradationCode encoder
- `dacf/flow.py` — Conditional normalizing flow with FiLM
- `dacf/model.py` — DACFNet (encoder + flow + projector)
- `dacf/losses.py` — DACFLoss (phase1 + phase2)
- `dacf/engine.py` — train(), adapt(), evaluate_dataset(), data loaders
- `notebooks/DACF_Kaggle_PaviaU_Chikusei.ipynb` — self-contained notebook
- `docs/ARCHITECTURE.md` — design documentation
- `README.md` — this file

## Running

```bash
# On Kaggle: select GPU T4 x2, then Run All
# Locally:
python -c "from proposal9.dacf import Config, DACFNet; print('OK')"
```
