# Proposal 8: ASON — Adaptive Spectral Operator Network

## Overview

ASON (Adaptive Spectral Operator Network) is a rectified-flow-on-consistent-set framework for HSI-MSI image fusion. It composes verified modules from earlier proposals into a unified architecture:

- **DegradationCode** (from DAETF/Proposal 1): encodes LR-HSI + MSI into a compact degradation embedding
- **VelocityNet** (from ManifoldFlow/Proposal 5): predicts rectified flow velocity with FiLM conditioning on degradation code
- **RangeNullProjector** (from SpectralFlow/Proposal 5): enforces consistency D(y) = X during sampling

## Architecture

```
LR-HSI (B, C_lr, H_lr, W_lr) ──┐
                                 ├──► DegradationCode ──► code (B, 64)
MSI (B, M, H_lr, W_lr) ────────┘
                                          │
                                          ▼
              ┌──────────────────────────────────────────┐
              │  Rectified Flow ODE (4 steps default)    │
              │  y_0 = LR-HSI                           │
              │  for k in steps:                         │
              │    t = (k+0.5)/steps                     │
              │    v = VelocityNet(y, msi, t, code)      │
              │    v = NullProjector.project_null(v)     │
              │    y = y + v / steps                     │
              └──────────────────────────────────────────┘
                                          │
                                          ▼
                              HR-HSI (B, C, H, W)
```

## Loss Function

L = λ_v * MSE(v_pred, v_target)          # flow matching
  + λ_l1 * Charbonnier(y_sampled, GT)     # reconstruction
  + λ_c * Charbonnier(S(y_sampled), MSI)  # SRF consistency
  + λ_s * SAM(y_sampled, GT)              # spectral fidelity

## Datasets

- **Chikusei**: 128 bands, 1 scene, patch-based training
- **CAVE**: 31 bands, 32 scenes (20 train, 12 test)

## Protocol

- Scale: ×4
- Degradation: Gaussian blur σ=1.2 + bilinear downsampling
- SRF: 3-band Gaussian (Chikusei), learned (CAVE)
- Evaluation: Wald protocol

## Files

- `ason/__init__.py` — package exports
- `ason/config.py` — Config dataclass
- `ason/model.py` — ASONNet, VelocityNet, DegradationCode, RangeNullProjector
- `ason/losses.py` — ASONLoss (flow matching + physics + spectral)
- `ason/engine.py` — train(), evaluate_dataset()
- `notebooks/ASON_Kaggle_GPU.ipynb` — self-contained Kaggle notebook

## Running

```bash
# On Kaggle (GPU T4 x2 recommended):
# 1. Open notebooks/ASON_Kaggle_GPU.ipynb
# 2. Select GPU T4 x2 in Settings
# 3. Run All

# Locally:
python -c "from proposal8.ason import Config, ASONNet; print('OK')"
```

## Status

- Code: complete
- Kaggle training: running (T4 GPU)
- Results: pending
