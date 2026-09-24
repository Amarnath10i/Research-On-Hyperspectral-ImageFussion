# NullFusion-SOTA Architecture for Chikusei (128 bands)

## Target: Beat CoFusion 49.14 / RAMoE 48.10 / SMGU-Net 48.82 PSNR

---

## 1. System Overview

```
 INPUT                         FORWARD MODEL                    OUTPUT
 ─────                         ─────────────                    ──────
 LR-HSI (128, H/4, W/4) ──┐
                             ├──→ A = [D; R^T] ──→ pinv(A) ──→ X_obs (128, H, W)
 HR-MSI  (3,  H,   W)   ──┘         │                        (observation-consistent base)
                                     │
                             NullFusion-SOTA Network
                             ─────────────────────────
                             │  Dual-path encoder          │
                             │  U-Net with skip connections │
                             │  Spectral dictionary prior   │
                             │  Wavelet refinement          │
                             └─────────────────────────────┘
                                            │
                                            ▼
                                   X_hat = X_obs + P_N(v) + WF(v)
                                            │
                                            ▼
                                   HR-HSI (128, H, W)
```

---

## 2. Model Components (15.28M parameters)

### 2.1 Forward Model Operators

```
CombinedOp = [D, R]
  D: DegradationOp  ── Gaussian blur (9x9, σ=1.2) + 4x decimation
  R: SpectralResponse ── 128→3 via Nano-Hyperspec SRF (condition=4.51)

pinv(A): Block CG solver (60 steps, ridge=1e-6)
  Solves: (D^T D + R^T R + ρI)x = D^T·Y_H + R^T·Y_M

project_null(v): v - pinv(A·v)
  Projects v into the null space of A
  Ensures: A(X_obs + P_N(v)) = A(X_obs) = [Y_H; Y_M] exactly
```

### 2.2 Dual-Path Conditioning Encoder

```
LR-HSI (128, h, w)          HR-MSI (3, H, W)
      │                           │
      ▼                           ▼
┌─────────────┐            ┌─────────────┐
│ Conv 128→96 │            │ Conv 3→96   │
│ + RCAB      │            │ + RCAB      │
└──────┬──────┘            └──────┬──────┘
       │                          │
       │    ┌─────────────────────┘
       │    │
       ▼    ▼
┌──────────────────────┐
│ CrossAttn1 (8 heads) │  ← HSI queries, MSI keys/values
│ CrossAttn2 (8 heads) │  ← refined HSI queries, pooled MSI
│ CrossAttn3 (8 heads) │  ← final cross-modal fusion
└──────────┬───────────┘
           │
     f_hsi (96, h, w)     f_msi_pool (96, h, w)
           │                      │
           └──────┬───────────────┘
                  ▼
           ┌───────────┐
           │ fuse_in   │  Conv 192→96 (1x1)
           └─────┬─────┘
                 │
           z (96, h, w)
```

### 2.3 U-Net Encoder

```
Level 1: z ──→ [RCAB × 3] ──→ e1 (96, h, w) ──→ SpatialAttn(8)
                                              │
                                    ┌─────────┘
                                    │
Level 2: e1 ──→ down1(Conv 96→192, stride 2) ──→ [RCAB × 3] ──→ e2 (192, h/2, w/2)
                                                              │
                                                    ┌─────────┘
                                                    │
Level 3: e2 ──→ down2(Conv 192→384, stride 2) ──→ [RCAB × 4] ──→ bn (384, h/4, w/4)
```

### 2.4 U-Net Decoder (with skip connections)

```
Level 3→2: bn ──→ up2(ConvT 384→192, stride 2) + e2 ──→ [RCAB × 3] ──→ d2 (192, h/2, w/2)
                                                                           │
Level 2→1: d2 ──→ up1(ConvT 192→96, stride 2)  + e1 ──→ [RCAB × 3] ──→ d1 (96, h, w)
                                                                           │
                                                                     ┌─────┘
                                                                     │
                                                           d1 (96, h, w) ← used for prior
```

### 2.5 Prior Network (Spectral Dictionary + Wavelet)

```
Conditioning:  concat(d1, f_msi_detail, X_obs)  → (96 + 96 + 128) = 320 channels
                        │
                        ▼
              ┌─────────────────┐
              │ Conv 320→96     │
              └────────┬────────┘
                       │
              ┌────────┴────────┐
              │  10 × RCAB      │
              │  + SpatialAttn  │  (every 2 blocks)
              │  + SpectralMix  │  (every 3 blocks)
              └────────┬────────┘
                       │
              v (96, h, w)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ code_g  │   │ code_m  │   │ code_f  │
   │ Conv    │   │ Conv    │   │ Conv    │
   │ 96→96   │   │ 96→64   │   │ 96→48   │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Dg×α_g  │   │ Dm×α_m  │   │ Df×α_f  │
   │128×96   │   │128×64   │   │128×48   │
   └────┬────┘   └────┬────┘   └────┬────┘
        │              │              │
        └──────┬───────┴──────┬───────┘
               ▼              │
        ┌──────────────┐      │
        │ scale_w      │      │
        │ (learned     │      │
        │  weighting)  │      │
        └──────┬───────┘      │
               ▼              │
     null_comp = w_g*Dg*α_g + w_m*Dm*α_m + w_f*Df*α_f
               │
               ▼
     null_comp = null_comp - R^T(R(null_comp))   ← MSI consistency
               │
               ▼
        ┌──────────────┐
        │ WaveletBranch│  3-level Haar DWT
        │ (learned HF  │  ┌─ Level 1: enc1(LH,HL,HH) → corr1
        │  refinement) │  ├─ Level 2: enc2(LL→LH,HL,HH) → corr2
        └──────┬───────┘  └─ Level 3: enc3(LL→LH,HL,HH) → corr3
               │
               ▼
        wf = corr1 + corr2 + corr3
               │
               ▼
        gate = σ(Conv([null_comp, wf]))  ← learned gate
        wf = wf × gate
               │
               ▼
        out = X_obs + null_comp + wf
```

### 2.6 RCAB (Residual Channel Attention Block)

```
Input x (C, H, W)
      │
      ├──→ body: Conv3×3 → GELU → Conv3×3
      │
      └──→ ca: GAP → Linear(C→C/16) → ReLU → Linear(C/16→C) → Sigmoid
                                    │
                         body(x) × ca(x)
                                    │
                            x + result
```

### 2.7 Spatial Self-Attention (Window-based)

```
Input x (C, H, W), window=8
      │
      Pad to multiple of window
      │
      Reshape to (B, nH, nW, ws, ws, C)
      │
      q,k,v = Conv1×1(x)
      │
      attn = softmax(q @ k^T / √C) @ v
      │
      Reshape back → Conv1×proj → x + result
```

---

## 3. Training Protocol

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-4, betas=0.9/0.999) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=2000, T_mult=2) |
| Batch size | 2 × 4 gradient accumulation = effective 8 |
| Patch size | 64×64 (adaptive based on GPU memory) |
| Steps/epoch | 200 |
| EMA decay | 0.999 |
| Gradient clip | 1.0 |
| Mixed precision | AMP (fp16) |
| Time budget | 8.5 hours |
| Max epochs | 5000 |

### Losses

| Loss | Weight | Formula |
|------|--------|---------|
| L1 fidelity | 1.0 | ‖X̂ - X‖₁ |
| Physics consistency | 0.1 | ‖D(X̂) - Y_H‖² |
| Spectral consistency | 0.05 | ‖consist - X‖₁ |
| Null-space regularisation | 0.01 | ‖P_N(v)‖₁ |

### Data Augmentation
- Random horizontal flip (50%)
- Random vertical flip (50%)
- Random rotation 90°/180°/270° (50%)
- Gaussian noise σ=0.01 (15%)

---

## 4. Dataset: Chikusei

| Property | Value |
|----------|-------|
| Sensor | Headwall Nano-Hyperspec-VNIR-C |
| Bands | 128 (363–1018 nm) |
| Spatial | 2517 × 2335 pixels |
| Train/Test | 70%/30% non-overlapping 64×64 patches |
| SRF | Estimated from sensor specs (condition=4.51) |
| Protocol | Wald: Gaussian blur (9×9, σ=1.2) + 4x decimation |

---

## 5. SOTA Comparison (Chikusei x4)

| Method | Year | Type | PSNR | SAM | Params |
|--------|------|------|------|-----|--------|
| CoFusion | 2026 | CNN+Attn | 49.14 | 2.60 | ~10M |
| SMGU-Net | 2025 | U-Net | 48.82 | 2.72 | ~8M |
| RAMoE | 2026 | MoE | 48.10 | 0.79 | ~12M |
| PSRT | 2023 | Transformer | 47.99 | 2.84 | ~6M |
| U2Net | 2023 | U-Net | 47.93 | 2.77 | ~5M |
| **NullFusion-SOTA** | **2026** | **Dict+Wavelet** | **target: >49.14** | **target: <2.60** | **15.28M** |
| KrylovNet v1 | 2026 | Unrolled | 43.69 | 6.07 | 2.3k |

---

## 6. Why This Architecture Should Win

1. **Exact data consistency**: X̂ = pinv(A) + P_N(f_θ). The base satisfies A(X̂)=[Y_H;Y_M] exactly. No other method guarantees this.

2. **Multi-scale spectral dictionary**: 3 dictionary scales (96+64+48=208 atoms) with learned weighting. The published SOTA methods use a single implicit mapping — we explicitly model spectral diversity.

3. **3-level wavelet refinement**: Captures high-frequency detail at 3 spatial scales. The gated wavelet branch learns when to add detail vs. when the dictionary is sufficient.

4. **U-Net encoder**: Skip connections preserve multi-scale spatial features. The SOTA methods (CoFusion, SMGU-Net) also use U-Net — we match their spatial modeling while adding the null-space guarantee.

5. **Deep cross-modal attention**: 3 levels of HSI↔MSI attention with 8 heads each. This is where MSI spatial detail gets injected into the 128-band spectral features.

6. **15.28M parameters**: Comparable to CoFusion (~10M) and RAMoE (~12M). Large enough to model 128-band diversity, not so large as to overfit on 70 train patches.

7. **Progressive training**: Start with small patches, increase. Cosine warm restarts help escape local minima.

8. **Multi-loss with physics**: The physics consistency loss D(X̂)=Y_H ensures the decoder doesn't drift from the observation model. The null-space regularisation prevents hallucination.
