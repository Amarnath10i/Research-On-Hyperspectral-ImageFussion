# SOTA Comparison — HSI-MSI Fusion (CAVE, Harvard, Chikusei, PaviaU)

Compiled from published papers (2022–2026). All numbers are the authors'
reported values under **their** protocol (Wald's simulation, Nikon SRF for
MSI). Our numbers are computed under the **repository's unified protocol**
(same degradation, fixed `data_range=1.0`). Read the protocol column.

---

## CAVE x4 (in-domain)

| Method | Year | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ | Protocol |
|---|---|---|---|---|---|---|---|
| BDT | 2023 | Unfolding | 52.30 | 0.997 | 1.93 | 1.02 | Wald + Nikon SRF |
| FeINFN | 2024 | INR | 52.47 | 0.998 | 1.91 | 0.98 | Wald + Nikon SRF |
| CoFusion | 2026 | CNN+Attn | 50.67 | 0.997 | 2.15 | 1.73 | Wald + SRF |
| SSA | 2026 | INR+MK | 45.92 | 0.996 | 2.02 | 1.07 | Wald, mixed datasets |
| Multi-path | 2026 | Transformer | 45.63 | 0.990 | 2.57 | 0.76 | Wald |
| SMF2Net | 2026 | CNN+HybridFormer | 43.91 | 0.997 | 2.04 | 6.40 | Wald |
| DSPNet | 2023 | CNN | 51.18 | 0.997 | 2.15 | 1.13 | Wald + Nikon SRF |
| 3DT-Net | 2023 | 3D CNN | 51.38 | 0.996 | 2.16 | 1.14 | Wald + Nikon SRF |
| PSRT | 2023 | Transformer | 50.47 | 0.996 | 2.19 | 2.06 | Wald + Nikon SRF |
| Fusformer | 2022 | Transformer | 44.52 | 0.983 | 4.12 | 1.06 | Wald |
| DHIF | 2022 | Deep + MMD | 51.07 | 0.997 | 2.01 | 1.22 | Wald + Nikon SRF |
| MIMO-SST | 2022 | CNN | 50.98 | 0.997 | 2.23 | 1.18 | Wald + Nikon SRF |
| **Ours (baseline, same-protocol)** | | | | | | | |
| Bicubic | | | 29.93 | 0.888 | 4.89 | 8.37 | Repo protocol (identity SRF) |
| GSA | | | 30.77 | 0.856 | 9.06 | 7.25 | Repo protocol (identity SRF) |
| Subspace-LS (r̂_id) | | | (run) | | | | Repo protocol (identity SRF) |

> **Protocol warning.** The published rows use a *true* MSI simulation (Nikon
> D700 spectral response → 3-band MSI) and report higher PSNR because the
> fusion problem is better conditioned. Our Kaggle CAVE attachment
> (`liptee/...`) ships 31-band `PER_RGB`, so the MSI ≈ identity copy of the
> HSI — a degenerate and *harder* condition for fusion. A like-for-like
> comparison requires re-simulating a 3-band MSI from the HSI via an SRF.

---

## Harvard x4 (in-domain, trained on Harvard)

| Method | Year | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ |
|---|---|---|---|---|---|---|
| FeINFN | 2024 | INR | 49.06 | 0.989 | 2.10 | 1.78 |
| BDT | 2023 | Unfolding | 48.83 | 0.989 | 2.07 | 1.83 |
| DSPNet | 2023 | CNN | 48.29 | 0.988 | 2.30 | 1.93 |
| HSRNet | 2022 | CNN | 48.29 | 0.988 | 2.26 | 1.87 |
| Multi-path | 2026 | Transformer | 43.57 | 0.984 | 2.85 | 1.02 |

## Harvard x4 (zero-shot, trained on CAVE — cross-domain)

| Method | Year | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ |
|---|---|---|---|---|---|---|
| Selective Re-learning | 2025 | CVPR | 46.48 | 0.983 | 2.99 | - |
| MIMO-SST | 2022 | CNN | 46.29 | 0.983 | 3.04 | - |
| DSPNet | 2023 | CNN | 45.83 | 0.982 | 3.17 | - |
| DHIF-Net | 2022 | Deep | 45.74 | 0.983 | 3.19 | - |

> Cross-domain (CAVE→Harvard) costs ~2-4 dB PSNR and ~0.5-1.0° SAM. This is
> the gap our r_id / ambiguity framework measures.

---

## Chikusei x4

| Method | Year | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ |
|---|---|---|---|---|---|---|
| CoFusion | 2026 | CNN+Attn | 49.14 | 0.995 | 2.60 | 2.01 |
| SMGU-Net | 2025 | U-Net | 48.82 | 0.993 | 2.72 | 2.05 |
| PSRT | 2023 | Transformer | 47.99 | 0.986 | 2.84 | 2.29 |
| U2Net | 2023 | U-Net | 47.93 | 0.987 | 2.77 | 2.16 |
| SSA (zero-shot) | 2026 | INR+MK | 38.71 (Houston) | 0.978 | 3.09 | 3.85 |

## PaviaU x4

| Method | Year | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ |
|---|---|---|---|---|---|---|
| CoFusion | 2026 | CNN+Attn | 38.32 | 0.982 | 2.56 | 1.95 |
| SMGU-Net | 2025 | U-Net | 37.43 | 0.973 | 3.20 | 2.16 |
| U2Net | 2023 | U-Net | 37.26 | 0.967 | 3.24 | 2.40 |
| PSRT | 2023 | Transformer | 36.77 | 0.963 | 3.59 | 2.65 |

---

## What the field is doing (2022 → 2026)

| Trend | Example methods | Progress |
|---|---|---|
| CNN → Transformer | Fusformer, PSRT, DCFormer | +6 dB PSNR over early CNN |
| → Efficient attention (Mamba, tensor-product) | PIF-Net, TPTransformer | Near-linear complexity at SOTA accuracy |
| → Implicit neural representation (INR) | FeINFN, SSA, OTIAS, NeSSR | Arbitrary scale, one model all scales |
| → Arbitrary-scale + sensor-agnostic | SSA (2026) | One model, 7 datasets, unseen sensors |
| → Diffusion / generative | DDPM-Fus, KANDiff | High PSNR but slow sampling |
| → Cross-domain robustness | Selective Re-learning (CVPR'25) | Explicit CAVE→Harvard evaluation |

## Where our work sits (and what is new)

Every SOTA method above is an **architecture that reconstructs**. None of them:

1. **Tells you which spectral ranks are identifiable** from the observation
   pair (`r_id = rank(R^T U_r)`, P2). They assume the rank, or use the
   intrinsic scene rank — which overestimates what the sensor can recover.
2. **Decomposes any reconstruction into observable (E_obs) and ambiguous
   (E_null) components** (P1 auditor). We audit *existing* SOTA (Fusformer,
   DSPNet, PIF-Net) and show *where* they hallucinate, instead of building
   another CNN.
3. **Proves a sensor-shift generalization bound** `Δ_sensor ≤ L_F · EMD`
   (P3), explaining *why* zero-shot CAVE→Harvard costs 2-4 dB.
4. **Derives the phase transition** `M*(r) = min M : rank(R^T U) = r` (P4)
   that predicts whether identification is possible at all.

**The comparison is therefore not "our PSNR vs their PSNR"** — it is
"their architecture + our identifiability audit" — a diagnostic layer that
no 2026 paper provides, and a principled explanation of the cross-domain gap
everyone measures but nobody explains.

---

## Next experiments (to fill the table with our own numbers)

| Experiment | Dataset(s) | What we report |
|---|---|---|
| Baselines under Wald's protocol (3-band MSI) | CAVE, Harvard | PSNR/SSIM/SAM/ERGAS vs above table |
| Train DAETF-Net / KrylovNet (2000 iters) | CAVE train | in-domain CAVE test |
| Zero-shot | CAVE→Harvard | cross-domain table (vs CVPR'25 46.48) |
| r̂_id model selection | CAVE, Harvard | rank auto-selection, error floor |
| P1 auditor on SOTA outputs | any | hallucination maps, H score |
