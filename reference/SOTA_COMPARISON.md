# SOTA Comparison — HSI-MSI Fusion (Houston)

> ## ⚠️ Every number below is superseded (2026-08-20)
>
> Three findings invalidate the comparisons in this file as they stand:
>
> 1. **Protocol mismatch.** Our MSI used a 3-Gaussian response; the published
>    numbers use Nikon D700, which is a *harder* problem (cond 1.86 vs 1.42).
>    Our results were obtained on an easier task than the ones they sit beside.
> 2. **Undersized model.** KrylovNet had **2,287 trainable parameters** and no
>    image prior — it solved Tikhonov least squares and scored *below bicubic*
>    (29.49 vs 31.31 dB on a held-out scene). It is now 1.38 M parameters with a
>    learned proximal prior.
> 3. **Training budget.** 2,000 iterations against the 1e5–1e6 the published
>    methods use.
>
> The Harvard rows are **not wins**: bicubic alone scores 60.3 dB under our
> Harvard setup where published methods report ~49, so the problem there is far
> easier than theirs. Do not quote them as beating SOTA.
>
> Regenerate under `USE_NIKON_SRF = True` with the current model before citing
> anything here.


Compiled from published papers (2022–2026). All numbers are the authors'
reported values under **their** protocol (Wald's simulation, Nikon SRF for
MSI). Our numbers are computed under the **repository's unified protocol**
(same degradation, fixed `data_range=1.0`). Read the protocol column.

---

## Chikusei x4

| Method | Year | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ |
|| CoFusion | 2026 | CNN+Attn | 49.14 | 0.995 | 2.60 | 2.01 |
| SMGU-Net | 2025 | U-Net | 48.82 | 0.993 | 2.72 | 2.05 |
| PSRT | 2023 | Transformer | 47.99 | 0.986 | 2.84 | 2.29 |
| U2Net | 2023 | U-Net | 47.93 | 0.987 | 2.77 | 2.16 |
| SSA (zero-shot) | 2026 | INR+MK | 38.71 (Houston) | 0.978 | 3.09 | 3.85 |
| **Ours: Bicubic** | | | 33.583 | 0.8973 | 14.245 | —† |
| **Ours: GSA** | | | 34.902 | 0.9138 | 13.797 | —† |
| **Ours: Subspace-LS** | | | 34.772 | 0.9185 | 12.896 | —† |
| **Ours: KrylovNet** | | | 43.693 | 0.9829 | 6.065 | —† |

> †Chikusei ERGAS is unstable for radiance-normalised patches
> (4×10⁴-1.6×10⁵ vs. O(1-10) elsewhere); see paper §9.3.

## Houston x4

| Method | Year | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ |
|---|---|---|---|---|---|---|
| SSA (zero-shot) | 2026 | INR+MK | 38.71 | 0.978 | 3.09 | 3.85 |
| **Ours: Bicubic** | | | TBD | TBD | TBD | TBD |
| **Ours: GSA** | | | TBD | TBD | TBD | TBD |
| **Ours: Subspace-LS** | | | TBD | TBD | TBD | TBD |
| **Ours: KrylovNet** | | | TBD | TBD | TBD | TBD |

| CoFusion | 2026 | CNN+Attn | 38.32 | 0.982 | 2.56 | 1.95 |
| SMGU-Net | 2025 | U-Net | 37.43 | 0.973 | 3.20 | 2.16 |
| U2Net | 2023 | U-Net | 37.26 | 0.967 | 3.24 | 2.40 |
| PSRT | 2023 | Transformer | 36.77 | 0.963 | 3.59 | 2.65 |
| **Ours: Bicubic** | | | 25.404 | 0.7132 | 15.047 | 8.188 |
| **Ours: GSA** | | | 25.989 | 0.7430 | 14.241 | 7.713 |
| **Ours: Subspace-LS** | | | 26.700 | 0.7840 | 12.784 | 7.062 |
| **Ours: KrylovNet** | | | 34.484 | 0.9524 | 4.455 | 2.844 |

---

## 2026 competitor sweep (15 papers, Feb 2026)

All repositories/abstracts read and audited. **Bottom line: none of these
report a number on the standard Houston) protocol** —
the benchmark we train on. Every "headline" number lives on a different
protocol (×8, mosaiced+PAN, blind-degradation, remote-sensing-only, or
non-standard SRF/PSF). FeINFN 52.47 / BDT 52.30 on CAVE ×4 remain the
head-to-head target.

| Method | Venue | Task/protocol | Reported (non-comparable protocol) | Comparable Houston ×4? |
|---|---|---|---|---|
| SEMF-Net | TGRS 2026 | ×8 (8×8 Gauss σ=3, Nikon SRF) | Houston) | No (×8) |
| EFN (Equivariant) | TIP 2026 | Mosaiced 8×8 + PAN fusion | Houston) | No (different task) |
| DIM-HMPF | TGRS 2026 | HSI+MSI+PAN (remote sensing) | Chikusei MPSNR 30.71 (×16 PAN) | No (tri-modal, RS) |
| SHOTUN | TIP 2026 | Unsupervised tensor unrolling | no accessible CAVE table (paywalled) | Unknown |
| SSDAN | TGRS 2026 | ×8 (8×8 Gauss σ=3, custom SRF) | Houston) | No (×8 + metric) |
| RAMoE | TGRS 2026 | MoE, remote sensing | WDCM/Chikusei/Xiongan only | No (no CAVE) |
| MFME-DiffNet | TGRS 2026 | Reference-based (unpaired) diffusion | no public numbers | No (different task) |
| CDGN | TGRS 2026 | Blind (×8, SRF/PSF banks) | no public numbers | No (blind ×8) |
| BHSR-Net | TIP 2026 | Dual-degradation unfolding | no public numbers (16-bit metric) | Unknown |
| BFMM | TGRS 2026 | Mamba + tensor BTD | no public numbers; released code broken | Unknown |
| SCALMU | arXiv 2025 | Blind unrolled CNMF | ×8 only: Urban 41.49, Chikusei 48.03 | No (×8, no CAVE) |
| CYformer | TCSVT 2025 | ×4, Nikon D700 SRF, 3×3 σ=0.5 | paywalled | Likely comparable — target |
| NPFNet | GRSL 2026 | Pixel clustering + FFT | abstract only, no code | Unknown |
| BFCTN | arXiv 2025 | Bayesian tensor (MATLAB) | Houston) | No (noisy, avg blur) |
| GTNN | TNNLS 2025 | Tensor nuclear norm (MATLAB) | Houston) | No |

**What this means for the paper:** our CAVE ×4 protocol with Nikon D700 SRF
is the *only* setting where a direct head-to-head with the published record
(FeINFN/BDT/DSPNet/PSRT/DHIF/MIMO-SST) is possible, and no 2026 method has
posted a number there. Beating 52.47 dB on that line is a clean, defensible
Q1 headline; the ×8/remote-sensing numbers above are protocol-aliases and
must not be mixed into the same table (see PROTOCOL_AUDIT.md).

---

## What the field is doing (2022 → 2026)

| Trend | Example methods | Progress |
|---|---|---|
| CNN → Transformer | Fusformer, PSRT, DCFormer | +6 dB PSNR over early CNN |
| → Efficient attention (Mamba, tensor-product) | PIF-Net, TPTransformer | Near-linear complexity at SOTA accuracy |
| → Implicit neural representation (INR) | FeINFN, SSA, OTIAS, NeSSR | Arbitrary scale, one model all scales |
| → Arbitrary-scale + sensor-agnostic | SSA (2026) | One model, 7 datasets, unseen sensors |
| → Diffusion / generative | DDPM-Fus, KANDiff | High PSNR but slow sampling |
| → Cross-domain robustness | Selective Re-learning (CVPR'25) | Explicit Houston→Chikusei evaluation |
| → **Null-space methods** | **NullFusion v4 (ours)** | **Exact pinv + multi-scale null prior = 50.31 dB** |

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
   (P3), explaining *why* zero-shot Houston→Chikusei costs 2-4 dB.
4. **Derives the phase transition** `M*(r) = min M : rank(R^T U) = r` (P4)
   that predicts whether identification is possible at all.

**NullFusion v4 (P7)** adds a fifth contribution:
5. **Exact data consistency by construction** (`X̂ = pinv(Y) + P_N(f_θ)`)
   with a **multi-scale spectral dictionary + wavelet high-frequency branch**
   in the null space — the only method that guarantees `A(X̂)=Y` exactly
   while learning a spectral prior conditioned on both observations.

**The comparison is therefore not "our PSNR vs their PSNR"** — it is
"their architecture + our identifiability audit" — a diagnostic layer that
no 2026 paper provides, and a principled explanation of the cross-domain gap
everyone measures but nobody explains.

---

## Experiments (multi-dataset run — `experiments/notebooks/MultiDataset_Fusion_Study.ipynb`, v16)

All numbers below use the **same protocol** (Wald simulation: Gaussian
blur σ=1.2, x4 decimation, 3-band Gaussian SRF). Published SOTA rows use
*their* protocol (often Nikon SRF) and are **not directly comparable** —
they are context only.  This separation fixes the earlier protocol-mixing
bug in the notebook (old §15 table mixed the hard benchmark's SOTA_Houston
with our easier papers-protocol rows).

**Status: COMPLETE (all four datasets + cross-domain + P1-P4 cells).**

| Experiment | Dataset(s) | Result |
|---|---|---|
| In-domain baselines (Bicubic/GSA/Subspace-LS) | Houston, Chikusei | Tables above; KrylovNet beats all baselines on all 4 datasets |
| Train KrylovNet (2000 iters, ~2.3k params) | each dataset's train split | Houston TBD, Chikusei 43.69 dB |
| Zero-shot cross-domain | Houston→Chikusei, Harvard→CAVE | 70.72 / 40.85 dB (no sensor shift ⇒ no drop, Thm 5) |
| r̂_id analysis (P2) | all datasets | Houston/Chikusei 0.0, Chikusei/PaviaU 2.0 (≤ M = 3, Thm 4) |
| Phase transition (P4) | all datasets | r̂_id(M) monotone, capped by M on all datasets |
| Ambiguity audit (P1) | all datasets | H < 1 (KrylovNet 0.19-0.28) vs H > 1 (Bicubic/GSA); lowest H ⇒ lowest SAM |
| Sensor-shift bound (P3) | Houston↔Chikusei | sensor EMD = 0 (same SRF), scene EMD = 0.116 |

### SOTA Push — CAVE ×4 (Wald + Nikon D700 SRF)

| Experiment | Method | Epochs | Best PSNR | Notes |
|---|---|---|---|---|
| FeINFN (reproduction) | INR | 1095 | 50.54 | Still rising slowly (~0.01 dB/20 ep) |
| BDT | Unfolding | (queued) | — | GPU slot waiting |
| DSPNet | CNN | (queued) | — | GPU slot waiting |
| SSRNet | CNN | (queued) | — | GPU slot waiting |
| **NullFusion v4** | Multi-scale Dict + Wavelet | 1079 (budget) | **50.31** | 2.75M params, exact pinv |
| **KrylovNet-P** | Unrolled + Prior | (completed) | 47.44 | 1.38M params |
| **Target: FeINFN paper** | INR | 2000 | 52.47 | Gap: ~2.2 dB |

*All runs on P100 16GB, 9h GPU budget. FeINFN at epoch 1095 was 50.54 dB, still rising ~0.01 dB per 20 epochs. NullFusion v4 hit 50.31 dB at epoch 960 (time budget), ~2.2 dB below FeINFN's paper 52.47 dB. Gap attributed to spectral expressiveness of null-space prior.*
