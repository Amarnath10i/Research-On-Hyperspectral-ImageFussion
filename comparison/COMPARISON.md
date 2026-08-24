# HSI–MSI Image Fusion: Per-Dataset Method Comparison (2024–2026)

**Only IEEE Xplore-indexed venues (TGRS, TIP, TCSVT, CVPR) are included**;
arXiv-only and MDPI (Remote Sensing) entries are omitted. Metadata and SOTA
for the 11 papers marked **[PDF]** were read directly from the source PDFs
(`C:\Users\sande\Downloads\Projects\Projects`). Where a result table was a
rasterized image, values were recovered via OCR and are **approximate**
(marked `≈OCR`). Entries whose tables could not be OCR'd are marked
`see paper`.

A broader 23-paper survey (dataset-usage matrix) is in
[`../review/DATASETS.md`](../review/DATASETS.md).

## Legend
- **Type**: Sup = supervised, Unsup = unsupervised/blind, Zero = zero-shot, MB = model-based/unrolled.
- **Notes**: `exact` = quoted from paper; `≈OCR` = approximate (rasterized table OCR'd); `Δ` = only improvement over a baseline reported; `QNR` = real data, no-reference metric (no PSNR/SAM); `see paper` = not extracted; `n/r` = not reported.
- External baselines (BDT, FeINFN, CoFusion, SSA, Multi-path, SMF2Net, DSPNet, 3DT-Net, PSRT, Fusformer, DHIF) are as provided; full titles/DOIs are pending verification.

## Paper index

| Key | Paper title | Venue / Year | Type | DOI |
|---|---|---|---|---|
| fang2026detail [PDF] | A Detail Injection-Based Fusion Framework for Hyperspectral, Multispectral, and Panchromatic Remote Sensing Images | TGRS 2026 | Sup (unfolding) | 10.1109/TGRS.2026.3683056 |
| li2026bfmm [PDF] | Block Term Decomposition-Guided Frequency Mamba Modulation for Hyperspectral Image Fusion | TGRS 2026 | Sup | 10.1109/TGRS.2026.3699818 |
| dou2026bhsrnet [PDF] | Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model | TIP 2026 | Sup (unfolding) | 10.1109/TIP.2026.3714832 |
| liu2026causal [PDF] | Causal Degradation-Guided Network With Spatial-Frequency Attention for Blind Hyperspectral Image Fusion | TGRS 2026 | Sup blind | 10.1109/TGRS.2026.3703367 |
| he2026diffusion [PDF] | Diffusion-Driven Mutual Enhancement of Matching and Fusion for Reference-Based Hyperspectral Image Super-Resolution | TGRS 2026 | Sup (diffusion) | 10.1109/TGRS.2026.3656069 |
| wang2026equivariant [PDF] | Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion | TIP 2026 | Unsup | 10.1109/TIP.2026.3657219 |
| xiao2026region [PDF] | Region-Aware MoE Network for Hyperspectral and Multispectral Image Fusion | TGRS 2026 | Sup | 10.1109/TGRS.2026.3680287 |
| song2026s2 [PDF] | S²-Differential Feature Awareness Network for Hyperspectral Image Fusion | TGRS 2026 | Sup | 10.1109/TGRS.2026.3671284 |
| xu2026scalmu [PDF] | SCALMU: Synthetically Trained Coupling of Adaptive Learned Multiplicative Updates for Hyperspectral-Multispectral Fusion | TGRS 2026 | MB unrolled (blind) | 10.1109/TGRS.2026.3712501 |
| wang2026shotun [PDF] | Self-Expressive High-Order Tensor Unrolling Network for Unsupervised Hyperspectral and Multispectral Image Fusion | TIP 2026 | Unsup/blind | 10.1109/TIP.2026.3695389 |
| liu2026semfnet [PDF] | SEMF-Net: A Spatial-Spectral Edge-Enhancement-Based Multistage Fusion Network for Hyperspectral and Multispectral Image Fusion | TGRS 2026 | Sup | 10.1109/TGRS.2026.3653545 |
| xiao2025dmzs | Hyperspectral Pansharpening via Diffusion Models with Iteratively Zero-Shot Guidance | CVPR 2025 | Zero (diffusion) | 10.1109/CVPR52734.2025.01182 |
| peng2024fusionmamba | FusionMamba: Efficient Remote Sensing Image Fusion With State Space Model | TGRS 2024 | Sup | 10.1109/TGRS.2024.3496073 |
| qu2025irarf | IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images | TIP 2025 | Sup | 10.1109/TIP.2025.3551531 |
| chen2025cyformer | Cyclic Cross-Modality Interaction for Hyperspectral and Multispectral Image Fusion | TCSVT 2025 | Sup | 10.1109/TCSVT.2024.3461829 |
| guarino2025rhopnn | Zero-Shot Hyperspectral Pansharpening Using Hysteresis-Based Tuning for Spectral Quality Control | TGRS 2025 | Zero | 10.1109/TGRS.2025.3583877 |
| fang2024mimosst | MIMO-SST: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion | TGRS 2024 | Sup | 10.1109/TGRS.2024.3361553 |

---

## CAVE

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Block Term Decomposition-Guided Frequency Mamba Modulation | li2026bfmm | 2026 | Sup | ×4 | 53.41 | 1.66 | — | — | Wald | ≈OCR |
| Block Term Decomposition-Guided Frequency Mamba Modulation | li2026bfmm | 2026 | Sup | ×8 | 51.20 | 1.98 | — | — | Wald | ≈OCR |
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s8 | 46.69 | 2.98 | — | — | Wald | ≈OCR |
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s32 | 41.33 | 5.10 | — | — | Wald | ≈OCR |
| S²-Differential Feature Awareness Network | song2026s2 | 2026 | Sup | ×8 | 49.05 | 2.22 | — | — | Wald | ≈OCR |
| Equivariant HR-HSI via Mosaiced and PAN Fusion | wang2026equivariant | 2026 | Unsup | ×4 (mosaiced+PAN) | 40.55 | 5.31 | — | — | mosaiced+PAN | exact |
| Cyclic Cross-Modality Interaction | chen2025cyformer | 2025 | Sup | ×4 | — | — | — | — | Wald | see paper |
| Cyclic Cross-Modality Interaction | chen2025cyformer | 2025 | Sup | ×8 | — | — | — | — | Wald | see paper |
| MIMO-SST Spatial-Spectral Transformer | fang2024mimosst | 2024 | Sup | ×8 | 47.30 | — | — | — | Wald | exact (PSNR only) |
| FusionMamba | peng2024fusionmamba | 2024 | Sup | HISR | — | — | — | — | HISR | see paper |
| BDT (external; verify) | BDT | 2023 | Unfolding | ×4 | 52.30 | 1.93 | 0.997 | 1.02 | Wald + Nikon SRF | provided |
| FeINFN (project; verify) | FeINFN | 2024 | INR | ×4 | 52.47 | 1.91 | 0.998 | 0.98 | Wald + Nikon SRF | provided |
| CoFusion (external; verify) | CoFusion | 2026 | CNN+Attn | ×4 | 50.67 | 2.15 | 0.997 | 1.73 | Wald + SRF | provided |
| SSA (external; verify) | SSA | 2026 | INR+MK | ×4 | 45.92 | 2.02 | 0.996 | 1.07 | Wald, mixed | provided |
| Multi-path (external; verify) | Multi-path | 2026 | Transformer | ×4 | 45.63 | 2.57 | 0.990 | 0.76 | Wald | provided |
| SMF2Net (external; verify) | SMF2Net | 2026 | CNN+HybridFormer | ×4 | 43.91 | 2.04 | 0.997 | 6.40 | Wald | provided |
| DSPNet (external; verify) | DSPNet | 2023 | CNN | ×4 | 51.18 | 2.15 | 0.997 | 1.13 | Wald + Nikon SRF | provided |
| 3DT-Net (external; verify) | 3DT-Net | 2023 | 3D CNN | ×4 | 51.38 | 2.16 | 0.996 | 1.14 | Wald + Nikon SRF | provided |
| PSRT (external; verify) | PSRT | 2023 | Transformer | ×4 | 50.47 | 2.19 | 0.996 | 2.06 | Wald + Nikon SRF | provided |
| Fusformer (external; verify) | Fusformer | 2022 | Transformer | ×4 | 44.52 | 4.12 | 0.983 | 1.06 | Wald | provided |
| DHIF (external; verify) | DHIF | 2022 | Deep + MMD | ×4 | 51.07 | 2.01 | 0.997 | 1.22 | Wald + Nikon SRF | provided |

## Harvard

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Block Term Decomposition-Guided Frequency Mamba Modulation | li2026bfmm | 2026 | Sup | ×4 | 48.61 | 2.52 | — | — | Wald | ≈OCR |
| Block Term Decomposition-Guided Frequency Mamba Modulation | li2026bfmm | 2026 | Sup | ×8 | 47.69 | 2.72 | — | — | Wald | ≈OCR |
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s8 | 48.08 | 2.29 | — | — | Wald | ≈OCR |
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s32 | 46.34 | 2.69 | — | — | Wald | ≈OCR |
| S²-Differential Feature Awareness Network | song2026s2 | 2026 | Sup | ×8 | 47.42 | 2.85 | — | — | Wald | ≈OCR |
| Cyclic Cross-Modality Interaction | chen2025cyformer | 2025 | Sup | ×8 | — | — | — | — | Wald | see paper |
| SEMF-Net | liu2026semfnet | 2026 | Sup | ×32 | +0.21 dB Δ | — | — | — | Wald | Δ vs subopt |
| MIMO-SST Spatial-Spectral Transformer | fang2024mimosst | 2024 | Sup | ×8 | — | — | — | — | Wald | see paper |

## Chikusei

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| A Detail Injection-Based Fusion Framework (HSI-MSI-PAN) | fang2026detail | 2026 | Sup | ×16 (HSI→PAN) | 30.71 | 7.52 | — | — | HSI-MSI-PAN | ≈OCR |
| Block Term Decomposition-Guided Frequency Mamba Modulation | li2026bfmm | 2026 | Sup | ×4 | 44.61 | 2.50 | — | — | Wald | ≈OCR |
| Block Term Decomposition-Guided Frequency Mamba Modulation | li2026bfmm | 2026 | Sup | ×8 | 41.37 | 2.85 | — | — | Wald | ≈OCR |
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s8 | 40.76 | 2.24 | — | — | Wald | ≈OCR |
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s32 | 37.42 | 2.92 | — | — | Wald | ≈OCR |
| Region-Aware MoE Network | xiao2026region | 2026 | Sup | ×4 | 48.10 | 0.79 | — | — | Wald | ≈OCR |
| Region-Aware MoE Network | xiao2026region | 2026 | Sup | ×8 | 47.21 | 0.88 | — | — | Wald | ≈OCR |
| SCALMU | xu2026scalmu | 2026 | MB (blind) | ×8 | 48.03 | 2.65 | — | — | Wald | ≈OCR |
| Cyclic Cross-Modality Interaction | chen2025cyformer | 2025 | Sup | ×8 | — | — | — | — | Wald | see paper |
| MIMO-SST Spatial-Spectral Transformer | fang2024mimosst | 2024 | Sup | ×8 | — | — | — | — | Wald | see paper |
| IR&ArF (unregistered) | qu2025irarf | 2025 | Sup | arbitrary | — | — | — | — | arbitrary | see paper |
| Hyperspectral Pansharpening via Diffusion (Zero-Shot) | xiao2025dmzs | 2025 | Zero | pansharp | — | — | — | — | pansharp | see paper |

## PaviaU

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| SCALMU | xu2026scalmu | 2026 | MB (blind) | ×8 | 40.15 | 2.43 | — | — | Wald | ≈OCR |
| SEMF-Net | liu2026semfnet | 2026 | Sup | ×32 | RMSE −2.23 Δ | — | — | — | Wald | Δ vs subopt |
| MIMO-SST Spatial-Spectral Transformer | fang2024mimosst | 2024 | Sup | ×8 | — | — | — | — | Wald | see paper |

## Houston

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| A Detail Injection-Based Fusion Framework (HSI-MSI-PAN) | fang2026detail | 2026 | Sup | ×16 (HSI→PAN) | 29.68 | 8.13 | — | — | HSI-MSI-PAN | ≈OCR |
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s8 | 54.92 | 0.75 | — | — | Wald | ≈OCR |
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s32 | 53.36 | 0.80 | — | — | Wald | ≈OCR |
| Diffusion-Driven Mutual Enhancement (Reference-Based) | he2026diffusion | 2026 | Sup (diff.) | ×4 | — | — | — | — | Wald | see paper |

## WDCM

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Region-Aware MoE Network | xiao2026region | 2026 | Sup | ×4 | 53.93 | 0.72 | — | — | Wald | ≈OCR |
| Region-Aware MoE Network | xiao2026region | 2026 | Sup | ×8 | 52.46 | 0.84 | — | — | Wald | ≈OCR |

## Xiongan

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| A Detail Injection-Based Fusion Framework (HSI-MSI-PAN) | fang2026detail | 2026 | Sup | ×16 (HSI→PAN) | 31.68 | 4.46 | — | — | HSI-MSI-PAN | ≈OCR |
| Region-Aware MoE Network | xiao2026region | 2026 | Sup | ×4 | 52.60 | 0.38 | — | — | Wald | ≈OCR |
| Region-Aware MoE Network | xiao2026region | 2026 | Sup | ×8 | 50.36 | 0.49 | — | — | Wald | ≈OCR |

## KAIST

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| S²-Differential Feature Awareness Network | song2026s2 | 2026 | Sup | ×8 | 45.88 | 2.47 | — | — | Wald | ≈OCR |

## ICVL

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Causal Degradation-Guided Network (Blind) | liu2026causal | 2026 | Sup blind | ×8 | — | — | — | — | Wald | rasterized → see paper |
| Equivariant HR-HSI via Mosaiced and PAN Fusion | wang2026equivariant | 2026 | Unsup | ×4 (mosaiced+PAN) | 46.43 | 1.08 | — | — | mosaiced+PAN | exact |
| MIMO-SST Spatial-Spectral Transformer | fang2024mimosst | 2024 | Sup | ×8 | — | — | — | — | Wald | see paper |

## Urban

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| SCALMU | xu2026scalmu | 2026 | MB (blind) | ×8 | 41.49 | 2.17 | — | — | Wald | ≈OCR |

## PRISMA-Paris

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| SCALMU | xu2026scalmu | 2026 | MB (blind) | ×8 | 43.24 | 4.42 | — | — | Wald | ≈OCR |

## LN01 (real)

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Blur-Resistant HSI SR via Dual-Degradation Fusion Model | dou2026bhsrnet | 2026 | Sup | s8 | 42.17 | 3.20 | — | — | real (no GT) | ≈OCR |

## Ziyuan-1 (real)

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| SCALMU | xu2026scalmu | 2026 | MB (blind) | ×3 | QNR 0.958 | — | — | — | real (no GT) | QNR |

## WV-3

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Block Term Decomposition-Guided Frequency Mamba Modulation | li2026bfmm | 2026 | Sup | pansharp | 37.95 | 2.76 | — | — | pansharp | ≈OCR |
| SEMF-Net | liu2026semfnet | 2026 | Sup | real | QNR best | — | — | — | real (no GT) | Δ vs subopt |

## Pavia

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| MIMO-SST Spatial-Spectral Transformer | fang2024mimosst | 2024 | Sup | ×8 | — | — | — | — | Wald | see paper |
| IR&ArF (unregistered) | qu2025irarf | 2025 | Sup | arbitrary | — | — | — | — | arbitrary | see paper |
| Hyperspectral Pansharpening via Diffusion (Zero-Shot) | xiao2025dmzs | 2025 | Zero | pansharp | — | — | — | — | pansharp | see paper |
| Diffusion-Driven Mutual Enhancement (Reference-Based) | he2026diffusion | 2026 | Sup (diff.) | ×4 | — | — | — | — | Wald | see paper |
| FusionMamba | peng2024fusionmamba | 2024 | Sup | HyperPanCollection | — | — | — | — | pansharp | see paper |

## WDC

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Hyperspectral Pansharpening via Diffusion (Zero-Shot) | xiao2025dmzs | 2025 | Zero | pansharp | — | — | — | — | pansharp | see paper |
| IR&ArF (unregistered) | qu2025irarf | 2025 | Sup | arbitrary | — | — | — | — | arbitrary | see paper |
| FusionMamba | peng2024fusionmamba | 2024 | Sup | PanCollection | — | — | — | — | pansharp | see paper |

## FR1 (PRISMA)

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Hyperspectral Pansharpening via Diffusion (Zero-Shot) | xiao2025dmzs | 2025 | Zero | pansharp | — | — | — | — | pansharp | see paper |

## Houston 2018 (real)

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Self-Expressive High-Order Tensor Unrolling | wang2026shotun | 2026 | Unsup/blind | real | QNR 0.910 | — | — | — | real (no GT) | QNR |

## Real-world mosaiced (Equivariant)

| Paper title | Method | Year | Type | Scale | PSNR (dB) | SAM (°) | SSIM | ERGAS | Protocol | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Equivariant HR-HSI via Mosaiced and PAN Fusion | wang2026equivariant | 2026 | Unsup | ×4 | QNR 0.868 | — | — | — | real (no GT) | QNR |

---

## Corrections vs. the provided "verified" BibTeX
The supplied BibTeX claimed several DOIs/authors as verified; the source PDFs
contradict the following (now fixed in `hsim_fusion_comparison.bib`):

| Key | Pasted (WRONG) | Corrected from PDF |
|---|---|---|
| wang2026equivariant | DOI 10.1109/TIP.2026.3537982; datasets CAVE/Chikusei | DOI **10.1109/TIP.2026.3657219**; datasets **CAVE/ICVL** |
| dou2026bhsrnet | TGRS, DOI 10.1109/TGRS.2026.3541234, "Dou, Yujie et al." | **TIP 2026**, DOI **10.1109/TIP.2026.3714832**, authors **Mai Xu et al.** |
| li2026bfmm | DOI 10.1109/TGRS.2026.3567891, "Li, Rui and others" | DOI **10.1109/TGRS.2026.3699818**, authors **Yan Li et al.** |
| liu2026semfnet | TIP, DOI 10.1109/TIP.2026.3567890, "Xing, Yuan et al." | **TGRS 2026 (5501916)**, DOI **10.1109/TGRS.2026.3653545**, authors **Siyuan Liu et al.** |
| chen2025cyformer | DOI 10.1109/TCSVT.2024.3456789 | DOI **10.1109/TCSVT.2024.3461829** |
| wang2026shotun | datasets CAVE/Harvard/Chikusei | datasets are **PaviaC/KSC/SanDiego/Houston** |

## Dataset conventions
- **Simulated (Wald)**: CAVE (×8/×16/×32), Harvard (×8), Chikusei (×8), Pavia/ICVL.
- **Real / no-GT**: PaviaU, Chikusei, Houston2018, WV-3, PRISMA, LN01, Ziyuan-1 — no-reference metrics (QNR, Dλ, Ds).
- **Pansharpening / mosaiced-PAN** (DM-ZS, rho-PNN, Equivariant) use PRISMA/WV-3/PanCollection.
