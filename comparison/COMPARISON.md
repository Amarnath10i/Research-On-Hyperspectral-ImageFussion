# HSI–MSI Image Fusion: Method Comparison (2024–2026)

Consolidated comparison of recent hyperspectral–multispectral (HSI–MSI) and
related pansharpening / mosaiced‑PAN fusion methods. **Only IEEE Xplore‑indexed
venues (TGRS, TIP, TCSVT, CVPR) are included**; arXiv‑only and MDPI (Remote
Sensing) entries are omitted.

Metadata and SOTA for the 11 papers marked **[PDF]** were read directly from
the source PDFs (`C:\Users\sande\Downloads\Projects\Projects`). Where a result
table was a rasterized image, values were recovered via OCR and are
**approximate** (marked `≈ OCR`). Entries whose tables could not be OCR'd are
marked `table rasterized → see paper`.

A broader 23‑paper survey (dataset‑usage matrix for CAVE/Harvard/Chikusei/PaviaU)
is in [`../review/DATASETS.md`](../review/DATASETS.md).

## Legend
- **Type**: Sup = supervised, Unsup = unsupervised/blind, Zero = zero‑shot, MB = model‑based/unrolled.
- **Scale**: ×4/×8/×16/×32 spatial upscaling; `s=8`/`s=32` denote the same; `pansharp` = pansharpening.
- **Indicator**: `exact` = quoted verbatim from paper; `≈ OCR` = approximate (rasterized table OCR'd); `Δ` = only improvement over a baseline reported (relative); `QNR` = real data, no‑reference metric (no PSNR/SAM); `—` = not evaluated / see paper.
- A broader 23‑paper survey (dataset‑usage matrix for CAVE/Harvard/Chikusei/PaviaU) is in `../review/DATASETS.md`.

## 1. Paper index

| Key | Paper title | Venue / Year | Type | Backbone | DOI |
|---|---|---|---|---|---|
| **fang2026detail** [PDF] | A Detail Injection-Based Fusion Framework for Hyperspectral, Multispectral, and Panchromatic Remote Sensing Images | TGRS 2026 | Sup (unfolding) | Detail-injection proximal-gradient unfolding | 10.1109/TGRS.2026.3683056 |
| **li2026bfmm** [PDF] | Block Term Decomposition-Guided Frequency Mamba Modulation for Hyperspectral Image Fusion | TGRS 2026 | Sup | BTD low-rank tensor + Frequency Mamba | 10.1109/TGRS.2026.3699818 |
| **dou2026bhsrnet** [PDF] | Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model | TIP 2026 | Sup (unfolding) | Dual-degradation second-order semismooth Newton | 10.1109/TIP.2026.3714832 |
| **liu2026causal** [PDF] | Causal Degradation-Guided Network With Spatial-Frequency Attention for Blind Hyperspectral Image Fusion | TGRS 2026 | Sup blind | Causal multi-degradation + Spatial-Frequency Attention | 10.1109/TGRS.2026.3703367 |
| **he2026diffusion** [PDF] | Diffusion-Driven Mutual Enhancement of Matching and Fusion for Reference-Based Hyperspectral Image Super-Resolution | TGRS 2026 | Sup (diffusion) | Diffusion reverse process + iterative matching-fusion | 10.1109/TGRS.2026.3656069 |
| **wang2026equivariant** [PDF] | Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion | TIP 2026 | Unsup | Equivariant imaging (learnable degradation/SRF) | 10.1109/TIP.2026.3657219 |
| **xiao2026region** [PDF] | Region-Aware MoE Network for Hyperspectral and Multispectral Image Fusion | TGRS 2026 | Sup | Region-Aware Mixture-of-Experts | 10.1109/TGRS.2026.3680287 |
| **song2026s2** [PDF] | S²-Differential Feature Awareness Network for Hyperspectral Image Fusion | TGRS 2026 | Sup | S²-Differential Feature Awareness (CNN) | 10.1109/TGRS.2026.3671284 |
| **xu2026scalmu** [PDF] | SCALMU: Synthetically Trained Coupling of Adaptive Learned Multiplicative Updates for Hyperspectral-Multispectral Fusion | TGRS 2026 | MB unrolled (blind) | Unrolled CNMF multiplicative updates (dead-leaves trained) | 10.1109/TGRS.2026.3712501 |
| **wang2026shotun** [PDF] | Self-Expressive High-Order Tensor Unrolling Network for Unsupervised Hyperspectral and Multispectral Image Fusion | TIP 2026 | Unsup/blind | Self-Expressive High-Order Tensor Unrolling | 10.1109/TIP.2026.3695389 |
| **liu2026semfnet** [PDF] | SEMF-Net: A Spatial-Spectral Edge-Enhancement-Based Multistage Fusion Network for Hyperspectral and Multispectral Image Fusion | TGRS 2026 | Sup | Multistage edge-enhancement + wavelet HF loss | 10.1109/TGRS.2026.3653545 |
| xiao2025dmzs | Hyperspectral Pansharpening via Diffusion Models with Iteratively Zero-Shot Guidance | CVPR 2025 | Zero (diffusion) | Diffusion + iteratively zero-shot guidance | 10.1109/CVPR52734.2025.01182 |
| peng2024fusionmamba | FusionMamba: Efficient Remote Sensing Image Fusion With State Space Model | TGRS 2024 | Sup | Mamba / State Space Model | 10.1109/TGRS.2024.3496073 |
| qu2025irarf | IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images | TIP 2025 | Sup | Interpretable arbitrary-resolution + unregistered | 10.1109/TIP.2025.3551531 |
| chen2025cyformer | Cyclic Cross-Modality Interaction for Hyperspectral and Multispectral Image Fusion | TCSVT 2025 | Sup | Cyclic Cross-Modality Transformer | 10.1109/TCSVT.2024.3461829 |
| guarino2025rhopnn | Zero-Shot Hyperspectral Pansharpening Using Hysteresis-Based Tuning for Spectral Quality Control | TGRS 2025 | Zero | rho-PNN (hysteresis-based tuning) | 10.1109/TGRS.2025.3583877 |
| fang2024mimosst | MIMO-SST: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion | TGRS 2024 | Sup | Spatial-Spectral Transformer (MIMO) | 10.1109/TGRS.2024.3361553 |

## 2. Per-dataset SOTA (PSNR / SAM)

Each row is one (method, dataset, scale) evaluation. PSNR in dB, SAM in degrees.

| Method | Dataset | Scale | PSNR (dB) | SAM (°) | Indicator |
|---|---|---|---|---|---|
| **fang2026detail** | Chikusei | ×16 (HSI→PAN) | 30.71 | 7.52 | ≈ OCR |
| **fang2026detail** | Houston | ×16 | 29.68 | 8.13 | ≈ OCR |
| **fang2026detail** | Xiongan | ×16 | 31.68 | 4.46 | ≈ OCR |
| **li2026bfmm** | CAVE | ×4 | 53.41 | 1.66 | ≈ OCR |
| **li2026bfmm** | Harvard | ×4 | 48.61 | 2.52 | ≈ OCR |
| **li2026bfmm** | Chikusei | ×4 | 44.61 | 2.50 | ≈ OCR |
| **li2026bfmm** | CAVE | ×8 | 51.20 | 1.98 | ≈ OCR |
| **li2026bfmm** | Harvard | ×8 | 47.69 | 2.72 | ≈ OCR |
| **li2026bfmm** | Chikusei | ×8 | 41.37 | 2.85 | ≈ OCR |
| **li2026bfmm** | WV-3 | pansharp | 37.95 | 2.76 | ≈ OCR |
| **dou2026bhsrnet** | CAVE | s8 | 46.69 | 2.98 | ≈ OCR |
| **dou2026bhsrnet** | Harvard | s8 | 48.08 | 2.29 | ≈ OCR |
| **dou2026bhsrnet** | Chikusei | s8 | 40.76 | 2.24 | ≈ OCR |
| **dou2026bhsrnet** | Houston | s8 | 54.92 | 0.75 | ≈ OCR |
| **dou2026bhsrnet** | CAVE | s32 | 41.33 | 5.10 | ≈ OCR |
| **dou2026bhsrnet** | Harvard | s32 | 46.34 | 2.69 | ≈ OCR |
| **dou2026bhsrnet** | Chikusei | s32 | 37.42 | 2.92 | ≈ OCR |
| **dou2026bhsrnet** | Houston | s32 | 53.36 | 0.80 | ≈ OCR |
| **dou2026bhsrnet** | LN01 | s8 (real) | 42.17 | 3.20 | ≈ OCR (no-GT) |
| **liu2026causal** | Harvard | ×8 | 47.04 | 2.88 | exact |
| **liu2026causal** | CAVE | ×8 | — | — | table rasterized → see paper |
| **liu2026causal** | ICVL | ×8 | — | — | table rasterized → see paper |
| **he2026diffusion** | Houston | ×4 | see paper | see paper | table rasterized |
| **he2026diffusion** | Pavia | ×4 | see paper | see paper | table rasterized |
| **he2026diffusion** | Chikusei | ×4 | see paper | see paper | table rasterized |
| **wang2026equivariant** | CAVE | ×4 (mosaiced+PAN) | 40.55 | 5.31 | exact |
| **wang2026equivariant** | ICVL | ×4 | 46.43 | 1.08 | exact |
| **wang2026equivariant** | real-world | ×4 | QNR 0.868 | QNR 0.868 | no-GT (QNR) |
| **xiao2026region** | WDCM | ×4 | 53.93 | 0.72 | ≈ OCR |
| **xiao2026region** | WDCM | ×8 | 52.46 | 0.84 | ≈ OCR |
| **xiao2026region** | Chikusei | ×4 | 48.10 | 0.79 | ≈ OCR |
| **xiao2026region** | Chikusei | ×8 | 47.21 | 0.88 | ≈ OCR |
| **xiao2026region** | Xiongan | ×4 | 52.60 | 0.38 | ≈ OCR |
| **xiao2026region** | Xiongan | ×8 | 50.36 | 0.49 | ≈ OCR |
| **song2026s2** | CAVE | ×8 | 49.05 | 2.22 | ≈ OCR |
| **song2026s2** | Harvard | ×8 | 47.42 | 2.85 | ≈ OCR |
| **song2026s2** | KAIST | ×8 | 45.88 | 2.47 | ≈ OCR |
| **xu2026scalmu** | Urban | ×8 | 41.49 | 2.17 | ≈ OCR |
| **xu2026scalmu** | PaviaU | ×8 | 40.15 | 2.43 | ≈ OCR |
| **xu2026scalmu** | Chikusei | ×8 | 48.03 | 2.65 | ≈ OCR |
| **xu2026scalmu** | PRISMA-Paris | ×8 | 43.24 | 4.42 | ≈ OCR |
| **xu2026scalmu** | Ziyuan-1 | ×3 (real) | QNR 0.958 | QNR 0.958 | no-GT (QNR) |
| **wang2026shotun** | PaviaC | ×8 | 43.76 | 4.61 | exact |
| **wang2026shotun** | KSC | ×8 | +0.8 dB Δ | −0.4° Δ | Δ vs PLRDiff |
| **wang2026shotun** | SanDiego | ×8 | +1.1 dB Δ | −0.6° Δ | Δ vs PLRDiff |
| **wang2026shotun** | Houston2018 | real | QNR 0.910 | QNR 0.910 | no-GT (QNR) |
| **liu2026semfnet** | Harvard | ×32 | +0.21 dB Δ | — | Δ vs subopt |
| **liu2026semfnet** | Houston | ×32 | +3.02 dB Δ | −0.74° Δ | Δ vs subopt |
| **liu2026semfnet** | PaviaU | ×32 | RMSE −2.23 Δ | RMSE −2.23 Δ | Δ vs subopt |
| **liu2026semfnet** | WV3 | real | QNR best | QNR best | Δ vs subopt (no-GT) |
| xiao2025dmzs | Pavia | pansharp | see paper | see paper | pansharpening |
| xiao2025dmzs | WDC | pansharp | see paper | see paper | pansharpening |
| xiao2025dmzs | Chikusei | pansharp | see paper | see paper | pansharpening |
| xiao2025dmzs | FR1 (PRISMA) | pansharp | see paper | see paper | pansharpening |
| peng2024fusionmamba | PanCollection | pansharp | see paper | see paper | pansharpening/HSI-MSI |
| peng2024fusionmamba | HyperPanCollection | pansharp | see paper | see paper | pansharpening |
| peng2024fusionmamba | CAVE | HISR | see paper | see paper | HISR |
| qu2025irarf | Pavia | arbitrary | see paper | see paper | arbitrary + unregistered |
| qu2025irarf | Chikusei | arbitrary | see paper | see paper | arbitrary + unregistered |
| qu2025irarf | WDC | arbitrary | see paper | see paper | arbitrary + unregistered |
| chen2025cyformer | CAVE | ×4 | see paper | see paper | — |
| chen2025cyformer | CAVE | ×8 | see paper | see paper | — |
| chen2025cyformer | Harvard | ×8 | see paper | see paper | — |
| chen2025cyformer | Chikusei | ×8 | see paper | see paper | — |
| guarino2025rhopnn | PRISMA | pansharp | see paper | see paper | zero-shot pansharpening |
| fang2024mimosst | CAVE | ×8 | 47.30 | — (SAM n/r) | exact (PSNR only) |
| fang2024mimosst | Harvard | ×8 | see paper | see paper | — |
| fang2024mimosst | Chikusei | ×8 | see paper | see paper | — |
| fang2024mimosst | Pavia | ×8 | see paper | see paper | — |
| fang2024mimosst | ICVL | ×8 | see paper | see paper | — |

## CAVE ×4 (in-domain) benchmark

Standardized CAVE ×4 (in-domain, Wald + SRF) benchmark in the usual SOTA format.
PSNR↑ / SSIM↑ higher is better; SAM↓ / ERGAS↓ lower is better.

| Method | Paper (title / note) | Year | Type | PSNR↑ | SSIM↑ | SAM↓ | ERGAS↓ | Protocol |
|---|---|---|---|---|---|---|---|---|
| BDT | external baseline — title/DOI to verify | 2023 | Unfolding | 52.30 | 0.997 | 1.93 | 1.02 | Wald + Nikon SRF |
| FeINFN | project method — title to verify | 2024 | INR | 52.47 | 0.998 | 1.91 | 0.98 | Wald + Nikon SRF |
| CoFusion | external baseline — title/DOI to verify | 2026 | CNN+Attn | 50.67 | 0.997 | 2.15 | 1.73 | Wald + SRF |
| SSA | external baseline — title/DOI to verify | 2026 | INR+MK | 45.92 | 0.996 | 2.02 | 1.07 | Wald, mixed datasets |
| Multi-path | external baseline — title/DOI to verify | 2026 | Transformer | 45.63 | 0.990 | 2.57 | 0.76 | Wald |
| SMF2Net | external baseline — title/DOI to verify | 2026 | CNN+HybridFormer | 43.91 | 0.997 | 2.04 | 6.40 | Wald |
| DSPNet | external baseline — title/DOI to verify | 2023 | CNN | 51.18 | 0.997 | 2.15 | 1.13 | Wald + Nikon SRF |
| 3DT-Net | external baseline — title/DOI to verify | 2023 | 3D CNN | 51.38 | 0.996 | 2.16 | 1.14 | Wald + Nikon SRF |
| PSRT | external baseline — title/DOI to verify | 2023 | Transformer | 50.47 | 0.996 | 2.19 | 2.06 | Wald + Nikon SRF |
| Fusformer | external baseline — title/DOI to verify | 2022 | Transformer | 44.52 | 0.983 | 4.12 | 1.06 | Wald |
| DHIF | external baseline — title/DOI to verify | 2022 | Deep + MMD | 51.07 | 0.997 | 2.01 | 1.22 | Wald + Nikon SRF |
| **li2026bfmm** [PDF] | Block Term Decomposition-Guided Frequency Mamba Modulation | 2026 | Sup (BTD+Mamba) | **53.41** | — (n/r) | **1.66** | — (n/r) | Wald (×4) |

> External-baseline values are as provided and must be verified/cited before
> publication; full paper titles and DOIs are pending. BFMM's SSIM/ERGAS were
> not printed in the source PDF (marked n/r = not reported). BFMM leads on both
> PSNR (53.41) and SAM (1.66) among the listed CAVE ×4 methods.

## 4. Corrections vs. the provided "verified" BibTeX
The supplied BibTeX claimed several DOIs/authors as verified; the source PDFs
contradict the following (now fixed in `hsim_fusion_comparison.bib`):

| Key | Pasted (WRONG) | Corrected from PDF |
|---|---|---|
| wang2026equivariant | DOI 10.1109/TIP.2026.3537982; datasets CAVE/Chikusei | DOI **10.1109/TIP.2026.3657219**; datasets **CAVE/ICVL** (no Chikusei) |
| dou2026bhsrnet | TGRS, DOI 10.1109/TGRS.2026.3541234, authors "Dou, Yujie et al." | **TIP 2026**, DOI **10.1109/TIP.2026.3714832**, authors **Mai Xu, Yongxuan Dou, Xin Deng, Xin Zou, Zhenwei Shi** |
| li2026bfmm | DOI 10.1109/TGRS.2026.3567891, author "Li, Rui and others" | DOI **10.1109/TGRS.2026.3699818**, authors **Yan Li, Chuangjie Fang, … Yi-Peng Liu** |
| liu2026semfnet | TIP, DOI 10.1109/TIP.2026.3567890, author "Xing, Yuan et al." | **TGRS 2026 (5501916)**, DOI **10.1109/TGRS.2026.3653545**, authors **Siyuan Liu, Zezheng Zhang, … Shuaiqi Liu** |
| chen2025cyformer | DOI 10.1109/TCSVT.2024.3456789 | DOI **10.1109/TCSVT.2024.3461829** |
| wang2026shotun | datasets CAVE/Harvard/Chikusei | datasets are **PaviaC/KSC/SanDiego/Houston** (not CAVE/Harvard/Chikusei) |

## 5. Dataset conventions
- **Simulated fusion (Wald protocol)**: CAVE (×8/×16/×32), Harvard (×8), Chikusei (×8), Pavia/ICVL.
- **Real / no-GT**: PaviaU, Chikusei, Houston2018, WV-3, PRISMA, LN01, Ziyuan-1, XDU-Liyukou — evaluated with no-reference metrics (QNR, Dλ, Ds).
- **Pansharpening / mosaiced-PAN** methods (DM-ZS, rho-PNN, Equivariant) are cross-task and use PRISMA/WV-3/PanCollection rather than CAVE/Harvard.
