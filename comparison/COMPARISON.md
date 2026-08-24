# HSI–MSI Image Fusion: Method Comparison (2024–2026)

Consolidated comparison of recent hyperspectral–multispectral (and related
pansharpening / mosaiced-PAN) fusion methods. Metadata and SOTA figures for the
11 papers marked **[PDF]** were extracted directly from the source PDFs
(`C:\Users\sande\Downloads\Projects\Projects`). Where a result table was a
rasterized image, values were recovered via OCR and are **approximate**; such
cells are marked "(OCR)". Entries whose tables could not be OCR'd are marked
"table rasterized".

A broader 23-paper survey (dataset-usage matrix for CAVE/Harvard/Chikusei/PaviaU)
is in [`../review/DATASETS.md`](../review/DATASETS.md).

## Legend
- **Type**: Sup = supervised, Unsup = unsupervised/blind, Zero = zero-shot, MB = model-based/unrolled.
- Scales: ×4 / ×8 / ×16 / ×32 spatial upscaling; "s=8" etc. means the same.
- SOTA format: `PSNR/SAM` in `dB/°` unless noted.

## Comparison table

| Method (key) | Venue / Year | Type | Eval datasets | Best reported PSNR/SAM (scale) | DOI |
|---|---|---|---|---|---|
| **fang2026detail** [PDF] | TGRS 2026 | Sup (unfolding) | Chikusei, Houston, Xiongan (HSI-MSI-PAN) | Chikusei 30.71/7.52; Houston 29.68/8.13; Xiongan 31.68/4.46 (MPSNR/MPSAM) | 10.1109/TGRS.2026.3683056 |
| **li2026bfmm** [PDF] | TGRS 2026 | Sup (BTD+Mamba) | CAVE, Harvard, Chikusei (×4,×8), WHU-MHF, WV-3 | CAVE×4 53.41/1.66; Harvard×4 48.61/2.52; Chikusei×4 44.61/2.50; CAVE×8 51.20/1.98; Harvard×8 47.69/2.72; Chikusei×8 41.37/2.85; WV-3 37.95/2.76 (OCR) | 10.1109/TGRS.2026.3699818 |
| **dou2026bhsrnet** [PDF] | TIP 2026 | Sup (unfolding, blur-robust) | CAVE, Harvard, Chikusei, Houston (s=8,s=32), LN01(real) | CAVE s8 46.69/2.98; Harvard s8 48.08/2.29; Chikusei s8 40.76/2.24; Houston s8 54.92/0.75; CAVE s32 41.33/5.10; Harvard s32 46.34/2.69; Chikusei s32 37.42/2.92; Houston s32 53.36/0.80; LN01 42.17/3.20 (OCR) | 10.1109/TIP.2026.3714832 |
| **liu2026causal** [PDF] | TGRS 2026 | Sup blind | CAVE, Harvard, ICVL (×8), UH(real) | Harvard×8 47.04/2.88; CAVE/ICVL table rasterized | 10.1109/TGRS.2026.3703367 |
| **he2026diffusion** [PDF] | TGRS 2026 | Sup (diffusion, ref-based) | Houston, Pavia, Chikusei (×4), XDU-Liyukou(real) | tables rasterized (not OCR-readable) | 10.1109/TGRS.2026.3656069 |
| **wang2026equivariant** [PDF] | TIP 2026 | Unsup (equivariant) | CAVE, ICVL (×4 mosaiced+PAN), real(16b) | CAVE×4 40.55/5.31; ICVL×4 46.43/1.08; real QNR 0.868 | 10.1109/TIP.2026.3657219 |
| **xiao2026region** [PDF] | TGRS 2026 | Sup (MoE) | WDCM, Chikusei, Xiongan (×4,×8), YRE(real) | WDCM×4 53.93/0.72, ×8 52.46/0.84; Chikusei×4 48.10/0.79, ×8 47.21/0.88; Xiongan×4 52.60/0.38, ×8 50.36/0.49 (OCR) | 10.1109/TGRS.2026.3680287 |
| **song2026s2** [PDF] | TGRS 2026 | Sup | CAVE, Harvard, KAIST (×8) | CAVE×8 49.05/2.22; Harvard×8 47.42/2.85; KAIST×8 45.88/2.47 (OCR) | 10.1109/TGRS.2026.3671284 |
| **xu2026scalmu** [PDF] | TGRS 2026 | MB unrolled (blind) | Urban, PaviaU, Chikusei, PRISMA-Paris (×8), Ziyuan-1(real) | Urban×8 41.49/2.17; PaviaU×8 40.15/2.43; Chikusei×8 48.03/2.65; PRISMA-Paris×8 43.24/4.42; Ziyuan-1 QNR 0.958 (OCR) | 10.1109/TGRS.2026.3712501 |
| **wang2026shotun** [PDF] | TIP 2026 | Unsup/blind | PaviaC, KSC, SanDiego (×8), Houston2018(real) | PaviaC×8 43.76/4.61; KSC/SanDiego ~+0.8–1.1 dB over PLRDiff; Houston2018 QNR 0.910 | 10.1109/TIP.2026.3695389 |
| **liu2026semfnet** [PDF] | TGRS 2026 | Sup | Harvard, Houston, PaviaU (×32), WV3(real) | Harvard +0.21 dB vs subopt; Houston +3.02 dB / −0.74° SAM; PaviaU RMSE −2.23; WV3 QNR best (tables rasterized) | 10.1109/TGRS.2026.3653545 |
| xiao2025dmzs | CVPR 2025 | Zero (diffusion pansharp.) | Pavia, WDC, Chikusei, FR1(PRISMA) | see paper (pansharpening) | 10.1109/CVPR52734.2025.01182 |
| peng2024fusionmamba | TGRS 2024 | Sup (Mamba) | PanCollection, HyperPanCollection, CAVE | see paper | 10.1109/TGRS.2024.3496073 |
| qu2025irarf | TIP 2025 | Sup (arbitrary+unregistered) | Pavia, Chikusei, WDC | see paper | 10.1109/TIP.2025.3551531 |
| chen2025cyformer | TCSVT 2025 | Sup (Transformer) | CAVE (×4,×8), Harvard, Chikusei | see paper | 10.1109/TCSVT.2024.3461829 |
| hsu2024csakd | arXiv 2406.19666 (TIP subm.) | Sup (KD) | AVIRIS→Landsat MSI | see arXiv | arXiv:2406.19666 |
| guarino2025rhopnn | TGRS 2025 | Zero (pansharp.) | PRISMA | see paper | 10.1109/TGRS.2025.3583877 |
| fang2024mimosst | TGRS 2024 | Sup (Transformer) | CAVE, Harvard, Chikusei, Pavia, ICVL | CAVE×8 47.30 dB (reported prior SOTA) | 10.1109/TGRS.2024.3361553 |
| liu2026dsirnet | Remote Sensing 2026 | Sup (SSM+INR) | Houston, PaviaU, Botswana, Chikusei | +0.04–0.20 dB over baselines | 10.3390/rs18050789 |
| shan2025bfctn | arXiv 2510.18400 | MB (Bayesian tensor) | CAVE, Harvard, Chikusei | see arXiv | arXiv:2510.18400 |
| jiang2025that | arXiv 2508.08183 | Sup (Transformer pansharp.) | WV3 etc. | see arXiv | arXiv:2508.08183 |

## Corrections vs. the provided "verified" BibTeX
The pasted BibTeX claimed several DOIs/authors as verified; the source PDFs
contradict the following (now fixed in `hsim_fusion_comparison.bib`):

| Key | Pasted (WRONG) | Corrected from PDF |
|---|---|---|
| wang2026equivariant | DOI 10.1109/TIP.2026.3537982; datasets CAVE/Chikusei | DOI **10.1109/TIP.2026.3657219**; datasets **CAVE/ICVL** (no Chikusei) |
| dou2026bhsrnet | TGRS, DOI 10.1109/TGRS.2026.3541234, authors "Dou, Yujie et al." | **TIP 2026**, DOI **10.1109/TIP.2026.3714832**, authors **Mai Xu, Yongxuan Dou, Xin Deng, Xin Zou, Zhenwei Shi** |
| li2026bfmm | DOI 10.1109/TGRS.2026.3567891, author "Li, Rui and others" | DOI **10.1109/TGRS.2026.3699818**, authors **Yan Li, Chuangjie Fang, … Yi-Peng Liu** |
| liu2026semfnet | TIP, DOI 10.1109/TIP.2026.3567890, author "Xing, Yuan et al." | **TGRS 2026 (5501916)**, DOI **10.1109/TGRS.2026.3653545**, authors **Siyuan Liu, Zezheng Zhang, … Shuaiqi Liu** |
| chen2025cyformer | DOI 10.1109/TCSVT.2024.3456789 | DOI **10.1109/TCSVT.2024.3461829** |
| hsu2024csakd | DOI 10.1109/TIP.2024.3412345 | TIP DOI **unverified**; only arXiv:2406.19666 confirmed |
| wang2026shotun | datasets CAVE/Harvard/Chikusei | datasets are **PaviaC/KSC/SanDiego/Houston** (not CAVE/Harvard/Chikusei) |

## Notes on dataset conventions
- **Simulated fusion (Wald protocol)**: CAVE (×8/×16/×32), Harvard (×8), Chikusei (×8), Pavia/ICVL.
- **Real / no-GT**: PaviaU, Chikusei, Houston2018, WV-3, PRISMA, LN01, Ziyuan-1, XDU-Liyukou — evaluated with no-reference metrics (QNR, Dλ, Ds).
- **Pansharpening / mosaiced-PAN** methods (DM-ZS, rho-PNN, THAT, Equivariant) are cross-task and use PRISMA/WV-3/PanCollection rather than CAVE/Harvard.
