# Chikusei HSI–MSI Fusion: State of the Art (literature_survey/chikusei_papers)

All numbers below come from the tables in the 10 papers in `literature_survey/chikusei_papers/`,
read from the rendered PDF pages. **Chikusei PSNR values are only comparable inside one paper**,
because each paper uses its own SRF, blur kernel, crop, split, normalization and PSNR definition.
The spread for the same method across papers is large. PSRT, for example, gets 24.47 dB in one
paper, 30.73 dB in another, 40.02 dB in a third and 50.67 dB in a fourth.

## ×4 Chikusei: best method in each paper

| Paper (venue, year) | Protocol (MSI SRF / blur / split) | Best method (theirs) | PSNR ↑ | SAM ↓ | ERGAS ↓ | SSIM ↑ | Runner-up |
|---|---|---|---|---|---|---|---|
| **Two-Stage Cond. Diffusion + Diff. Attention** (TIP 2026) | WorldView-2 8-band / Gauss 7×7 σ=2 / 832 train 64², 64 val 64², **8 test 256²** | Ours | **56.1871** | 0.7203 | 1.5158 | 0.9982 | CLSNet 55.2007, SMGU 54.5244, MIMOSST 53.6073, DDIF 53.4421, MIMFormer 53.3578, DCT 52.1998 |
| Region-Aware MoE (TGRS 2026) | YOKOYA 5-band / Gauss 5×5 σ=2 / 70 %/30 % split along height, 64² patches | Ours (RAMoE) | 48.10 | 0.79 | 0.4713 | – | DCTransformer 47.54, PSTUN 47.40, DTUML 47.23, 3DT-Net 47.05 |
| SSCNet – Stagewise Spectral-Structural (TGRS 2026) | own | SSCNet | 46.1319 | 1.1614 | 1.2611 | 0.9941 | AELF 43.6370, SFIGNet 43.2249 |
| CDPS – Coupled Diffusion Posterior Sampling (TIP 2026, *unsupervised*) | 5-band MSI | CDPS | 44.26 | 1.59 | 1.87 | 0.990 | HyCoNet 43.84 |
| PMHIF-Net (TGRS 2026) | own | PMHIF-Net | 44.0506 | 1.0514 | 1.4041 | 0.9806 | LGCT 43.4825, DHMF 43.3834 |
| MIAN – Modal Interaction + Window Dilation (TGRS 2026) | airborne SRF, ×4 | MIAN | 31.7995 | 1.8900 | 2.8673 | 0.9669 | DCT 31.5188 |
| HDGMamba (TGRS 2026, *noisy inputs*) | trained-noise setting | HDGMamba | 36.555 | 2.359 | 0.924 | 0.977 | AHMNet 35.748 |
| DIM-HMPF – Detail Injection (TGRS 2026, HSI+MSI+PAN) | Nikon D700 / Gauss σ=2 | DIM-HMPF | 30.7108 | 7.5218 | 5.4223 | 0.8370 (MSSIM) | HMPNet 30.4856 |

## Additional papers found by web search (2025–2026)

| Paper | Chikusei protocol | ×4 best (PSNR / SSIM / SAM / ERGAS) | Other ×4 rows in the same table | Code |
|---|---|---|---|---|
| [PIF-Net](https://arxiv.org/abs/2508.00453) (arXiv 2508.00453) | top-left 1000×2000 train, rest cut into 680×680 test tiles; Gaussian 3×3, σ=0.5; also ×2 / ×8 | PIF-Net 51.6257 / 0.9983 / 2.0653 / 1.5401 | SMGU-Net 51.3382, PSRT 50.5377, U2Net 50.5061, Fusformer 50.1466, HSRnet 49.3548, 3DT-Net 48.1940, GSA 31.2757 | not released |
| [CoFusion](https://arxiv.org/abs/2604.10584) (arXiv 2604.10584) | Wald, Gaussian blur, ×2/×4/×8 (kernel not stated) | CoFusion 50.6742 / 0.9971 / 2.1494 / 1.7252 | SMGU-Net 49.8316, PSRT 49.0346, U2Net 49.0028, FMPM-DNet 48.7125, Fusformer 48.6413, BUGPan 48.3217, HSRnet 47.8521, 3DT-Net 46.6914 | not released |
| [HyDeFuse](https://arxiv.org/abs/2509.02477) (arXiv 2509.02477) | 540×480 crop, classical methods only | HyDeFuse 42.27 / – / 1.79 / 1.39 | HySure 40.56, bicubic 29.02 | not released |
| [ASSR-Net](https://arxiv.org/abs/2604.05742) (arXiv 2604.05742) | does **not** evaluate Chikusei (CAVE / Harvard / Gaofen5) | – | – | – |
| [USP-Mamba](https://arxiv.org/abs/2608.02401) (arXiv 2608.02401) | single-image HSI SR, not HSI–MSI fusion | – | – | – |

**No paper found reports Chikusei ×4 above 56.19 dB.** The TIP'26 Two-Stage Diffusion number is
still the highest, so the target stands.

**Baselines that could not be located.** CLSNet (2026, the 55.20 dB runner-up) did not turn up in any
search, so its own Chikusei protocol is unknown. SMGU-Net (Pattern Recognition 2025,
[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324010288)) evaluates Chikusei
but has no public code. DDIF (Information Fusion 2024) has code at
[294coder/Dif-PAN](https://github.com/294coder/Dif-PAN), which only ships pansharpening datasets.

### Protocol facts pinned down by the search

* **PSRT metric code** ([shangqideng/PSRT `metrics.py`](https://github.com/shangqideng/PSRT)), which the
  TIP'26 paper cites as its data source:
  * `PSNR = 10·log10(max(GT)² / mean((GT−X)²))`: a single PSNR over the whole cube, with the
    peak set to the GT image's own max.
  * `SAM`: each cube is first divided by its own max, then the angle is averaged over pixels.
  * `ERGAS = 100/4 · sqrt(mean_b(RMSE_b² / mean_b²))`.

  PSNR is therefore **not** computed band by band with peak 1. Because the peak is the GT max, it
  comes out higher than a peak-1 PSNR on dark scenes whose max is below 1, which explains part of
  the 24–56 dB spread between papers. Our evaluation reports both definitions.
* **WorldView-2 SRF**: DigitalGlobe's technical note
  ([PDF](https://wp-cdn.apollomapping.com/web_assets/user_uploads/2014/10/14123451/Spectral_Response_for_DigitalGlobe_Earth_Imaging_Instruments_102214.pdf),
  Table 5) gives the 5 %-response edges and centre wavelengths:
  Coastal 396–458 (427), Blue 442–515 (478), Green 506–586 (546), Yellow 584–632 (608),
  Red 624–694 (659), Red-Edge 699–749 (724), NIR1 765–901 (833), NIR2 856–1043 (949) nm.
  The full curves are published only as plots, so `hsifuse/ops.py::wv2_srf` builds flat-top bands
  whose flanks reach 5 % response exactly at these edges.

## ×8 Chikusei (for reference)

| Paper | Best | PSNR | Runner-up |
|---|---|---|---|
| PIF-Net (arXiv 2508.00453) | PIF-Net | 50.0124 | SMGU-Net 49.6582, U2Net 48.8911 |
| CoFusion (arXiv 2604.10584) | CoFusion | 48.9371 | SMGU-Net 48.1512, U2Net 47.3945 |
| Region-Aware MoE | RAMoE | 47.21 | DCTransformer 46.03 |
| LGP-Net – Local/Global Progressive (TGRS 2026) | LGP-Net | 44.96 | BDT 44.47, 3DT 43.78 |
| SSCNet | SSCNet | 43.7902 | AELF 42.8146 |
| MosyMamba (TGRS 2026, Landsat-8 SRF) | MosyMamba | 42.5068 | FusionMamba 40.5878 |

## Target selected for this project

The ×4 protocol with the highest reported numbers, and the one matching the "≈52–53 dB" SOTA
band we had been targeting, is the **Two-Stage Conditional Diffusion (TIP 2026)** protocol:

* LR-HSI: Gaussian blur 7×7, σ = 2, ×4 decimation
* HR-MSI: WorldView-2 SRF (8 bands)
* Test: 8 images of 256×256×128; train on 64×64 patches from the remaining area

Numbers to beat under that protocol:

| Method | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ |
|---|---|---|---|---|
| **Two-Stage Diffusion (SOTA)** | **56.1871** | **0.9982** | **0.7203** | **1.5158** |
| CLSNet (2026) | 55.2007 | 0.9979 | 0.7836 | 1.6318 |
| SMGU (2025) | 54.5244 | 0.9977 | 0.8719 | 1.6405 |
| MIMOSST (2024) | 53.6073 | 0.9972 | 0.9315 | 1.8119 |
| DDIF (2024) | 53.4421 | 0.9972 | 0.9387 | 1.8087 |
| DCT (2024) | 52.1998 | 0.9977 | 0.9239 | 1.7448 |
| PSRT (2023) | 50.6722 | 0.9952 | 1.2733 | 2.0107 |

Our earlier in-repo Chikusei results, for context: KrylovNet 43.69 dB (3-Gaussian SRF protocol).
The diffusion-NullFusion and NullFusion-PP-v2 Chikusei Kaggle runs all crashed before
producing a test number (shape mismatches, a missing dataset path, and the MATLAB v7.3 `.mat`
needing `h5py`).
