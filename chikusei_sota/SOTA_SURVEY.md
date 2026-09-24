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

## Scope: published, peer-reviewed papers only

This survey counts only peer-reviewed journal and conference papers. arXiv-only preprints
(PIF-Net, CoFusion, HyDeFuse and others) are excluded. All 10 papers in the local folder are
IEEE TGRS / TIP 2026 publications.

**The highest published Chikusei ×4 number is still the TIP'26 Two-Stage Diffusion result,
56.19 dB**, so the target stands.

The runner-ups in that table are also published: SMGU-Net (Pattern Recognition 2025), MIMO-SST
(IEEE TGRS 2024, [10419133](https://ieeexplore.ieee.org/document/10419133)), DDIF (Information
Fusion 2024; code at [294coder/Dif-PAN](https://github.com/294coder/Dif-PAN), pansharpening data
only), DCT and AMSF (IEEE TGRS 2025, [10824890](https://ieeexplore.ieee.org/document/10824890)).
CLSNet (2026) could not be located.

LGCT (IEEE TGRS 2024, [10742406](https://ieeexplore.ieee.org/document/10742406)) is a published
Chikusei method with public code ([Hewq77/LGCT](https://github.com/Hewq77/LGCT)).

### IEEE Xplore full-text search: published Chikusei HSI–MSI fusion papers in top venues

Query: full text contains "Chikusei", title contains "fusion", metadata contains "hyperspectral"
and "multispectral", years 2023–2026. It returned 108 hits (84 subscription, 24 open access). The table keeps
only **subscription (non-open-access)** papers from TPAMI / TIP / TNNLS / TGRS / TCSVT / TMM /
TCYB / TCI / CVPR, using the access type IEEE Xplore reports for each paper.

The papers not marked "yes" still need their Chikusei tables read. That needs the PDFs in
`literature_survey/chikusei_papers/`, and bulk download from IEEE Xplore was not possible from
this session.

| Year | Venue | Title | Cites | In local folder | IEEE |
|---|---|---|---|---|---|
| 2026 | TGRS | Diffusion-Driven Mutual Enhancement of Matching and Fusion for Reference-Based Hyperspectral Image Super-Resolution | 17 |  | [11358932](https://ieeexplore.ieee.org/document/11358932) |
| 2026 | TGRS | Region-Aware MoE Network for Hyperspectral and Multispectral Image Fusion | 8 | yes | [11471838](https://ieeexplore.ieee.org/document/11471838) |
| 2026 | TGRS | Leveraging Modal Interaction and Window Dilation in Attention Network for Hyperspectral and Multispectral Remote Sensing Image Fusion | 2 | yes | [11370239](https://ieeexplore.ieee.org/document/11370239) |
| 2026 | TGRS | A Stagewise Spectral-Structural Coordinated Reconstruction Network for Hyperspectral and Multispectral Image Fusion | 1 | yes | [11612934](https://ieeexplore.ieee.org/document/11612934) |
| 2026 | TIP | Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion | 1 | yes | [11321074](https://ieeexplore.ieee.org/document/11321074) |
| 2026 | TGRS | S2TA-Fuse: Semantic-Superpixel Tokenized Attention for Spatial–Spectral Fusion | 1 |  | [11363600](https://ieeexplore.ieee.org/document/11363600) |
| 2026 | TGRS | A Detail Injection-Based Fusion Framework for Hyperspectral, Multispectral, and Panchromatic Remote Sensing Images | 0 | yes | [11480191](https://ieeexplore.ieee.org/document/11480191) |
| 2026 | TIP | A Two-Stage Conditional Diffusion Model With Differential Attention for Hyperspectral and Multispectral Image Fusion | 0 | yes | [11658817](https://ieeexplore.ieee.org/document/11658817) |
| 2026 | TMM | Arbitrary-Scale Fusion Operator for High-Resolution Hyperspectral Imaging | 0 |  | [11364051](https://ieeexplore.ieee.org/document/11364051) |
| 2026 | TGRS | Block Term Decomposition-Guided Frequency Mamba Modulation for Hyperspectral Image Fusion | 0 |  | [11550081](https://ieeexplore.ieee.org/document/11550081) |
| 2026 | TIP | Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model | 0 |  | [11623387](https://ieeexplore.ieee.org/document/11623387) |
| 2026 | TCI | Constrained Conditional Denoising Diffusion for Hyperspectral-Multispectral Fusion | 0 |  | [11297763](https://ieeexplore.ieee.org/document/11297763) |
| 2026 | TCYB | FK-Net: Frequency-Aware and Kernelizable Mamba–Transformer for Multispectral and Hyperspectral Image Fusion | 0 |  | [11692970](https://ieeexplore.ieee.org/document/11692970) |
| 2026 | TGRS | GSFL: Graph-Based Spatial–Frequency Learning for Multispectral and Hyperspectral Image Fusion | 0 |  | [11618738](https://ieeexplore.ieee.org/document/11618738) |
| 2026 | TGRS | HDGMamba: High-Frequency Dynamic Guided Mamba for Robust Multispectral-Hyperspectral Image Fusion | 0 | yes | [11523545](https://ieeexplore.ieee.org/document/11523545) |
| 2026 | TGRS | Hyperspectral and Multispectral Image Fusion via Coupled Tensor Wheel Decomposition | 0 |  | [11538234](https://ieeexplore.ieee.org/document/11538234) |
| 2026 | TGRS | Local–Global Progressive Network for Hyperspectral and Multispectral Image Fusion | 0 | yes | [11589378](https://ieeexplore.ieee.org/document/11589378) |
| 2026 | TCSVT | MoEformer: a Frequency-Guided Mixture of Experts Transformer for Hyperspectral and Multispectral Image Fusion | 0 |  | [11690674](https://ieeexplore.ieee.org/document/11690674) |
| 2026 | TGRS | MosyMamba: Modal-Synergy Mamba Network for Hyperspectral and Multispectral Image Fusion | 0 | yes | [11614875](https://ieeexplore.ieee.org/document/11614875) |
| 2026 | TGRS | PMHIF-Net: A Prior-Guided Mamba Hierarchical Interactive Fusion Network for Hyperspectral and Multispectral Image Fusion | 0 | yes | [11655942](https://ieeexplore.ieee.org/document/11655942) |
| 2026 | TGRS | SCALMU: Synthetically Trained Coupling of Adaptive Learned Multiplicative Updates for Hyperspectral–Multispectral Fusion | 0 |  | [11605969](https://ieeexplore.ieee.org/document/11605969) |
| 2026 | TGRS | TDP-Net: Unsupervised HSI-MSI Fusion via Tucker Decomposition With Generative Diffusion Priors | 0 |  | [11660849](https://ieeexplore.ieee.org/document/11660849) |
| 2026 | TGRS | TM-MOE: A Degradation-Aware Transformer-Mamba MoE for Hyperspectral and Multispectral Image Fusion | 0 |  | [11494874](https://ieeexplore.ieee.org/document/11494874) |
| 2026 | TGRS | Unsupervised Deformable Bilinear Fusion Network for Unregistered Hyperspectral Image Super-Resolution | 0 |  | [11627931](https://ieeexplore.ieee.org/document/11627931) |
| 2025 | TCSVT | Cyclic Cross-Modality Interaction for Hyperspectral and Multispectral Image Fusion | 57 |  | [10681101](https://ieeexplore.ieee.org/document/10681101) |
| 2025 | TGRS | Mamba Collaborative Implicit Neural Representation for Hyperspectral and Multispectral Remote Sensing Image Fusion | 54 |  | [10869490](https://ieeexplore.ieee.org/document/10869490) |
| 2025 | TPAMI | An Efficient Image Fusion Network Exploiting Unifying Language and Mask Guidance | 28 |  | [11091495](https://ieeexplore.ieee.org/document/11091495) |
| 2025 | TPAMI | Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model | 24 |  | [11092683](https://ieeexplore.ieee.org/document/11092683) |
| 2025 | TNNLS | Unsupervised Hyperspectral and Multispectral Image Blind Fusion Based on Deep Tucker Decomposition Network With Spatial–Spectral Manifold Learning | 24 |  | [10705122](https://ieeexplore.ieee.org/document/10705122) |
| 2025 | TGRS | SSDT: Multiscale Spatial–Spectral Dilated Transformer for Hyperspectral and Multispectral Image Fusion | 23 |  | [11165009](https://ieeexplore.ieee.org/document/11165009) |
| 2025 | TGRS | An Asymptotic Multiscale Symmetric Fusion Network for Hyperspectral and Multispectral Image Fusion | 19 |  | [10824890](https://ieeexplore.ieee.org/document/10824890) |
| 2025 | TGRS | GFHMP: Gradual Fusion Framework of Hyperspectral, Multispectral, and Panchromatic Images Using a Novel Spatial–Spectral Cross-Fusion Network | 18 |  | [11126176](https://ieeexplore.ieee.org/document/11126176) |
| 2025 | TNNLS | Advancing Hyperspectral and Multispectral Image Fusion: An Information-Aware Transformer-Based Unfolding Network | 16 |  | [10536168](https://ieeexplore.ieee.org/document/10536168) |
| 2025 | TGRS | Adaptive Expert Learning for Hyperspectral and Multispectral Image Fusion | 15 |  | [11202498](https://ieeexplore.ieee.org/document/11202498) |
| 2025 | TGRS | CESFusion: Cross-Frequency Enhanced Spatial—Spectral Fusion Network for Hyperspectral and Multispectral Image Fusion | 11 |  | [10975030](https://ieeexplore.ieee.org/document/10975030) |
| 2025 | TGRS | Dilated Transformation-Guided Unsupervised Multimodal Learning for Hyperspectral and Multispectral Image Fusion | 8 |  | [11271725](https://ieeexplore.ieee.org/document/11271725) |
| 2025 | TGRS | RAMSF: A Novel Generic Framework for Optical Remote Sensing Multimodal Spatial-Spectral Fusion | 8 |  | [10934049](https://ieeexplore.ieee.org/document/10934049) |
| 2025 | TGRS | Integrated Fusion for Panchromatic, Multispectral, Hyperspectral Remote Sensing Images: Insights From Multispectral Images | 6 |  | [11097365](https://ieeexplore.ieee.org/document/11097365) |
| 2025 | TGRS | Progressive Synergistic Registration and Fusion Diffusion Network for Unregistered Hyperspectral and Multispectral Image Fusion | 6 |  | [10976398](https://ieeexplore.ieee.org/document/10976398) |
| 2025 | TPAMI | Building Non-Uniform Degradation Model for Position-Aware Hyperspectral Image Fusion | 5 |  | [11230110](https://ieeexplore.ieee.org/document/11230110) |
| 2025 | TGRS | AEWFNet: Adaptive Enhancement and Wavelet Convolution for Hyperspectral and Multispectral Image Fusion | 4 |  | [11245613](https://ieeexplore.ieee.org/document/11245613) |
| 2025 | TGRS | Hyperspectral and Multispectral Image Fusion With Functional Data Analysis Techniques | 4 |  | [11002560](https://ieeexplore.ieee.org/document/11002560) |
| 2025 | TGRS | Spatial–Spectral Cross Mamba Network for Hyperspectral and Multispectral Image Fusion | 3 |  | [11153570](https://ieeexplore.ieee.org/document/11153570) |
| 2025 | TGRS | A Progressive Registration-Fusion Co-Optimization A-Mamba Network: Toward Deep Unregistered Hyperspectral and Multispectral Fusion | 1 |  | [11006139](https://ieeexplore.ieee.org/document/11006139) |
| 2025 | TGRS | A Progressive Spatial–Spectral Interactive Network for Integrated Fusion of Panchromatic, Multispectral, and Hyperspectral Images | 0 |  | [11129110](https://ieeexplore.ieee.org/document/11129110) |
| 2025 | TGRS | Unsupervised Model-Embedded Two-Stage Diffusion Method for Multispectral and Hyperspectral Image Fusion | 0 |  | [11165469](https://ieeexplore.ieee.org/document/11165469) |
| 2024 | TGRS | Unsupervised Hybrid Network of Transformer and CNN for Blind Hyperspectral and Multispectral Image Fusion | 77 |  | [10415455](https://ieeexplore.ieee.org/document/10415455) |
| 2024 | TGRS | MIMO-SST: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion | 58 |  | [10419133](https://ieeexplore.ieee.org/document/10419133) |
| 2024 | TNNLS | Interpretable Model-Driven Deep Network for Hyperspectral, Multispectral, and Panchromatic Image Fusion | 40 |  | [10138912](https://ieeexplore.ieee.org/document/10138912) |
| 2024 | TNNLS | Unsupervised Deep Tensor Network for Hyperspectral–Multispectral Image Fusion | 40 |  | [10115230](https://ieeexplore.ieee.org/document/10115230) |
| 2024 | TGRS | LGCT: Local-Global Collaborative Transformer for Fusion of Hyperspectral and Multispectral Images | 38 |  | [10742406](https://ieeexplore.ieee.org/document/10742406) |
| 2024 | TGRS | A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion | 34 |  | [10500430](https://ieeexplore.ieee.org/document/10500430) |
| 2024 | TGRS | Progressive Multi-Iteration Registration-Fusion Co-Optimization Network for Unregistered Hyperspectral Image Super-Resolution | 29 |  | [10546322](https://ieeexplore.ieee.org/document/10546322) |
| 2024 | TGRS | Domain Transform Model Driven by Deep Learning for Anti-Noise Hyperspectral and Multispectral Image Fusion | 15 |  | [10335620](https://ieeexplore.ieee.org/document/10335620) |
| 2024 | TGRS | CODE-IF: A Convex/Deep Image Fusion Algorithm for Efficient Hyperspectral Super-Resolution | 14 |  | [10493051](https://ieeexplore.ieee.org/document/10493051) |
| 2024 | TCI | INF3: Implicit Neural Feature Fusion Function for Multispectral and Hyperspectral Image Fusion | 14 |  | [10750035](https://ieeexplore.ieee.org/document/10750035) |
| 2024 | TGRS | All in One: A Unified Network for Hyperspectral Image Fusion | 13 |  | [10557664](https://ieeexplore.ieee.org/document/10557664) |
| 2024 | TGRS | Deep Unfolding Network Enhanced by Transformer Priors for Unregistered Hyperspectral and Multispectral Image Fusion | 12 |  | [10681157](https://ieeexplore.ieee.org/document/10681157) |
| 2024 | TGRS | A Self-Supervised Spaceborne Multispectral and Hyperspectral Image Fusion Unrolling Network | 11 |  | [10555311](https://ieeexplore.ieee.org/document/10555311) |
| 2024 | TGRS | MMIF: Interpretable Hyperspectral and Multispectral Image Fusion via Maximum Mutual Information | 6 |  | [10360850](https://ieeexplore.ieee.org/document/10360850) |
| 2023 | TGRS | Decoupled-and-Coupled Networks: Self-Supervised Hyperspectral Image Super-Resolution With Subpixel Fusion | 193 |  | [10285378](https://ieeexplore.ieee.org/document/10285378) |
| 2023 | TGRS | PSRT: Pyramid Shuffle-and-Reshuffle Transformer for Multispectral and Hyperspectral Image Fusion | 164 |  | [10044141](https://ieeexplore.ieee.org/document/10044141) |
| 2023 | TGRS | Dual Spatial–Spectral Pyramid Network With Transformer for Hyperspectral Image Fusion | 47 |  | [10264151](https://ieeexplore.ieee.org/document/10264151) |
| 2023 | TGRS | Stereo Cross-Attention Network for Unregistered Hyperspectral and Multispectral Image Fusion | 14 |  | [10250890](https://ieeexplore.ieee.org/document/10250890) |
| 2023 | TGRS | MGFEI-Net: Multiscale Grouping Feedback Embedded Integrated Network for Panchromatic, Multispectral, and Hyperspectral Image Fusion | 8 |  | [10292795](https://ieeexplore.ieee.org/document/10292795) |

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
