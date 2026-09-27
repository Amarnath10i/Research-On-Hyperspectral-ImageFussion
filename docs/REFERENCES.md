# References used in this repository

Every research paper referenced anywhere in the repository, deduplicated by DOI and title (generated 2026-09-26). The PDFs themselves are kept locally in `literature/` and are not in git.

| | Papers |
|---|---|
| Papers with a full title (bibliographies, survey, literature-search database) | **244** |
| Papers cited only by name or author-year in docs and code | **64** |
| **Total unique papers** | **308** |
| of which read in full from the PDF | 19 |
| Dataset references (not papers) | 18 |

Of the 244 titled papers, 199 are on HSI-MSI fusion, HSI super-resolution or pansharpening. The other 45 come from the raw IEEE literature search (HSI classification, change detection, multimodal HSI+LiDAR, general super-resolution).

Where each titled paper appears (one paper can appear in several places):

| Code | Source | Papers |
|---|---|---|
| PDF | Read in full: `literature/chikusei_papers/` PDFs and the 11 `[PDF]` entries of `paper/comparison/COMPARISON.md` | 19 |
| M | Manuscript bibliography `paper/hsim_fusion_ieee_if9.bib` (also `docs/DATASETS.md`) | 23 |
| C | Comparison bibliography `paper/comparison/hsim_fusion_comparison.bib` and `COMPARISON.md` | 29 |
| S | IEEE Xplore list in `docs/CHIKUSEI_SOTA_SURVEY.md` | 65 |
| Lc | Literature candidates `docs/literature/lit_candidates.json` | 45 |
| Lm | Multimodal list `docs/literature/lit_multimodal.json` | 28 |
| Lr | Raw IEEE search `docs/literature/lit_raw.json` | 149 |

## 1. Papers read in full (19)

| # | Paper | Venue | Year |
|---|---|---|---|
| 1 | A Detail Injection-Based Fusion Framework for Hyperspectral, Multispectral, and Panchromatic Remote Sensing Images | TGRS | 2026 |
| 2 | A Stagewise Spectral-Structural Coordinated Reconstruction Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 |
| 3 | A Two-Stage Conditional Diffusion Model With Differential Attention for Hyperspectral and Multispectral Image Fusion | TIP | 2026 |
| 4 | Block Term Decomposition-Guided Frequency Mamba Modulation for Hyperspectral Image Fusion | TGRS | 2026 |
| 5 | Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model | TIP | 2026 |
| 6 | Causal Degradation-Guided Network With Spatial-Frequency Attention for Blind Hyperspectral Image Fusion | TGRS | 2026 |
| 7 | Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion | TIP | 2026 |
| 8 | Diffusion-Driven Mutual Enhancement of Matching and Fusion for Reference-Based Hyperspectral Image Super-Resolution | TGRS | 2026 |
| 9 | Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion | TIP | 2026 |
| 10 | HDGMamba: High-Frequency Dynamic Guided Mamba for Robust Multispectral-Hyperspectral Image Fusion | TGRS | 2026 |
| 11 | Leveraging Modal Interaction and Window Dilation in Attention Network for Hyperspectral and Multispectral Remote Sensing Image Fusion | TGRS | 2026 |
| 12 | Local–Global Progressive Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 |
| 13 | MosyMamba: Modal-Synergy Mamba Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 |
| 14 | PMHIF-Net: A Prior-Guided Mamba Hierarchical Interactive Fusion Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 |
| 15 | Region-Aware MoE Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 |
| 16 | S2-Differential Feature Awareness Network for Hyperspectral Image Fusion | TGRS | 2026 |
| 17 | SCALMU: Synthetically Trained Coupling of Adaptive Learned Multiplicative Updates for Hyperspectral-Multispectral Fusion | TGRS | 2026 |
| 18 | Self-Expressive High-Order Tensor Unrolling Network for Unsupervised Hyperspectral and Multispectral Image Fusion | TIP | 2026 |
| 19 | SEMF-Net: A Spatial-Spectral Edge-Enhancement-Based Multistage Fusion Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 |

The Chikusei x4 target, *A Two-Stage Conditional Diffusion Model With Differential Attention* (TIP 2026), also supplies the 16 methods compared in its Table IV (see `results/puformer_chikusei_x4/tip26_table4_comparison.csv`).

## 2. All 244 titled papers

Newest first. `Where` uses the codes above.

| # | Title | Venue | Year | Where |
|---|---|---|---|---|
| 1 | 3SP-DUNet: An Interpretable Deep-Unfolded Network With Spatial–Spectral Sparse Priors for Hyperspectral Pansharpening | TGRS | 2026 | Lc Lr |
| 2 | A Deep Learning–Embedded Sparse Variational Optimization Method for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 3 | A Detail Injection-Based Fusion Framework for Hyperspectral, Multispectral, and Panchromatic Remote Sensing Images | TGRS | 2026 | PDF C S Lc Lr |
| 4 | A Differential Memory Attention Mamba for Spatial-Spectral Representation Learning Toward Hyperspectral Image Classification | IEEE Access | 2026 | Lr |
| 5 | A Lightweight Mamba Network for the Joint Classification of Hyperspectral and LiDAR Data | TGRS | 2026 | Lm |
| 6 | A Multi-Scale Hybrid Attention Residual Network for Super-Resolution Image Reconstruction | IEEE Access | 2026 | Lr |
| 7 | A Novel Panchromatic-Guided Tensor Low-Rank Model for Multispectral Image Sharpening | IEEE Signal Processing Letters | 2026 | Lr |
| 8 | A Stagewise Spectral-Structural Coordinated Reconstruction Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 | PDF S Lc Lr |
| 9 | A Two-Stage Conditional Diffusion Model With Differential Attention for Hyperspectral and Multispectral Image Fusion | TIP | 2026 | PDF S |
| 10 | Adaptive Contourlet-Guided Fuzzy Fusion Network for Joint Classification of Hyperspectral and LiDAR Data | TGRS | 2026 | Lm |
| 11 | Adaptive Neural Operator for Arbitrary-Scale Asteroid Remote Sensing Image Super-Resolution With Unsupervised and Supervised Learning | TGRS | 2026 | Lr |
| 12 | AMHF-Net: A Multispectral and Hyperspectral Image Fusion Network for Arbitrary-Band Hyperspectral Images | JSTARS | 2026 | Lr |
| 13 | An Ultralightweight Multidomain Feature Extraction Network With Cross Spatial–Spectral Attention for Hyperspectral Image Classification | JSTARS | 2026 | Lr |
| 14 | Arbitrary-Scale Fusion Operator for High-Resolution Hyperspectral Imaging | TMM | 2026 | S |
| 15 | AS2G-Transformer: An Adaptive Spatial–Spectral Guided Transformer Network for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 16 | ASSR-Net: Anisotropic Structure-Aware and Spectrally Recalibrated Network for Hyperspectral Image Fusion | TNNLS | 2026 | Lc Lr |
| 17 | Band-Kernel Stochastic Learning for Unsupervised Blind Hyperspectral Image Super-Resolution | TPAMI | 2026 | Lc Lr |
| 18 | Bayesian Fully-Connected Tensor Network for Hyperspectral-Multispectral Image Fusion | TIP | 2026 | Lc Lr |
| 19 | Benchmarking Microscopy Super-Resolution (×4) Deep Learning Models With Point Spread Degradation | IEEE Access | 2026 | Lr |
| 20 | Blind Hyperspectral–Multispectral Image Fusion Robust to Spatial Misregistration and Spectral Range Nonoverlap | JSTARS | 2026 | Lr |
| 21 | BLMRS: A Balanced Learning Method for Multimodal Remote Sensing Image Classification | TGRS | 2026 | Lm |
| 22 | Block Term Decomposition-Guided Frequency Mamba Modulation for Hyperspectral Image Fusion | TGRS | 2026 | PDF C S Lc Lr |
| 23 | Blur-Resistant Hyperspectral Image Super-Resolution via Dual-Degradation Fusion Model | TIP | 2026 | PDF C S Lc Lr |
| 24 | Boosting Multimodal Remote Sensing Image Classification With Transformer-Based Heterogeneously Salient Graph Representation | TGRS | 2026 | Lm |
| 25 | Branch-and-Bound Compressive Learning and Reconstruction of Hyperspectral Data With Deep Tucker Decomposition Using Spatial–Spectral Learning Network | IEEE Access | 2026 | Lr |
| 26 | Bridging Component Learning With Degradation Modelling for Blind Image Super-Resolution | TMM | 2026 | Lr |
| 27 | Butterfly Residual Network: A Hybrid Approach With Spectral Transformers and Depth-Wise Convolutions for Hyperspectral Image Super-Resolution | TNNLS | 2026 | Lc Lr |
| 28 | Cascade Pixel Transformer with Distance-Driven Spatial Fusion for Hyperspectral Image Classification Using Limited Training Samples | IEEE Latin America Transactions | 2026 | Lr |
| 29 | Causal Degradation-Guided Network With Spatial-Frequency Attention for Blind Hyperspectral Image Fusion | TGRS | 2026 | PDF C Lc Lr |
| 30 | Center-Pixel Guided Dual-Path Multihead Attention Fusion Transformer for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 31 | CLCNet: A Curriculum-Guided Contrastive Learning Network With Spectral-Hierarchical Attention for Hyperspectral-LiDAR Image Classification | TGRS | 2026 | Lm |
| 32 | Constrained Conditional Denoising Diffusion for Hyperspectral-Multispectral Fusion | TCI | 2026 | S Lr |
| 33 | Content-Adaptive Unfolding Wavelet Transformer for Hyperspectral Image Super-Resolution | TIP | 2026 | Lc Lr |
| 34 | Context-Reinforced Random-Field Ensembles for Arbitrary-Resolution Hyperspectral Pansharpening | TGRS | 2026 | Lc Lr |
| 35 | Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion | TIP | 2026 | PDF S Lc Lr |
| 36 | Cross-Modal Fusion and Classification of Hyperspectral and Panchromatic Remote Sensing Images With Deep Learning and Multiscale CRFs | TGRS | 2026 | Lm Lr |
| 37 | Cross-Modality Spatial–Spectral Fusion Transformer With Spectral Amplification for VHR Image Classification | TGRS | 2026 | Lm Lr |
| 38 | Cross-Scene Diffusion-Enhanced Uncertainty Attention Network for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 39 | Cross-SPECL: Cross-Scene Hyperspectral Image Classification via Spectral Stability Exploiting and Causal Learning | TGRS | 2026 | Lr |
| 40 | CSS-Mamba: Competitive Sparse-Selection Mamba Network for Hyperspectral–LiDAR Joint Classification | TGRS | 2026 | Lm |
| 41 | CTSNet: Cross-Modal Token Selection Network for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 42 | D3BSR: Blind Super-Resolution via Diffusion-Based Disentangled Degradation Representation | TMM | 2026 | Lr |
| 43 | DBNSR: Hyperspectral Image Super-Resolution Based on Dual-Branch Network | TGRS | 2026 | Lc Lr |
| 44 | DCRF-Net: Dual-Observation Consistency Residual Feedback Network for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 45 | DF-PEM: Dual-Flow Prompt-Expert Mamba for Multimodal Remote Sensing Incremental Classification | TGRS | 2026 | Lm |
| 46 | Diffusion Feature Completion-Driven Textual Dual Distillation for Multimodal Remote Sensing Classification With Missing Modality | TGRS | 2026 | Lm |
| 47 | Diffusion-Driven Mutual Enhancement of Matching and Fusion for Reference-Based Hyperspectral Image Super-Resolution | TGRS | 2026 | PDF C S Lc Lr |
| 48 | DINO-MTP: Topology-Aware Multimodal Fusion of GPS Trajectories and Optical Imagery for Remote Sensing Road Extraction | TGRS | 2026 | Lm |
| 49 | Domain-Adaptive Mamba for Cross-Scene Hyperspectral Image Classification | TIP | 2026 | Lr |
| 50 | Dual-Attention-Enhanced Convolutional-Graph Fusion Network for Hyperspectral Image Classification | GRSL | 2026 | Lr |
| 51 | Dual-branch Selective Multiscale Fusion Network for Hyperspectral and LiDAR Classification | TGRS | 2026 | Lm |
| 52 | DULRNet: A Deep Unfolding Low-Rank Network for Infrared and Visible Image Fusion | IEEE Sensors Journal | 2026 | Lr |
| 53 | Dynamic Fusion of Hyperspectral and LiDAR Data for Target Detection | TGRS | 2026 | Lm |
| 54 | Dynamic Recurrent Self-Refinement Network for Hyperspectral Remote Sensing Image Super-Resolution | TGRS | 2026 | Lc Lr |
| 55 | Edge-Aware Fusion Network for Accurate Hyperspectral Image Classification | IEEE Transactions on Consumer Electronics | 2026 | Lr |
| 56 | Efficient Low Average Rank Tensor Recovery With Implicit Low-Rank Subspace Regularization | TMM | 2026 | Lr |
| 57 | EFSANet: Explicit Frequency Modeling and Style-Aware Alignment for Hyperspectral–Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 58 | Enhancing Leaf Nitrogen Estimation From Satellite Multispectral Imagery via Hyperspectral Super-Resolution and Structural Correction | TGRS | 2026 | Lr |
| 59 | Equivariant High-Resolution Hyperspectral Imaging via Mosaiced and PAN Image Fusion | TIP | 2026 | PDF C Lc Lr |
| 60 | Exploiting Continuity for Unsupervised Single Depth Map Super-Resolution | IEEE Signal Processing Letters | 2026 | Lr |
| 61 | FK-Net: Frequency-Aware and Kernelizable Mamba–Transformer for Multispectral and Hyperspectral Image Fusion | TCYB | 2026 | S |
| 62 | FMSE-DGAF: Frequency-Modulated Spectral Enhancement and Dynamic Global–Local Attention Fusion Network for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 63 | FS-CrossMamba: Dual-Branch Frequency-Guided Spatial and Spectral-Attention-Driven Spectral Cross-Mamba Network for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 64 | Fusion-Driven Task Mutual-Guidance Network for Few-Shot Hyperspectral Image Classification | TMM | 2026 | Lr |
| 65 | Generative Data-Driven Dynamic Deep-Learning Classifier Selection for Hyperspectral Image Classification | JSTARS | 2026 | Lr |
| 66 | Global-Local Dual-Domain Latent Diffusion Network for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 67 | Grouped Attention-Based Spectral–Spatial Feature Learning for Collaborative Hyperspectral and LiDAR Classification | TGRS | 2026 | Lm |
| 68 | GSFL: Graph-Based Spatial–Frequency Learning for Multispectral and Hyperspectral Image Fusion | TGRS | 2026 | S Lc Lr |
| 69 | HCDi-Fusion: Hybrid-Conditioned Diffusion Model for Cross-Resolution Multimodal Image Fusion | TCSVT | 2026 | Lr |
| 70 | HCFMaNet: A Novel Holistic Cross-Modal Fusion Mamba Network for Multi-Modal Medical Image Fusion | TMM | 2026 | Lr |
| 71 | HCMA-Net: Hierarchical Cross-Modality Aggregation Network for Multimodal Remote Sensing Image Classification | TGRS | 2026 | Lm |
| 72 | HDGMamba: High-Frequency Dynamic Guided Mamba for Robust Multispectral-Hyperspectral Image Fusion | TGRS | 2026 | PDF S Lc Lr |
| 73 | HDiff-HIR: Hierarchically Conditional Diffusion Model for Hyperspectral Image Reconstruction | TCSVT | 2026 | Lc Lr |
| 74 | HGTFN: A Hypergraph and Transformer Fusion Network for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 75 | Hierarchical Progressive Transform With Learnable Prior for Hyperspectral Image Super-Resolution | TGRS | 2026 | Lc Lr |
| 76 | HPGC-Diff: Hybrid-Prior Guided Coupled Diffusion for Unsupervised Hyperspectral Image Super-Resolution | TGRS | 2026 | Lc Lr |
| 77 | HSCovMamba: CNN–Mamba Gated-Fusion Network for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 78 | HSI–LiDAR Joint Classification via Progressive Spatial–Spectral–Frequency Fusion Learning | TGRS | 2026 | Lm |
| 79 | Hybrid Hyperspectral Unmixing Using Fusion Mamba And Performer -Attention | IEEE Access | 2026 | Lr |
| 80 | Hyperspectral and Multispectral Image Fusion via Coupled Tensor Wheel Decomposition | TGRS | 2026 | C S Lc Lr |
| 81 | Hyperspectral and Multispectral Image Fusion via Double Decomposition Network | JSTARS | 2026 | Lr |
| 82 | Hyperspectral and Multispectral Image Fusion via Neural Tensor Decomposition With Factor Alignment | IEEE Access | 2026 | Lr |
| 83 | Hyperspectral Image Denoising and Destriping With Adaptive Learnable Subspace Coefficient Tensor Low-Rank Prior | JSTARS | 2026 | Lr |
| 84 | Hyperspectral Image Super-Resolution via Boundary Perception and Topology Inference | TMM | 2026 | Lc Lr |
| 85 | HyPyraMamba: A Pyramid Spectral Attention and Mamba-Based Architecture for Robust Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 86 | Iformer: Irregular Spatial–Spectral Transformer for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 87 | IGECNet: An Interaction-Guided Multimodal Fusion Network With Elevation Constraint for Remote Sensing Semantic Segmentation | TGRS | 2026 | Lm |
| 88 | Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion | TIP | 2026 | Lr |
| 89 | Joint Classification of Hyperspectral and LiDAR Data Using Multiscale Convolution and Cross-Attention-Guided Mamba | TGRS | 2026 | Lm |
| 90 | Joint Degradation-Aware Arbitrary-Scale Super-Resolution for Variable-Rate Extreme Image Compression | IEEE Transactions on Broadcasting | 2026 | Lr |
| 91 | Joint Low-Rank and Smooth Tensor Regularization for Multispectral and Hyperspectral Image Fusion | TGRS | 2026 | Lc Lr |
| 92 | Joint Multiscale Spatial Pyramid and Frequency-Domain Learning for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 93 | KG-Mamba: A Structure-Aware and Knowledge-Guided Mamba Network for Hyperspectral Image Classification | JSTARS | 2026 | Lr |
| 94 | Language-Informed Hyperspectral Image Synthesis for Imbalanced-Small Sample Classification via Semi-Supervised Conditional Diffusion Model | TGRS | 2026 | Lr |
| 95 | Learning a Self-Supervised Low-Rank Decomposition Network for Hyperspectral Image Super-Resolution | TCSVT | 2026 | Lc Lr |
| 96 | Learning Low-Rank Spectral Gradient Subspace With Structure Tensor Model for Guaranteed Hyperspectral Image Destriping and Denoising | TGRS | 2026 | Lr |
| 97 | Leveraging Modal Interaction and Window Dilation in Attention Network for Hyperspectral and Multispectral Remote Sensing Image Fusion | TGRS | 2026 | PDF M C S Lc Lr |
| 98 | Linearized Tensor Low-rank Convolutional Network | IEEE Transactions on Artificial Intelligence | 2026 | Lr |
| 99 | LKA-GFNet: Language Knowledge-Augmented Graph Fusion for Tri-Source Heterogeneous Remote Sensing Data Classification | TGRS | 2026 | Lm |
| 100 | Local–Global Progressive Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 | PDF C S Lc Lr |
| 101 | Low-Rank Gradient Guidance With Mutual-Guided Mamba for Hyperspectral Pansharpening | JSTARS | 2026 | Lr |
| 102 | Low-Rank Representation Learning With Spectral-to-Spatial Aggregation Transformer for Hyperspectral Image Super-Resolution | TGRS | 2026 | Lc Lr |
| 103 | Low-Rank Spectral–Spatial Super-Resolution of Hyperspectral Images Using KAN-Based GAN | TGRS | 2026 | Lc Lr |
| 104 | Low-Rank Tensor Ring Alternating Least Squares With Tikhonov Regularization for Image Completion | IEEE Access | 2026 | Lr |
| 105 | LSLFormer: A Lightweight Spectral–LiDAR Fusion Network for Remote Sensing Image Classification | TGRS | 2026 | Lm |
| 106 | Mamba-MPSE: Multipattern State Evolution Based on the Mamba Model for Intraclass Heterogeneous Wetland Classification With UAV Hyperspectral Imagery | TGRS | 2026 | Lr |
| 107 | MambaFedCD: Spatial–Spectral–Temporal Collaborative Mamba-Based Active Federated Hyperspectral Change Detection | TIP | 2026 | Lr |
| 108 | Mask-DiFuser: A Masked Diffusion Model for Unified Unsupervised Image Fusion | TPAMI | 2026 | Lr |
| 109 | Masked Self-Attention Fusion Network for Joint Classification of Hyperspectral and LiDAR Data | TIP | 2026 | Lm Lr |
| 110 | MCA-Net: A Multiscale Cross-Attention Transformer Network for HSI-MSI Fusion | Sensing and Imaging | 2026 | C |
| 111 | MCF-SSGN: A Multiscale Spectral–Spatial Graph Fusion Network for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 112 | MCIB: Multi-Modal Complementary Information Bottleneck for Hyperspectral and LiDAR Classification | TIP | 2026 | Lm |
| 113 | MD <sup>2</sup> F-Mamba: Multidirectional Depthwise Convolution and Dual-Branch Mamba Feature Fusion Networks for Hyperspectral Image Classification | JSTARS | 2026 | Lr |
| 114 | MGINR-WGAN: A Multiscale Guided Implicit Neural Representation Wasserstein GAN for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 115 | MoEformer: a Frequency-Guided Mixture of Experts Transformer for Hyperspectral and Multispectral Image Fusion | TCSVT | 2026 | S |
| 116 | MosyMamba: Modal-Synergy Mamba Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 | PDF S Lc Lr |
| 117 | MSF-Mamba: Motion-Aware State Fusion Mamba for Efficient Micro-Gesture Recognition | TMM | 2026 | Lr |
| 118 | MSFE-Mamba: Multiscale Frequency-Enhanced Mamba for Hyperspectral Image Classification | JSTARS | 2026 | Lr |
| 119 | Multi-scale Meets Active Learning: A Deep Graph Fusion Paradigm for Hyperspectral Image Classification | TIP | 2026 | Lr |
| 120 | Multi-View Single-Scan Visual State-Space Network for Efficient Image Super-Resolution | TIP | 2026 | Lr |
| 121 | Multimodal Prompt Tuning With Modified Knowledge Distillation for the Joint Classification of Hyperspectral Image and LiDAR Data | TGRS | 2026 | Lm |
| 122 | Multiscale Spectral–Spatial Coordinated Awareness Fusion Network for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 123 | Multiwavelet Based Multibranch Skip Feature Fusion Network for Remote Sensing Image Super Resolution | IEEE Transactions on Emerging Topics in Computational Intelligence | 2026 | Lr |
| 124 | NPFNet: Nonlocal Pixel Clustering and Fourier Enhancement Network for Multispectral and Hyperspectral Image Fusion | GRSL | 2026 | Lr |
| 125 | PASG-Net:Spatial-Guided Frequency Compensation Polarized Attention Fusion Network for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 126 | PLGNet: A Novel Point-Local-Global Multi-level Collaborative Fusion Method for Hyperspectral and Multispectral Remote Sensing Images | IEEE Access | 2026 | Lr |
| 127 | PMFNet: A PCA-Driven Mixture-of-Expert Fusion Network for Hyperspectral Pansharpening | GRSL | 2026 | Lr |
| 128 | PMHIF-Net: A Prior-Guided Mamba Hierarchical Interactive Fusion Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 | PDF S |
| 129 | Positive Matching Benefits Fusion: A Novel Contrastive Learning Framework for Hyperspectral and LiDAR Data Classification | TGRS | 2026 | Lm |
| 130 | PSFD-Net: Progressive Supervision-Based Frequency-Domain Decoupling Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 | C Lc Lr |
| 131 | Reconstruction of Cloud-Covered Land Surface Temperature From Fengyun-4B TIR Remote Sensing Images Using an Improved DTC Model and Multi-Source Data Fusion Approach | TGRS | 2026 | Lm |
| 132 | Region-aware fusion for non-overlapping hyperspectral and multispectral remote sensing images | ISPRS Journal of Photogrammetry and Remote Sensing | 2026 | C |
| 133 | Region-Aware MoE Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 | PDF C S Lc Lr |
| 134 | Rethinking the Upsampling Layer in Hyperspectral Image Super Resolution | TMM | 2026 | Lr |
| 135 | Robust Deep Low-Rank Tensor Factorization for Hyperspectral Image Restoration | TCSVT | 2026 | Lr |
| 136 | Robust Low-Tubal-Rank Tensor Sensing via Preconditioned Subgradient Descent | IEEE Signal Processing Letters | 2026 | Lr |
| 137 | RWKVSR: Receptance Weighted Key-Value Network for Hyperspectral Image Super-Resolution | TCSVT | 2026 | Lc Lr |
| 138 | S$^3$RNet: Sparse Spatial–Spectral Representation With Hybrid Knowledge Distillation for Efficient Multispectral and Hyperspectral Image Fusion | JSTARS | 2026 | Lr |
| 139 | S2-Differential Feature Awareness Network for Hyperspectral Image Fusion | TGRS | 2026 | PDF C Lc Lr |
| 140 | S2CaT: Spatial–Spectral-Based CNN and Transformer Network for Hyperspectral Image Classification | IEEE Access | 2026 | Lr |
| 141 | S2CIFTNet: Spatial–Spectral Coupling and Interactive Fusion Transformer Network for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 142 | S2CMamba: Structure-to-Contrast Semantic Mamba for Hyperspectral and LiDAR Joint Classification | TGRS | 2026 | Lm |
| 143 | S2TA-Fuse: Semantic-Superpixel Tokenized Attention for Spatial–Spectral Fusion | TGRS | 2026 | S |
| 144 | SafaSR: An Arbitrary-Scale Image Super-Resolution Network Based on Multidomain Feature Fusion for Enhancing Diverse IoT Vision | IEEE Internet of Things Journal | 2026 | Lr |
| 145 | SAMamba: State Attention Mamba for Hyperspectral Image Classification | TGRS | 2026 | Lr |
| 146 | SCALMU: Synthetically Trained Coupling of Adaptive Learned Multiplicative Updates for Hyperspectral-Multispectral Fusion | TGRS | 2026 | PDF C S Lc Lr |
| 147 | SCIAU-Net: A Spatial-Spectral Cross-Modal Interaction ADMM Unfolding Network for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 148 | SDG-DSB: Spectral Degradation Guided Diffusion Schrödinger Bridge for Hyperspectral Images Super-Resolution | TGRS | 2026 | Lc Lr |
| 149 | Self-Expressive High-Order Tensor Unrolling Network for Unsupervised Hyperspectral and Multispectral Image Fusion | TIP | 2026 | PDF C Lc Lr |
| 150 | SEMF-Net: A Spatial-Spectral Edge-Enhancement-Based Multistage Fusion Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 | PDF C Lc Lr |
| 151 | SFCFNet: A Spatial–Frequency Cross-Attention Fusion Network for Hyperspectral Image Classification | JSTARS | 2026 | Lr |
| 152 | SFVEGNet: Spatial-Frequency Vision Mamba Enhanced Edge-Gaussian Cooperative Fusion Network for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | C Lr |
| 153 | Shuffle Mamba: State Space Models With Random Shuffle for Multi-Modal Image Fusion | TCSVT | 2026 | Lr |
| 154 | SMO-Net: Unsupervised Hyperspectral Image Super-Resolution via Spectral Mapping and Overall Consistency Optimization | GRSL | 2026 | Lr |
| 155 | Spatial-Frequency Joint Learning Mamba for Hyperspectral Image Classification | GRSL | 2026 | Lr |
| 156 | Spatial-Spectral Synergy Learning in the Frequency Domain for Hyperspectral Image Super-Resolution | IEEE Access | 2026 | Lr |
| 157 | Spectral Dynamic Attention Network for Hyperspectral Image Super-Resolution | GRSL | 2026 | Lr |
| 158 | Spectral State Fusion Tree Mamba for Hyperspectral Image Classification | TIP | 2026 | Lr |
| 159 | Spectrally–Spatially Coupled Dynamic Reduction for Efficient Hyperspectral Image Super-Resolution | TGRS | 2026 | Lc Lr |
| 160 | SSA-Mamba: Spatial-Spectral Attentive State Space Model for Hyperspectral Image Classification | JSTARS | 2026 | Lr |
| 161 | SSMFMamba: A Spatial-Spectral Mamba-Convolution Fusion Network for Hyperspectral Image Classification | JSTARS | 2026 | Lr |
| 162 | STG-Net: Shared Tensor Guidance HSI-MSI Fusion Under Locally Overlapping Condition | JSTARS | 2026 | C Lr |
| 163 | Structure-Preserved Conditional Diffusion for Hyperspectral LiDAR Fusion via Cross-Modal Alignment | IEEE Access | 2026 | Lr |
| 164 | Subpixel-Aware Hyperspectral Image Classification Using ETAP-Guided Autoencoder and Vision Transformer Fusion | IEEE Access | 2026 | Lr |
| 165 | Synthetic Abundance Maps for Unsupervised Super-Resolution of Hyperspectral Remote Sensing Images | JSTARS | 2026 | Lr |
| 166 | TCNet: Topological Consistency Network for Hyperspectral and Multispectral Image Fusion | GRSL | 2026 | Lr |
| 167 | TDP-Net: Unsupervised HSI-MSI Fusion via Tucker Decomposition With Generative Diffusion Priors | TGRS | 2026 | S |
| 168 | Tensor Low-Rank Orthogonal Compression for Convolutional Neural Networks | IEEE/CAA Journal of Automatica Sinica | 2026 | Lr |
| 169 | Tensor Wheel Completion With Low-Rank Factor Prior and Adaptive Graph Regularizer for Hyperspectral Image Recovery | TMM | 2026 | Lr |
| 170 | TETNet: A Target-Enhanced Triple-Branch Network With Spectral–Spatial Attention for Hyperspectral Image Classification | IEEE Access | 2026 | Lr |
| 171 | Three-Branch Multiscale Abundance Feature Fusion Network for Hyperspectral Image Change Detection | JSTARS | 2026 | Lr |
| 172 | TM-MOE: A Degradation-Aware Transformer-Mamba MoE for Hyperspectral and Multispectral Image Fusion | TGRS | 2026 | S Lc Lr |
| 173 | Uncertainty-Based Dendritic Model for Multimodal Remote Sensing Data Classification | TGRS | 2026 | Lm |
| 174 | Uncertainty-Driven Generative Prior Learning for Sparse Model-Guided Hyperspectral Image Fusion | TIP | 2026 | Lc Lr |
| 175 | Unfolding Degradation-Aware Transformer for Low-Light Hyperspectral Image Super-Resolution | TCI | 2026 | Lr |
| 176 | Unsupervised Deformable Bilinear Fusion Network for Unregistered Hyperspectral Image Super-resolution | TGRS | 2026 | S Lc Lr |
| 177 | Unsupervised Frequency-Guided Network for Hyperspectral and Multispectral Image Fusion | JSTARS | 2026 | Lr |
| 178 | Unsupervised Super Spatial Resolution Brillouin Frequency Shift Extraction Based on Physical Enhanced Neural Network | Journal of Lightwave Technology | 2026 | Lr |
| 179 | Variational Model-Guided Three-Dimensional Recurrent Transformer for Hyperspectral and Multispectral Image Fusion | IEEE Access | 2026 | Lr |
| 180 | Visual Differential-Spatially Projected Transformer for Efficient Hyperspectral Images Super-Resolution | TGRS | 2026 | Lc Lr |
| 181 | WAS-Mamba: 3D Medical Image Segmentation via Windowed Attention State Space Model | TIP | 2026 | Lr |
| 182 | Wavelet-Enhanced Spatial–Spectral Prior Injection Mamba Network for Hyperspectral Image Super-Resolution | TGRS | 2026 | Lc Lr |
| 183 | WIMNet: A Whole-Image-Based Mamba Network for Joint Hyperspectral and LiDAR Classification | TGRS | 2026 | Lm |
| 184 | A Progressive Registration-Fusion Co-Optimization A-Mamba Network: Toward Deep Unregistered Hyperspectral and Multispectral Fusion | TGRS | 2025 | S |
| 185 | A Progressive Spatial–Spectral Interactive Network for Integrated Fusion of Panchromatic, Multispectral, and Hyperspectral Images | TGRS | 2025 | S |
| 186 | Adaptive Expert Learning for Hyperspectral and Multispectral Image Fusion | TGRS | 2025 | C S |
| 187 | Advancing Hyperspectral and Multispectral Image Fusion: An Information-Aware Transformer-Based Unfolding Network | TNNLS | 2025 | S |
| 188 | AEWFNet: Adaptive Enhancement and Wavelet Convolution for Hyperspectral and Multispectral Image Fusion | TGRS | 2025 | S |
| 189 | An Asymptotic Multiscale Symmetric Fusion Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2025 | M S |
| 190 | An Efficient Image Fusion Network Exploiting Unifying Language and Mask Guidance | TPAMI | 2025 | S |
| 191 | Arbitrary-Resolution Hyperspectral Pansharpening Neural Operators | TGRS | 2025 | M |
| 192 | Building Non-Uniform Degradation Model for Position-Aware Hyperspectral Image Fusion | TPAMI | 2025 | S |
| 193 | CESFusion: Cross-Frequency Enhanced Spatial—Spectral Fusion Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2025 | S |
| 194 | Cyclic Cross-Modality Interaction for Hyperspectral and Multispectral Image Fusion | TCSVT | 2025 | C S |
| 195 | Dilated Transformation-Guided Unsupervised Multimodal Learning for Hyperspectral and Multispectral Image Fusion | TGRS | 2025 | S |
| 196 | GFHMP: Gradual Fusion Framework of Hyperspectral, Multispectral, and Panchromatic Images Using a Novel Spatial–Spectral Cross-Fusion Network | TGRS | 2025 | S |
| 197 | Hyperspectral and Multispectral Image Fusion With Functional Data Analysis Techniques | TGRS | 2025 | C S |
| 198 | Hyperspectral Pansharpening via Diffusion Models with Iteratively Zero-Shot Guidance | CVPR | 2025 | C |
| 199 | Integrated Fusion for Panchromatic, Multispectral, Hyperspectral Remote Sensing Images: Insights From Multispectral Images | TGRS | 2025 | S |
| 200 | IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images | TIP | 2025 | M C |
| 201 | Mamba Collaborative Implicit Neural Representation for Hyperspectral and Multispectral Remote Sensing Image Fusion | TGRS | 2025 | S |
| 202 | Multispectral-Hyperspectral Image Fusion via Similarity-Guided Graph Attention and VAE-Transformer | TGRS | 2025 | M |
| 203 | Progressive Synergistic Registration and Fusion Diffusion Network for Unregistered Hyperspectral and Multispectral Image Fusion | TGRS | 2025 | S |
| 204 | RAMSF: A Novel Generic Framework for Optical Remote Sensing Multimodal Spatial-Spectral Fusion | TGRS | 2025 | S |
| 205 | Self-Learning Hyperspectral and Multispectral Image Fusion via Adaptive Residual Guided Subspace Diffusion Model | TPAMI | 2025 | S |
| 206 | Spatial–Spectral Cross Mamba Network for Hyperspectral and Multispectral Image Fusion | TGRS | 2025 | S |
| 207 | SSDT: Multiscale Spatial–Spectral Dilated Transformer for Hyperspectral and Multispectral Image Fusion | TGRS | 2025 | S |
| 208 | Unsupervised Hyperspectral and Multispectral Image Blind Fusion Based on Deep Tucker Decomposition Network With Spatial-Spectral Manifold Learning | TNNLS | 2025 | M C S |
| 209 | Unsupervised Model-Embedded Two-Stage Diffusion Method for Multispectral and Hyperspectral Image Fusion | TGRS | 2025 | S |
| 210 | Unsupervised Range-Nullspace Learning Prior for Multispectral Images Reconstruction | TIP | 2025 | M |
| 211 | VDMUFusion: A Versatile Diffusion Model-Based Unsupervised Framework for Image Fusion | TIP | 2025 | M |
| 212 | Zero-Shot Hyperspectral Pansharpening Using Hysteresis-Based Tuning for Spectral Quality Control | TGRS | 2025 | C |
| 213 | A Coupled Tensor Double-Factor Method for Hyperspectral and Multispectral Image Fusion | TGRS | 2024 | C S |
| 214 | A Principle Design of Registration-Fusion Consistency: Toward Interpretable Deep Unregistered Hyperspectral Image Fusion | TNNLS | 2024 | M |
| 215 | A Self-Supervised Spaceborne Multispectral and Hyperspectral Image Fusion Unrolling Network | TGRS | 2024 | S |
| 216 | All in One: A Unified Network for Hyperspectral Image Fusion | TGRS | 2024 | S |
| 217 | CODE-IF: A Convex/Deep Image Fusion Algorithm for Efficient Hyperspectral Super-Resolution | TGRS | 2024 | S |
| 218 | CS2DIPs: Unsupervised HSI Super-Resolution Using Coupled Spatial and Spectral Deep Image Priors | TIP | 2024 | M |
| 219 | Deep Unfolding Network Enhanced by Transformer Priors for Unregistered Hyperspectral and Multispectral Image Fusion | TGRS | 2024 | M S |
| 220 | Domain Transform Model Driven by Deep Learning for Anti-Noise Hyperspectral and Multispectral Image Fusion | TGRS | 2024 | S |
| 221 | Exploring the Spectral Prior for Hyperspectral Image Super-Resolution | TIP | 2024 | M |
| 222 | FusionMamba: Efficient Remote Sensing Image Fusion With State Space Model | TGRS | 2024 | C |
| 223 | INF3: Implicit Neural Feature Fusion Function for Multispectral and Hyperspectral Image Fusion | TCI | 2024 | S |
| 224 | Interpretable Model-Driven Deep Network for Hyperspectral, Multispectral, and Panchromatic Image Fusion | TNNLS | 2024 | M S |
| 225 | LGCT: Local-Global Collaborative Transformer for Fusion of Hyperspectral and Multispectral Images | TGRS | 2024 | S |
| 226 | MIMO-SST: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion | TGRS | 2024 | C S |
| 227 | MMIF: Interpretable Hyperspectral and Multispectral Image Fusion via Maximum Mutual Information | TGRS | 2024 | S |
| 228 | Progressive Multi-Iteration Registration-Fusion Co-Optimization Network for Unregistered Hyperspectral Image Super-Resolution | TGRS | 2024 | S |
| 229 | Spectral Super-Resolution via Model-Guided Cross-Fusion Network | TNNLS | 2024 | M |
| 230 | Unsupervised Deep Tensor Network for Hyperspectral–Multispectral Image Fusion | TNNLS | 2024 | S |
| 231 | Unsupervised Hybrid Network of Transformer and CNN for Blind Hyperspectral and Multispectral Image Fusion | TGRS | 2024 | S |
| 232 | Decoupled-and-Coupled Networks: Self-Supervised Hyperspectral Image Super-Resolution With Subpixel Fusion | TGRS | 2023 | S |
| 233 | Deep Hyperspectral and Multispectral Image Fusion With Inter-Image Variability | TGRS | 2023 | M |
| 234 | Dual Spatial–Spectral Pyramid Network With Transformer for Hyperspectral Image Fusion | TGRS | 2023 | S |
| 235 | GuidedNet: A General CNN Fusion Framework via High-Resolution Guidance for Hyperspectral Image Super-Resolution | TCYB | 2023 | M |
| 236 | MGFEI-Net: Multiscale Grouping Feedback Embedded Integrated Network for Panchromatic, Multispectral, and Hyperspectral Image Fusion | TGRS | 2023 | S |
| 237 | PSRT: Pyramid Shuffle-and-Reshuffle Transformer for Multispectral and Hyperspectral Image Fusion | TGRS | 2023 | S |
| 238 | Stereo Cross-Attention Network for Unregistered Hyperspectral and Multispectral Image Fusion | TGRS | 2023 | M S |
| 239 | MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion | TPAMI | 2022 | M |
| 240 | Model-Guided Deep Hyperspectral Image Super-Resolution | TIP | 2021 | M |
| 241 | An Integrated Approach to Registration and Fusion of Hyperspectral and Multispectral Images | TGRS | 2020 | M |
| 242 | A Convex Formulation for Hyperspectral Image Superresolution via Subspace-Based Regularization | TGRS | 2015 | M |
| 243 | Hyperspectral and Multispectral Image Fusion Based on a Sparse Representation | TGRS | 2015 | M |
| 244 | Coupled Nonnegative Matrix Factorization Unmixing for Hyperspectral and Multispectral Data Fusion | TGRS | 2012 | M |

## 3. Papers cited only by name (64)

Referenced in docs or code without a full title anywhere in the repository.

| # | Name | Paper / venue as cited | Where cited |
|---|---|---|---|
| 1 | FUSE | Wei et al., fast fusion via a Sylvester equation, TIP 2015 | TIP'26 Table IV |
| 2 | GSA | Aiazzi et al., TGRS 2007 | PUFormer baseline, manuscript references |
| 3 | U2Net | HSI-MSI fusion, 2023 | TIP'26 Table IV, SOTA_COMPARISON |
| 4 | DCT (DCTransformer) | HSI-MSI fusion transformer | TIP'26 Table IV |
| 5 | OTPNet | ODE-inspired tuning-free fusion | TIP'26 Table IV |
| 6 | DPFormer | HSI-MSI fusion transformer | TIP'26 Table IV |
| 7 | MIMFormer | HSI-MSI fusion transformer | TIP'26 Table IV |
| 8 | DDIF | Information Fusion 2024 (Dif-PAN code) | TIP'26 Table IV |
| 9 | ASMNet | HSI-MSI fusion | TIP'26 Table IV |
| 10 | SMGU-Net | Pattern Recognition 2025 | TIP'26 Table IV, README |
| 11 | CLSNet | 2026 (paper not located) | TIP'26 Table IV |
| 12 | CoFusion | arXiv 2026 | SOTA_COMPARISON, manuscript references |
| 13 | SSA | sensor-agnostic arbitrary-scale fusion, 2026 | SOTA_COMPARISON, manuscript references |
| 14 | BFCTN | Bayesian tensor fusion, arXiv 2025 | SOTA_COMPARISON |
| 15 | GTNN | tensor nuclear norm fusion, TNNLS 2025 | SOTA_COMPARISON |
| 16 | Fusformer | transformer HSI fusion, 2021 | manuscript references, DAETF |
| 17 | DCFormer | HSI fusion transformer | SOTA_COMPARISON |
| 18 | PIF-Net | Mamba fusion, arXiv | Chikusei survey (excluded as preprint), SOTA_COMPARISON |
| 19 | TPTransformer | tensor-product transformer | SOTA_COMPARISON |
| 20 | FeINFN | Fourier-enhanced implicit neural fusion network, 2024 | THEORY_AND_RESULTS, SOTA_COMPARISON |
| 21 | OTIAS | INR arbitrary-scale fusion | SOTA_COMPARISON |
| 22 | NeSSR | neural implicit spectral representation, IJCV 2024/25 | ContinuumFusion prior work |
| 23 | DDPM-Fus | diffusion fusion | SOTA_COMPARISON |
| 24 | KANDiff | KAN diffusion fusion | SOTA_COMPARISON |
| 25 | Selective Re-learning | zero-shot HSI fusion, CVPR 2025 | manuscript references, SOTA_COMPARISON |
| 26 | HyDeFuse | arXiv | Chikusei survey (excluded as preprint) |
| 27 | HyCoNet | unsupervised fusion (runner-up in CDPS) | Chikusei survey |
| 28 | BDT | Bidirectional Dilation Transformer, IJCAI 2023 | THEORY_AND_RESULTS |
| 29 | 3DT-Net | 3D-CNN + Transformer prior, Information Fusion 2023 | THEORY_AND_RESULTS |
| 30 | DHIF-Net | Zheng et al., TGRS 2021 | THEORY_AND_RESULTS, DAETF |
| 31 | IFCASformer | HSI fusion transformer | manuscript references, DAETF |
| 32 | LRU | HSI fusion network | manuscript references, DAETF |
| 33 | UTAL | HSI fusion network | manuscript references, DAETF |
| 34 | DBIN | deep blind iterative network | manuscript references, DAETF |
| 35 | AMGSGAN | GAN-based HSI fusion | DAETF, tools |
| 36 | TSFN | two-stream fusion network | DAETF, tools |
| 37 | SSR-NET | Zhang et al., TGRS 2021 | PUFormer anchor model |
| 38 | MHF-Net (CVPR version) | Xie et al., CVPR 2019 (the TPAMI 2022 version is in section 2) | DAETF |
| 39 | HyperFuse | Dian et al., IJCV 2023 | DAETF |
| 40 | CLoRF | arXiv 2405.17818 | ContinuumFusion |
| 41 | SINR | spectral implicit neural representation, 2023 | ContinuumFusion, positioning |
| 42 | INR-HSISR | INR for HSI super-resolution, TGRS 2022 | ContinuumFusion, positioning |
| 43 | PGU-Net | arXiv 2606.05759 | ContinuumFusion, positioning |
| 44 | Restormer | Zamir et al., CVPR 2022 | PUFormer prior architecture |
| 45 | SIREN | sinusoidal representation networks, 2020 | ContinuumFusion |
| 46 | DDNM | null-space diffusion, ICLR 2023 | SpectralFlow |
| 47 | Consistency models | Song et al., 2023 | ConsistentFlow |
| 48 | Wald | data-fusion definitions / Wald protocol | manuscript references, docs |
| 49 | Gavish & Donoho | optimal SVD hard threshold, IEEE TIT 2014 | KrylovNet, manuscript |
| 50 | Kantorovich & Rubinshtein | EMD / Mallows distance | manuscript references |
| 51 | Horn & Johnson | Matrix Analysis (book) | manuscript references |
| 52 | Geman | largest singular value of random matrices, 1980 | manuscript appendix |
| 53 | Vershynin | random-matrix concentration, 2012 | manuscript appendix |
| 54 | Bioucas-Dias & Nascimento | HySime subspace estimation, 2008 | KrylovNet, positioning |
| 55 | Bioucas-Dias et al. | hyperspectral unmixing overview, 2012 | ZeroFusion |
| 56 | Chang | hyperspectral data exploitation (book), 2003 | DAETF, ZeroFusion |
| 57 | Chang & Du | virtual dimensionality, 2004 | KrylovNet, ZeroFusion |
| 58 | Zhang et al. | intrinsic-rank estimation, 2020 | KrylovNet, positioning |
| 59 | Ruitenbeek et al. | range-space reconstruction, 2007 | DAETF |
| 60 | Foucart & Rauhut | compressive sensing (book), 2013 | DAETF |
| 61 | Donoho & Tanner | compressed-sensing phase transitions, 2005 | ZeroFusion, positioning |
| 62 | Blau & Michaeli | perception-distortion tradeoff, 2018 | DAETF |
| 63 | Yokoya & Iwasaki | Chikusei dataset technical report, 2016 | README |
| 64 | Vivone et al. | pansharpening toolbox (Q2n and SCC reference code) | PUFormer metrics |

## 4. Datasets (18)

CAVE Multispectral Image Database, Harvard Hyperspectral Image Database, Chikusei Hyperspectral Dataset, PanCollection: Benchmark Datasets for Pansharpening, HyperPanCollection: Benchmark for Hyperspectral Pansharpening, AVIRIS Hyperspectral Data (Pre-processed for CSAKD/KD studies), Hyperspectral Pansharpening Toolbox (PRISMA Benchmark), Houston Hyperspectral Dataset (IEEE GRSS Data Fusion Contest), WorldView-3 Dataset (via PanCollection), ICVL Hyperspectral Dataset, ARAD_1K / NTIRE 2022 Spectral Reconstruction Dataset, LN01 Hyperspectral Dataset, Xiong'an Hyperspectral Dataset, PRISMA-Paris Hyperspectral Dataset, Ziyuan-1 02D Hyperspectral Dataset, Urban Hyperspectral Dataset, XDU-Liyukou Hyperspectral Dataset, Washington DC Mall (WDCM) Hyperspectral Dataset.
