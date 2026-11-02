# The 17 IEEE papers most relevant to ExactFuse

The papers the ExactFuse work depends on most directly, restricted to IEEE journals and magazines whose articles are not open access (they are read through an IEEE Xplore subscription; the Crossref licence of every article was checked on 2 Nov 2026, and none carries a Creative Commons licence). MIMFormer (IEEE JSTARS, an open-access journal) and the non-IEEE baselines are therefore not in this list, although the paper retrains and cites them. All 17 are cited in the journal version ([`paper/puformer/main.pdf`](../paper/puformer/main.pdf)); the bibliographic data and DOIs are the Crossref records in [`paper/puformer/references.bib`](../paper/puformer/references.bib), and the full catalogue is in [`REFERENCES.md`](REFERENCES.md).

| # | Paper | Venue | Role for ExactFuse |
|---|---|---|---|
| 1 | A Two-Stage Conditional Diffusion Model With Differential Attention for Hyperspectral and Multispectral Image Fusion | IEEE TIP 2026 | Defines the Chikusei ×4 protocol ExactFuse follows (WorldView-2 response, 7×7 Gaussian blur with σ = 2, split sizes) and the 17-method table quoted for context |
| 2 | MIMO-SST: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion | IEEE TGRS 2024 | Retrained baseline |
| 3 | PSRT: Pyramid Shuffle-and-Reshuffle Transformer for Multispectral and Hyperspectral Image Fusion | IEEE TGRS 2023 | Retrained baseline, also with its authors' own recipe to test the effect of the pipeline |
| 4 | SSR-NET: Spatial–Spectral Reconstruction Network for Hyperspectral and Multispectral Image Fusion | IEEE TGRS 2021 | Retrained baseline on every dataset and scale factor |
| 5 | Deep Hyperspectral Image Fusion Network With Iterative Spatio-Spectral Regularization | IEEE TCI 2022 | Retrained unfolding baseline whose blur, decimation and response are learned convolution layers: the learned-operator design that the exact operators replace |
| 6 | MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion | IEEE TPAMI 2022 | Unfolds a fusion model with operators learned as convolutions |
| 7 | Model-Guided Deep Hyperspectral Image Super-Resolution | IEEE TIP 2021 | Model-guided fusion network with data-consistency steps, the family ExactFuse belongs to |
| 8 | Interpretable Model-Driven Deep Network for Hyperspectral, Multispectral, and Panchromatic Image Fusion | IEEE TNNLS 2024 | Interpretable model-driven unfolding for hyperspectral, multispectral and panchromatic fusion |
| 9 | Advancing Hyperspectral and Multispectral Image Fusion: An Information-Aware Transformer-Based Unfolding Network | IEEE TNNLS 2025 | Transformer-based unfolding for hyperspectral–multispectral fusion |
| 10 | Deep Unfolding Network Enhanced by Transformer Priors for Unregistered Hyperspectral and Multispectral Image Fusion | IEEE TGRS 2024 | Unfolding with transformer priors for unregistered inputs |
| 11 | IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images | IEEE TIP 2025 | Unfolded fusion of unregistered images at arbitrary resolution with a learned registration |
| 12 | Fast Fusion of Multi-Band Images Based on Solving a Sylvester Equation | IEEE TIP 2015 | Closed-form fusion step through a Sylvester equation under periodic borders |
| 13 | A Convex Formulation for Hyperspectral Image Superresolution via Subspace-Based Regularization | IEEE TGRS 2015 | Estimates the spatial and spectral responses from D·Y_M = R·Y_H |
| 14 | Hyperspectral Super-Resolution: A Coupled Tensor Factorization Approach | IEEE TSP 2018 | Separable (Kronecker) degradation operators for recoverable tensor models |
| 15 | Deconvolving Images With Unknown Boundaries Using the Alternating Direction Method of Multipliers | IEEE TIP 2013 | The border rule of real images is unknown |
| 16 | Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion | IEEE TIP 2026 | Unsupervised fusion by diffusion posterior sampling with the observation operators |
| 17 | Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing | IEEE SPM 2021 | Survey of algorithm unrolling, the design principle of ExactFuse's stages |

## A. Benchmark protocol

**1. Y. Chen, R. Liu, W. K. Wong, J. Wen, "A Two-Stage Conditional Diffusion Model With Differential Attention for Hyperspectral and Multispectral Image Fusion," *IEEE Transactions on Image Processing*, vol. 35, pp. 8980–8995, 2026.** [doi:10.1109/tip.2026.3723234](https://doi.org/10.1109/tip.2026.3723234)
- Defines the Chikusei ×4 protocol ExactFuse follows (WorldView-2 response, 7×7 Gaussian blur with σ = 2, split sizes) and the 17-method table quoted for context; no public code, so it could not be retrained.

## B. Published networks retrained under the equal-budget protocol

**2. J. Fang, J. Yang, A. Khader, L. Xiao, "MIMO-SST: Multi-Input Multi-Output Spatial-Spectral Transformer for Hyperspectral and Multispectral Image Fusion," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 62, pp. 1–20, 2024.** [doi:10.1109/tgrs.2024.3361553](https://doi.org/10.1109/tgrs.2024.3361553)
- Retrained baseline; the strongest one at ×4 on Chikusei and the one with the smallest error in the observed subspace, against which the 27× / 9× ratio of the null-space analysis is measured.

**3. S. Q. Deng, L. J. Deng, X. Wu, R. Ran, D. Hong, G. Vivone, "PSRT: Pyramid Shuffle-and-Reshuffle Transformer for Multispectral and Hyperspectral Image Fusion," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 61, pp. 1–15, 2023.** [doi:10.1109/tgrs.2023.3244750](https://doi.org/10.1109/tgrs.2023.3244750)
- Retrained baseline, also with its authors' own recipe to test the effect of the pipeline.

**4. X. Zhang, W. Huang, Q. Wang, X. Li, "SSR-NET: Spatial–Spectral Reconstruction Network for Hyperspectral and Multispectral Image Fusion," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 59, pp. 5953–5965, 2021.** [doi:10.1109/tgrs.2020.3018732](https://doi.org/10.1109/tgrs.2020.3018732)
- Retrained baseline on every dataset and scale factor.

**5. T. Huang, W. Dong, J. Wu, L. Li, X. Li, G. Shi, "Deep Hyperspectral Image Fusion Network With Iterative Spatio-Spectral Regularization," *IEEE Transactions on Computational Imaging*, vol. 8, pp. 201–214, 2022.** [doi:10.1109/tci.2022.3152700](https://doi.org/10.1109/tci.2022.3152700)
- Retrained unfolding baseline whose blur, decimation and response are learned convolution layers: the learned-operator design that the exact operators replace.

## C. Model-based and unfolded fusion networks

**6. Q. Xie, M. Zhou, Q. Zhao, Z. Xu, D. Meng, "MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, pp. 1457–1473, 2022.** [doi:10.1109/tpami.2020.3015691](https://doi.org/10.1109/tpami.2020.3015691)
- Unfolds a fusion model with operators learned as convolutions; reproduced by the learned-operator row of the ablation.

**7. W. Dong, C. Zhou, F. Wu, J. Wu, G. Shi, X. Li, "Model-Guided Deep Hyperspectral Image Super-Resolution," *IEEE Transactions on Image Processing*, vol. 30, pp. 5754–5768, 2021.** [doi:10.1109/tip.2021.3078058](https://doi.org/10.1109/tip.2021.3078058)
- Model-guided fusion network with data-consistency steps, the family ExactFuse belongs to.

**8. X. Tian, K. Li, W. Zhang, Z. Wang, J. Ma, "Interpretable Model-Driven Deep Network for Hyperspectral, Multispectral, and Panchromatic Image Fusion," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 35, pp. 14382–14395, 2024.** [doi:10.1109/tnnls.2023.3278928](https://doi.org/10.1109/tnnls.2023.3278928)
- Interpretable model-driven unfolding for hyperspectral, multispectral and panchromatic fusion; learned operators, no border rule.

**9. J. Sun, B. Chen, R. Lu, Z. Cheng, C. Qu, X. Yuan, "Advancing Hyperspectral and Multispectral Image Fusion: An Information-Aware Transformer-Based Unfolding Network," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 36, pp. 7407–7421, 2025.** [doi:10.1109/tnnls.2024.3400809](https://doi.org/10.1109/tnnls.2024.3400809)
- Transformer-based unfolding for hyperspectral–multispectral fusion; one of the reasons the transformer prior is not claimed as new.

## D. Registration and arbitrary resolution

**10. J. Fang, J. Yang, A. Khader, L. Xiao, "Deep Unfolding Network Enhanced by Transformer Priors for Unregistered Hyperspectral and Multispectral Image Fusion," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 62, pp. 1–16, 2024.** [doi:10.1109/tgrs.2024.3460186](https://doi.org/10.1109/tgrs.2024.3460186)
- Unfolding with transformer priors for unregistered inputs; ExactFuse instead estimates a global offset from the pair and registers the HR-MSI before fusion.

**11. J. Qu, X. Wu, W. Dong, J. Cui, Y. Li, "IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images," *IEEE Transactions on Image Processing*, vol. 34, pp. 1934–1949, 2025.** [doi:10.1109/TIP.2025.3551531](https://doi.org/10.1109/TIP.2025.3551531)
- Unfolded fusion of unregistered images at arbitrary resolution with a learned registration; related to the registration experiments and to the one network trained over ×4 and ×8.

## E. Operators, closed forms and identifiability

**12. Qi Wei, N. Dobigeon, J. Y. Tourneret, "Fast Fusion of Multi-Band Images Based on Solving a Sylvester Equation," *IEEE Transactions on Image Processing*, vol. 24, pp. 4109–4121, 2015.** [doi:10.1109/tip.2015.2458572](https://doi.org/10.1109/tip.2015.2458572)
- Closed-form fusion step through a Sylvester equation under periodic borders; Theorem 1 gives the closed form for the reflect border of the simulation.

**13. M. Simoes, J. Bioucas-Dias, L. B. Almeida, J. Chanussot, "A Convex Formulation for Hyperspectral Image Superresolution via Subspace-Based Regularization," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 53, pp. 3373–3388, 2015.** [doi:10.1109/tgrs.2014.2375320](https://doi.org/10.1109/tgrs.2014.2375320)
- Estimates the spatial and spectral responses from D·Y_M = R·Y_H; the identifiability theorem states when this relation fixes the blur, shift and grid offsets.

**14. C. I. Kanatsoulis, X. Fu, N. D. Sidiropoulos, W. K. Ma, "Hyperspectral Super-Resolution: A Coupled Tensor Factorization Approach," *IEEE Transactions on Signal Processing*, vol. 66, pp. 6503–6517, 2018.** [doi:10.1109/tsp.2018.2876362](https://doi.org/10.1109/tsp.2018.2876362)
- Separable (Kronecker) degradation operators for recoverable tensor models; ExactFuse uses the same separability for its exact data step and the fusion null space.

**15. M. S. C. Almeida, M. A. T. Figueiredo, "Deconvolving Images With Unknown Boundaries Using the Alternating Direction Method of Multipliers," *IEEE Transactions on Image Processing*, vol. 22, pp. 3074–3086, 2013.** [doi:10.1109/tip.2013.2258354](https://doi.org/10.1109/tip.2013.2258354)
- The border rule of real images is unknown; the border corollary bounds what a fixed border rule can cost.

## F. Restoration with known operators and unrolling

**16. Y. Xu, J. Zhu, D. Hong, Z. Wei, Z. Wu, "Coupled Diffusion Posterior Sampling for Unsupervised Hyperspectral and Multispectral Images Fusion," *IEEE Transactions on Image Processing*, vol. 35, pp. 69–84, 2026.** [doi:10.1109/tip.2025.3647207](https://doi.org/10.1109/tip.2025.3647207)
- Unsupervised fusion by diffusion posterior sampling with the observation operators; an alternative way to use known operators, without training on pairs.

**17. V. Monga, Y. Li, Y. C. Eldar, "Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing," *IEEE Signal Processing Magazine*, vol. 38, pp. 18–44, 2021.** [doi:10.1109/msp.2020.3016905](https://doi.org/10.1109/msp.2020.3016905)
- Survey of algorithm unrolling, the design principle of ExactFuse's stages.
