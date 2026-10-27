# The 17 papers most relevant to this project

The papers this project depends on most directly: the benchmark protocol it follows, the five published
networks it retrains under one pipeline, the closest unfolding and operator-aware methods, the theory its
contributions build on or are compared with, and the building block of its prior. All bibliographic data and DOIs
are taken from the Crossref records used in [`paper/puformer/references.bib`](../paper/puformer/references.bib);
the full literature catalogue (308 papers) is in [`REFERENCES.md`](REFERENCES.md).

| # | Paper | Venue | Role in this project |
|---|---|---|---|
| 1 | Two-stage conditional diffusion with differential attention (Chen et al.) | TIP 2026 | Benchmark protocol and the 17-method comparison |
| 2 | SSR-NET (Zhang et al.) | TGRS 2021 | Retrained baseline |
| 3 | PSRT (Deng et al.) | TGRS 2023 | Retrained baseline |
| 4 | DCT / Reciprocal transformer (Ma et al.) | Inf. Fusion 2024 | Retrained baseline |
| 5 | MIMFormer (Li et al.) | JSTARS 2024 | Retrained baseline (strongest) |
| 6 | DHIF-Net (Huang et al.) | TCI 2022 | Retrained unfolding baseline (learned operators) |
| 7 | MHF-Net (Xie et al.) | TPAMI 2022 | Unfolding with learned operators |
| 8 | MoG-DCN (Dong et al.) | TIP 2021 | Model-guided fusion network |
| 9 | CaFormer (Cao et al.) | Pattern Recognit. 2026 | Transformer-prior unfolding (prior art) |
| 10 | USRNet (Zhang, Van Gool, Timofte) | CVPR 2020 | Kernel passed to an unfolded data step |
| 11 | SRMD (Zhang, Zuo, Zhang) | CVPR 2018 | Training over multiple degradations |
| 12 | HySure (Simões et al.) | TGRS 2015 | Operator estimation from the observation pair |
| 13 | FUSE (Wei, Dobigeon, Tourneret) | TIP 2015 | Closed-form fusion under periodic borders |
| 14 | Coupled tensor factorization (Kanatsoulis et al.) | TSP 2018 | Separable degradation model |
| 15 | Unknown boundaries (Almeida, Figueiredo) | TIP 2013 | Border rule in real images |
| 16 | DDNM (Wang, Yu, Zhang) | ICLR 2023 | Range–null-space decomposition |
| 17 | Restormer (Zamir et al.) | CVPR 2022 | Transformer blocks of the prior |

## A. Benchmark protocol

**1. Y. Chen, R. Liu, W. K. Wong, J. Wen, "A Two-Stage Conditional Diffusion Model With Differential Attention for
Hyperspectral and Multispectral Image Fusion," *IEEE Trans. Image Process.*, vol. 35, pp. 8980–8995, 2026.**
[doi:10.1109/tip.2026.3723234](https://doi.org/10.1109/tip.2026.3723234)
- Defines the Chikusei ×4 protocol we follow (WorldView-2 response, 7×7 Gaussian blur with σ = 2, 832 training,
  64 validation and 8 test images) and supplies the 17 methods of its Table IV that the paper lists as context.
- Its code repository (github.com/Ruijie2580/DifferentialDiff) contains only a README, so it cannot be retrained.

## B. Published networks retrained under the equal-budget pipeline

**2. X. Zhang, W. Huang, Q. Wang, X. Li, "SSR-NET: Spatial–Spectral Reconstruction Network for Hyperspectral and
Multispectral Image Fusion," *IEEE Trans. Geosci. Remote Sens.*, vol. 59, pp. 5953–5965, 2021.**
[doi:10.1109/tgrs.2020.3018732](https://doi.org/10.1109/tgrs.2020.3018732)
- A small CNN; re-implemented in `methods/puformer/hsifuse/models.py`.

**3. S.-Q. Deng, L.-J. Deng, X. Wu, R. Ran, D. Hong, G. Vivone, "PSRT: Pyramid Shuffle-and-Reshuffle Transformer for
Multispectral and Hyperspectral Image Fusion," *IEEE Trans. Geosci. Remote Sens.*, vol. 61, pp. 1–15, 2023.**
[doi:10.1109/tgrs.2023.3244750](https://doi.org/10.1109/tgrs.2023.3244750) · code: [github.com/Deng-shangqi/PSRT](https://github.com/Deng-shangqi/PSRT)
- Retrained from its official code; also retrained with its authors' own recipe (AdamW, weight decay, clipping,
  batch 32) to check the effect of the pipeline.

**4. Q. Ma, J. Jiang, X. Liu, J. Ma, "Reciprocal transformer for hyperspectral and multispectral image fusion,"
*Information Fusion*, vol. 104, art. 102148, 2024.**
[doi:10.1016/j.inffus.2023.102148](https://doi.org/10.1016/j.inffus.2023.102148) · code: [github.com/qingma2016/DCTransformer](https://github.com/qingma2016/DCTransformer)
- "DCT" in the tables; among the strongest entries of the TIP 2026 comparison. Its repository also supplies the
  Nikon D700 response used for CAVE and Harvard.

**5. R. Li, L. Zhang, Z. Wang, X. Li, "MIMFormer: Multiscale Inception Mixer Transformer for Hyperspectral and
Multispectral Image Fusion," *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 17, pp. 15122–15135, 2024.**
[doi:10.1109/jstars.2024.3447648](https://doi.org/10.1109/jstars.2024.3447648) · code: [github.com/meiruni/MIMFormer](https://github.com/meiruni/MIMFormer)
- The strongest retrained baseline on every dataset; retrained with three seeds, with its authors' recipe and with
  PUFormer's recipe. Its authors also release the real ZY1-02D dataset "ZY1E".

**6. T. Huang, W. Dong, J. Wu, L. Li, X. Li, G. Shi, "Deep Hyperspectral Image Fusion Network With Iterative
Spatio-Spectral Regularization," *IEEE Trans. Comput. Imaging*, vol. 8, pp. 201–214, 2022.**
[doi:10.1109/tci.2022.3152700](https://doi.org/10.1109/tci.2022.3152700) · code: [github.com/TaoHuang95/DHIF-Net](https://github.com/TaoHuang95/DHIF-Net)
- An unfolding network whose blur, decimation and spectral operators are learned convolution layers: the
  learned-operator design that our exact operators are compared with.

## C. Model-guided and unfolded fusion

**7. Q. Xie, M. Zhou, Q. Zhao, Z. Xu, D. Meng, "MHF-Net: An Interpretable Deep Network for Multispectral and
Hyperspectral Image Fusion," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, pp. 1457–1473, 2022.**
[doi:10.1109/tpami.2020.3015691](https://doi.org/10.1109/tpami.2020.3015691)
- Unfolds a fusion model with operators learned as convolutions; the learned-operator ablation row reproduces this
  operator model.

**8. W. Dong, C. Zhou, F. Wu, J. Wu, G. Shi, X. Li, "Model-Guided Deep Hyperspectral Image Super-Resolution,"
*IEEE Trans. Image Process.*, vol. 30, pp. 5754–5768, 2021.**
[doi:10.1109/tip.2021.3078058](https://doi.org/10.1109/tip.2021.3078058)
- Model-guided fusion network with data-consistency steps; part of the unfolding family PUFormer belongs to.

**9. X. Cao, X. Wang, X. Dun, Y. Lian, X. Cheng, X. Hao, "Cross-domain-aware deep unfolding transformer for
hyperspectral image super-resolution," *Pattern Recognition*, vol. 172, art. 112374, 2026.**
[doi:10.1016/j.patcog.2025.112374](https://doi.org/10.1016/j.patcog.2025.112374)
- A recent unfolding network with a transformer prior, one of the reasons the paper does not claim the transformer
  prior as new.

## D. Operator-aware restoration

**10. K. Zhang, L. Van Gool, R. Timofte, "Deep Unfolding Network for Image Super-Resolution," *Proc. IEEE/CVF CVPR*,
pp. 3214–3223, 2020.** [doi:10.1109/cvpr42600.2020.00328](https://doi.org/10.1109/cvpr42600.2020.00328)
- Half-quadratic-splitting unfolding with the blur kernel passed to the data step and the noise level to a
  hyper-parameter module: the closest prior art of the operator-conditioned training, which this project extends
  to the two operators of fusion, with exact borders and operators estimated from the observation pair.

**11. K. Zhang, W. Zuo, L. Zhang, "Learning a Single Convolutional Super-Resolution Network for Multiple
Degradations," *Proc. IEEE/CVF CVPR*, pp. 3262–3271, 2018.** [doi:10.1109/cvpr.2018.00344](https://doi.org/10.1109/cvpr.2018.00344)
- Training one network over a family of degradations; the randomized-training baseline ("randomized data only")
  follows this idea.

## E. Theory the contributions build on or are compared with

**12. M. Simões, J. Bioucas-Dias, L. B. Almeida, J. Chanussot, "A Convex Formulation for Hyperspectral Image
Superresolution via Subspace-Based Regularization," *IEEE Trans. Geosci. Remote Sens.*, vol. 53, pp. 3373–3388,
2015.** [doi:10.1109/tgrs.2014.2375320](https://doi.org/10.1109/tgrs.2014.2375320)
- HySure estimates the spatial and spectral responses from the relation D·Y_M = R·Y_H; Theorem 1 proves when this
  relation identifies the operator parameters uniquely.

**13. Q. Wei, N. Dobigeon, J.-Y. Tourneret, "Fast Fusion of Multi-Band Images Based on Solving a Sylvester
Equation," *IEEE Trans. Image Process.*, vol. 24, pp. 4109–4121, 2015.** [doi:10.1109/tip.2015.2458572](https://doi.org/10.1109/tip.2015.2458572)
- The closest closed form of the fusion data subproblem, by Fourier transforms under periodic borders; Theorem 3
  gives a closed form for the true border rule through the separable eigendecomposition.

**14. C. I. Kanatsoulis, X. Fu, N. D. Sidiropoulos, W.-K. Ma, "Hyperspectral Super-Resolution: A Coupled Tensor
Factorization Approach," *IEEE Trans. Signal Process.*, vol. 66, pp. 6503–6517, 2018.**
[doi:10.1109/tsp.2018.2876362](https://doi.org/10.1109/tsp.2018.2876362)
- Uses separable (Kronecker) degradation operators to identify low-rank tensor models; Theorem 3 uses the same
  separability for an exact data step and the fusion null space.

**15. M. S. C. Almeida, M. A. T. Figueiredo, "Deconvolving Images With Unknown Boundaries Using the Alternating
Direction Method of Multipliers," *IEEE Trans. Image Process.*, vol. 22, pp. 3074–3086, 2013.**
[doi:10.1109/tip.2013.2258354](https://doi.org/10.1109/tip.2013.2258354)
- In real images the border rule is unknown; Corollary 1 bounds what any fixed border rule can cost.

**16. Y. Wang, J. Yu, J. Zhang, "Zero-shot image restoration using denoising diffusion null-space model," *Proc. ICLR*,
2023.**
- Range–null-space decomposition for general restoration; Theorem 3(c) gives the fusion null space in closed form
  and Section VI-C uses it to split the error of every method.

## F. Building block of the prior

**17. S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, "Restormer: Efficient Transformer for
High-Resolution Image Restoration," *Proc. IEEE/CVF CVPR*, pp. 5718–5729, 2022.**
[doi:10.1109/cvpr52688.2022.00564](https://doi.org/10.1109/cvpr52688.2022.00564)
- The transposed-attention and gated feed-forward blocks of PUFormer's prior; the ablation shows a convolutional
  mixer performs as well, so the prior is not claimed as a contribution.
