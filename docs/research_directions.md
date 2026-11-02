# Research directions after the CVPR paper

Gaps that most hyperspectral-multispectral fusion papers leave open, what this project already provides for each,
and a follow-up paper that would close it. Venues are suggestions by fit.

| # | Open problem | What we already have | Follow-up paper | Fits |
|---|---|---|---|---|
| 1 | **Registered and unregistered pixels in one image** (parallax, moving objects, view-angle differences) | global sub-pixel offset in the exact operator, estimated from the pair; confidence-weighted data step whose map marks unregistered pixels; `local_misreg` test condition | spatially varying operator: a displacement field estimated jointly with the image inside the unfolded network; evaluation on real multi-view pairs | IEEE TGRS, TIP, ISPRS J. |
| 2 | **Realistic noise**: band-dependent, signal-dependent, stripes | noise-adaptive step sizes from estimated noise levels; `band_noise` and stripe test conditions; imputation form of the data step keeps the closed form with per-band weights | per-band noise estimation and weighting in the data step; training with sensor noise models (EnMAP, PRISMA, AVIRIS-NG) | IEEE TGRS, JSTARS |
| 3 | **Changes between acquisition dates** (e.g. EnMAP + Sentinel-2 days apart) | confidence maps = learned change detector (ROC AUC in `cond_eval.change_auroc`); robust-fusion lineage (Ferraris et al., TCI 2017) | joint fusion and change detection on real multi-date pairs; the confidence map as an unsupervised change map | Remote Sensing of Environment, IEEE TGRS |
| 4 | **Unknown, non-Gaussian, spatially varying blur** | operator estimation from D Y_M = R Y_H on a parametric family; iterative step for non-separable kernels | nonparametric kernel estimation from the pair; operator families learned from calibration data | IEEE TCI, TIP |
| 5 | **One network for any sensor and scale** | band-space interface (R+ Y_M), random-response training, tests on 7 real sensor responses; scale-randomized training | multi-sensor model across real satellite pairs; arbitrary (non-integer) scale factors | IJCV, IEEE TPAMI (with depth) |
| 6 | **Trustworthy outputs** | exact null-space split; null-space uncertainty head (`unc_eval.py`: Spearman, AUSE, coverage) | calibrated uncertainty for the inferred (null-space) content; its effect on downstream tasks | IEEE TGRS, Information Fusion |
| 7 | **Real paired references** | NEON spectrometer + camera (reduced and full resolution) | a public real-pair benchmark with measured responses and registration; no-reference metrics that respect the null space | Scientific Data, ISPRS J., NeurIPS Datasets track |
| 8 | **Downstream value of fusion** | Chikusei has 19 land-cover classes | classification / unmixing with fused images, showing what consistency and null-space accuracy buy | JSTARS, Remote Sensing |
| 9 | **Other inverse problems with Kronecker structure** | closed-form data step, null-space split, band-space interface | hyperspectral pansharpening (M = 1), spectral super-resolution from RGB, multi-temporal fusion with the same framework | IEEE TGRS, TCI |
| 10 | **Efficient / on-board fusion** | eigen-operator prior (~10x fewer parameters than the U-Net), closed-form step without iterations | a small physics network for on-board processing; latency and memory on embedded GPUs | IEEE GRSL, JSTARS |

## Order that reuses most of the current code

1. Journal version of the CVPR paper (long main.tex): all ablations, x8/x16, CAVE/Harvard, NEON, sensor/scale results.
2. Directions 1 + 3 together (unregistered pixels and multi-date change): one network, real EnMAP/Sentinel-2 pairs.
3. Direction 2 (realistic noise) as a focused TGRS paper.
4. Directions 6 + 8 (trustworthy fusion and its downstream value).
