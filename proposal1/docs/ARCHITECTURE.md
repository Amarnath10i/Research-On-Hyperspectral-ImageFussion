# DAETF-Net: Domain-Adaptive Equivariant Tensor Fusion Network

**Status:** implemented and executable. Every mechanism described below exists in
[`proposal1/daetf/`](../daetf/) and is verified numerically by
[`selfcheck.py`](../daetf/selfcheck.py).

This document supersedes [`ARCHITECTURE_v1_original.md`](ARCHITECTURE_v1_original.md),
which described an architecture that the v1 code did not implement. The
differences are recorded in [§8](#8-what-changed-from-v1) rather than quietly
dropped, because knowing which claims were unsupported is what motivated this
version.

---

## 1. The problem

Ten published HSI-MSI fusion methods were re-run on CAVE and Harvard
(`existing/`). Their recorded results show something the usual PSNR-first
reporting hides:

| | in-domain | cross-domain | change |
|---|---|---|---|
| PSNR | 36-50 dB | 26-70 dB | often **improves** |
| SAM | 2.2-7.6 deg | 2.6-58.9 deg | **degrades badly** |
| ERGAS | 0.27-4.07 | 0.93-302 | **degrades badly** |

PSNR frequently *rises* on the harder dataset — five of nine methods score
higher on Harvard than on CAVE — because per-image maximum normalisation
inflates it on darker scenes. What actually collapses is spectral fidelity:
Fusformer's SAM goes from 2.35 to 58.89 deg, UTAL's from 4.62 to 36.46, and
IFCASformer's ERGAS from 3.55 to 85.41.

**So the target is spectral fidelity under domain shift, not another decimal of
in-domain PSNR.** Three consequences shape the design:

1. Optimise SAM directly; it is the metric that fails.
2. Build the observation model into the network and the loss, since the physics
   is the only thing that stays true when the domain changes.
3. Give the model a way to adapt without labels, because the target domain has
   no ground truth in deployment.

---

## 2. Architecture

```
 LR-HSI ──┬─────────────────► Back-Projection Upsampler ──► y0 (coarse HR-HSI)
          │                            ▲   │
          │                            └───┘  iterative:  y ← y + Up(x − Down(y))
          │
          ├──► Degradation Encoder ──► code d ──► FiLM ──┐
 MSI  ────┤                                              │
          └──► MSI Encoder ─────────────────► f_msi ──► FiLM
                                                          │
    y0 ──► Equivariant Feature Extractor ──► f_hsi ──► FiLM
                                                          │
                       (f_hsi, f_msi) ──► Tucker Interaction (TSSE)
                                                │
                                     Region-Aware MoE (per-pixel top-k)
                                                │
                                   Wavelet Refinement (FDRM)
                                                │
                                     residual ──►  ŷ = y0 + r
```

### 2.1 Equivariant Feature Extractor (EFE)

p4 group-equivariant convolutions: a lifting convolution Z² → p4 produces an
explicit orientation axis of size 4 by convolving with four rotated copies of a
shared kernel; group convolutions p4 → p4 rotate the spatial support **and**
cyclically shift the orientation axis together. Doing only one of the two is the
usual way a layer claims equivariance without having it.

`BatchNorm3d` shares statistics across orientations so normalisation preserves
equivariance, and a max over the orientation axis returns a plain feature map
that rotates with the input.

> Verified: `check_equivariance` measures `max|rot90(EFE(x)) − EFE(rot90(x))| =
> 7.6e-06`.

Rotation robustness is a real property of the network rather than something
hoped for from augmentation — which matters because sensor geometry differs
across datasets.

### 2.2 Degradation-Aware Encoder (DAE) + FiLM

A small CNN reads the observed pair and produces a degradation code, which
modulates every downstream block through zero-initialised FiLM layers (so the
conditioned model starts exactly at the unconditioned one). An auxiliary head
regresses the true degradation parameters — blur σx, σy, orientation as
sin2θ/cos2θ, and noise σ — which are known during training because the pipeline
synthesises them. Without that supervision the code collapses to a constant and
the conditioning does nothing.

This is the blind/degradation-aware idea from the 2026 TGRS literature: one
model covers a range of degradations instead of assuming the single fixed
bicubic kernel it was trained on.

### 2.3 Tensor Spectral-Spatial Encoder (TSSE)

A Tucker-style multilinear contraction:

```
z[b,k,h,w] = Σ_{i,j} G[i,j,k] · a[b,i,h,w] · b[b,j,h,w]
```

`a` and `b` are rank-R projections of the HSI-path and MSI-path features, and
`G` is a learned core tensor. It is realised as an outer product followed by a
1×1 convolution **whose weights are G**, so the core genuinely participates.
A nuclear-norm penalty on the mode-3 unfolding acts as a convex rank surrogate.

> Verified: `check_core_used` confirms gradient reaches `G`. In v1 the core was
> allocated and never referenced in `forward`, so it was a dead parameter.

### 2.4 Region-Aware Mixture-of-Experts (AF-MoE)

The gate is a convolution producing a per-pixel distribution over experts, with
top-k routing and a load-balancing penalty (squared coefficient of variation of
expert usage; without it top-k routing collapses onto one expert).

A globally pooled gate must commit to one fusion strategy per image. Routing per
pixel lets shadow, foliage and flat texture in the same scene take different
experts. The gate is also directly viewable as a map, which is the
interpretability the v1 document promised but could not deliver from a single
global vector.

### 2.5 Frequency-Domain Refinement Module (FDRM)

An orthonormal Haar DWT (fixed grouped convolution) splits features into LL/LH/HL/HH.
Each subband gets its own learnable processing; the three detail subbands get a
learnable soft-threshold — classical wavelet shrinkage, made differentiable and
learned. A 1×1 convolution mixes across subbands, then an exact inverse
transform reconstructs.

> Verified: `check_wavelet` measures `max|IDWT(DWT(x)) − x| = 2.4e-07`.

### 2.6 Back-Projection Upsampler (BPU) — replacing bicubic

```
y ← Up(x)
repeat:  y ← y + Up_res( x − Down(y) )
```

The residual is measured in LR space, where the actual observation is available,
so every step pulls the estimate back onto the observation manifold. Bicubic
interpolation cannot do this: it never checks whether its own output re-explains
the input. `Down` is learnable and `Up` is PixelShuffle-based over a bicubic
prior, so the module starts no worse than interpolation and improves from there.

---

## 3. The SPC loss

**Spectral-Physical Composite:**

```
L = w₁·Charbonnier + w₂·SAM + w₃·gradient + w₄·(1 − SSIM)     [fidelity]
  + w₅·‖Down(ŷ) − LR‖ + w₆·‖SRF(ŷ) − MSI‖                     [physics]
  + w₇·MoE-balance + w₈·‖G‖_* + w₉·degradation-regression      [regularisers]
  + w₁₀·MMD(source, target)                                    [domain]
```

Two properties carry the research claim.

**The SAM term optimises the metric that actually fails.** Every baseline
optimises L1/L2 or SSIM, which is why they hold spatial quality and lose
spectra under transfer.

**The physics terms need no ground truth.** `Down(ŷ)` should reproduce the LR
input; `SRF(ŷ)` should reproduce the MSI. Both are computable on any scene from
any sensor. That is what makes the same objective usable for test-time
adaptation (§4).

The SRF is not hand-picked: it is recovered from the data by least squares,
`min_S ‖HSI·S − RGB‖²`, so it adapts automatically to a dataset whose RGB was
rendered with a different response.

> Verified on synthetic data with a known response: recovery error 2.3e-08.

---

## 4. Domain-shift strategy

Three independent defences, each separately ablatable:

1. **Domain randomisation** (training) — anisotropic Gaussian blur with random
   σ and orientation, random noise, jittered synthetic SRFs, flips and 90°
   rotations. The model never sees one fixed degradation.
2. **MMD alignment** (training) — multi-bandwidth RBF MMD between source and
   *unlabelled* target features, with the bandwidth set from the median pairwise
   distance. No target ground truth is used, so cross-domain evaluation stays
   honest.
3. **Test-time adaptation** (inference) — per scene, minimise only the two
   physics terms, updating only the conditioning parameters (degradation
   encoder, FiLM, MoE gate, FDRM). Cheap, stable, and label-free.

---

## 5. Evaluation protocol

The v1 benchmark could not support cross-method claims: scale factors of ×4, ×8,
×16 and ×32, different normalisations, ERGAS scale arguments that did not match
the actual downsampling, and one method evaluated on a different dataset and
task entirely. Fixed here:

- **One scale factor** for every comparison, stated explicitly.
- **One metric implementation** (`daetf/metrics.py`) with a constant
  `data_range=1.0` — never the per-image maximum.
- **One degradation** for evaluation (fixed Gaussian blur + decimation).
- **Full scenes**, via Hann-weighted overlapping tiles, so no method is scored
  on centre crops while another gets whole images.
- **Per-scene results retained**, enabling paired tests.

---

## 6. Evidence produced

[`daetf/experiments.py`](../daetf/experiments.py) generates:

- paired Wilcoxon signed-rank tests and Cohen's *d* against each baseline
  (scenes are shared, so pairing is the correct and more sensitive design on
  10-20 scenes)
- bootstrap 95% confidence intervals
- component ablation against **matched control arms** — plain convolutions for
  the equivariant stem, bicubic for back-projection, concat-fusion for the
  Tucker interaction — so an ablation measures the mechanism, not lost capacity
- multi-seed repeats reported as mean ± std
- cost: parameters, GFLOPs, latency, peak memory
- Markdown and LaTeX tables

---

## 7. Cost

At the default configuration: **2.06 M parameters** (8.2 MB fp32). Designed for
a single P100 (16 GB): 64×64 HR patches, batch 16, fp16 AMP (P100 has real fp16
throughput but no tensor cores), and tiled inference for full 512×512 and
1040×1392 scenes.

---

## 8. What changed from v1

The v1 documents claimed six mechanisms. Auditing the v1 notebooks against them:

| Claim in v1 docs | v1 code | v2 |
|---|---|---|
| Group-equivariant CNN | plain `Conv2d`+BN+ReLU; `groupy` installed, never imported | real p4 convs, verified 7.6e-06 |
| Tucker decomposition | element-wise product; core allocated, **never used in forward**; `tensorly` imported, never called | real contraction, core carries gradient |
| Mixture-of-Experts | implemented (global gate) | per-pixel top-k + load balancing |
| Dual-tree wavelet FDRM | 3×3/5×5/7×7 convolutions, nothing frequency-domain | Haar DWT + learnable shrinkage, verified 2.4e-07 |
| MMD domain adaptation | absent | implemented |
| Training on CAVE/Harvard | `torch.randn(...)`, hardcoded length 100 | real `.mat` loading with a physical forward model |

The v1 training notebook also could not run: its dummy MSI was 64×64 while the
model upsampled the HSI 4× to 256×256, so `torch.cat` raised a shape error.

---

## 9. Honest limitations

- **Baseline numbers are not yet like-for-like.** The comparison table reuses
  results recorded under each baseline's own protocol. A defensible claim needs
  every baseline re-run under the protocol in §5; until then only rows marked
  same-protocol are strictly comparable.
- **Whether this beats SOTA is an empirical question**, settled by the training
  run, not by the design. The mechanisms are verified; the ranking is not
  claimed in advance.
- Scale generalisation is measured but the learned upsampler is tied to its
  training factor; evaluating at another factor tests degradation
  generalisation, not retrained performance.
- Only two datasets. Additional sensors would strengthen the transfer claim.
