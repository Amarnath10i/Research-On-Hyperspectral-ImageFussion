# Projective Spectral Embedding Fusion: a focused Q1 paper design

## Decision

Proposal 1 should **not** be submitted as a claim that a long list of familiar
modules is novel.  Equivariant convolutions, Tucker interactions, FiLM,
mixture-of-experts routing, wavelets, MMD, and test-time adaptation are all
existing techniques.  Combining them is engineering unless experiments prove
that a particular combination solves a well-defined problem better than simpler
alternatives.

The paper's contribution is instead the **Projective Spectral Embedding (PSE)**
in `daetf/spectral_embed.py`, operating on top of the range/null-space
formulation:

```text
Y_hat = D_hat^+(X) + P_perp_hat F_theta(X, M)
```

1. **The metric-aligned manifold (the headline idea).**  Pixel spectra are
   projected onto the unit sphere (intensity factored out — illumination can no
   longer hide spectral error) and mapped through a calibrated spectral MLP
   whose Euclidean distance tracks spectral angle.  Optimising L2 on this
   manifold *is* optimising SAM, the metric that actually fails under domain
   shift.  Intensity is neither learned nor penalised: it is factored out by
   construction, and the part the observation determines is carried exactly by
   the range/null decomposition.
2. **The constrained formulation (the enabling mechanism).**  The learned
   residual is projected into the null space of the estimated observation
   operator, so it can add missing spatial detail but cannot overwrite the LR
   measurement under the same operator.

The paper must call the operator consistency **estimated-operator data
consistency** in blind settings, reserving the word *exact* for tests where the
same operator generates, reconstructs, and verifies an observation.  The PSE
manifold claim must be supported by the calibration selfchecks and by a SAM-vs-
L2-in-manifold agreement statistic on held-out spectra.

## Focused architecture

```text
LR-HSI X ──► degradation encoder ────────────► condition code d ──► FiLM
    │                                                                     │
    ├──► estimated D_hat ─► D_hat^+(X) = Y_range ─► HSI encoder ─────────┤
    │                                                                     ▼
HR-MSI M ─────────────────────────────────────────► MSI encoder ─► lightweight fusion
                                                                      │
                                                               residual F_theta
                                                                      │
                                                         P_perp_hat(residual)
                                                                      │
                         final Y_hat = Y_range + P_perp_hat(residual) ◄┘
```

The **paper-core configuration** is available in code as
`Config.paper_core()`.  It keeps:

- projective spectral embedding (intensity-invariant, SAM-metric-calibrated);
- range/null-space decomposition;
- a degradation encoder and FiLM conditioning;
- plain HSI and MSI feature extractors;
- lightweight concatenation fusion and residual reconstruction;
- supervised spectral losses and physical observation losses.

It turns off p4 equivariance, Tucker fusion, MoE, wavelet refinement, and the
disagreement field.  Those are optional follow-up ablations, not automatic
paper contributions.

The corresponding controlled ladder is `PAPER_CORE_ABLATIONS` in
`daetf/experiments.py`.  Run it from `Config.paper_core()` rather than from the
legacy all-modules default.

## Why this hypothesis is worthwhile

HSI fusion is ill-posed: many HR cubes reduce to the same LR-HSI.  Standard
residual fusion asks a neural network to predict the whole cube and then
penalises a mismatch to the LR measurement.  The proposed split computes the
component that the LR observation determines and restricts the learned
component to the remaining degrees of freedom.  HR-MSI supplies spatial detail,
while LR-HSI retains the spectral identity.

The hypothesis is falsifiable:

> Compared with equally sized unconstrained and bicubic/back-projection
> baselines, (a) training on the projective spectral manifold and (b) the
> range/null-space residual projection reduce LR-consistency error and improve
> spectral fidelity (SAM and ERGAS), especially when blur/noise and illumination
> shift between source and target sensors.

A secondary, independently testable claim: after calibration, L2 distance in
the embedding predicts spectral angle on held-out spectra (reported as a
correlation / mean-absolute-error statistic), which is what justifies training
with L2 in the manifold instead of a raw SAM term.

## Correct degradation parameterisation

The degradation encoder predicts five physical quantities:

```text
[sigma_x, sigma_y, sin(2 theta), cos(2 theta), noise]
```

The model now decodes raw neural-head values once into physical values before
both (a) comparing them to the simulated ground truth and (b) constructing the
Gaussian kernel.  The old implementation supervised raw values but applied an
additional softplus when building the kernel, so a correct predicted sigma did
not create the correct blur.  This has been corrected in `nullspace.py` and
`model.py`.

## Claims that are allowed

Only make these claims after the required experiment succeeds:

1. **Data consistency:** report `||D(Y_hat)-X||` under the same operator used
   for generation and reconstruction; report estimated-operator consistency
   separately for blind reconstruction.
2. **Manifold calibration:** L2 distance in the learned embedding predicts
   spectral angle on held-out spectra (report agreement statistic), and the
   embedding is invariant to per-pixel intensity scaling.
3. **Spectral robustness:** demonstrate lower SAM and ERGAS than matched
   baselines over per-scene paired comparisons and three or more training
   seeds.
4. **Domain robustness:** use source-train, target-unlabelled-train (if UDA is
   used), and target-test splits with no scene overlap.  State clearly whether
   the protocol is inductive or transductive.
5. **Efficiency:** compare parameter count, peak memory, full-scene latency,
   and MACs under the same hardware, precision, and input size.

Never claim SOTA or Q1 readiness from a design diagram, mechanism self-check,
or differently generated published numbers.

## Required experiment sequence

All rows use the same CAVE/Harvard splits, scale factor, normalisation,
blur/downsample operator, SRF, full scenes, and metric implementation.

| Stage | Model | Question answered |
|---|---|---|
| 0 | Bicubic, GSA, HySure/CNMF where reproducible | Is the dataset/protocol correct? |
| 1 | Plain residual fusion, capacity matched | Is MSI-guided learning useful? |
| 2 | Stage 1 + physical losses | Do constraints help? |
| 3 | Stage 2 + range/null projection | Does the central consistency mechanism help? |
| 4 | Stage 3 + projective spectral embedding | Does the manifold training objective help? |
| 5 | Stage 4 + blind degradation conditioning | Does estimated degradation help? |
| 6 | One optional module at a time | Does each addition earn its complexity? |

For every stage report per-scene PSNR, SSIM, SAM, ERGAS, LR-consistency error,
and SRF-consistency error.  Use paired Wilcoxon tests, bootstrap confidence
intervals, and at least three seeds for the final contenders.  All comparison
code must be released.

## Stop/go criteria

Do **not** add architectural modules while a simpler baseline fails.  Advance
only when Stage 3 consistently improves SAM and ERGAS over Stage 2 and a
matched unconstrained residual model.  If it does not, report the negative
result and investigate the observation model, spatial registration, SRF
calibration, and loss balance before widening the model.

The preliminary `results/RESULTS.md` shows DAETF-Net below Bicubic and GSA on
the reported Harvard run.  Therefore the correct immediate objective is a
reproducible baseline audit, not a submission.

## Paper outline if the evidence succeeds

1. Introduction: spectral fidelity under sensor/domain shift; intensity hides
   spectral error.
2. Projective spectral embedding: illumination-invariant, metric-calibrated
   spectral manifold.
3. Observation model and range/null decomposition.
4. Degradation-aware residual fusion network on the manifold.
5. Consistency properties and their limitations under an estimated operator.
6. Unified protocol and experimental results.
7. Ablations, failure cases, cost, and uncertainty/limitations.

This is a credible Q1-oriented paper direction; acceptance remains an empirical
and peer-review outcome, not something an architecture alone can guarantee.
