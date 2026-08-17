# Assumptions and scope

## Modelling assumptions (controlled synthetic regime)

1. **Linear mixing / low rank.** `X = U_r Z` with an orthonormal spectral
   basis `U_r` (B x r) and spatial coefficients `Z` (r x HW) whose rows are
   linearly independent (orthogonal cosine modes used in the generator).

2. **Observation model.** `Y_H = D(X)` with per-band (channel-wise) Gaussian
   blur followed by strided downsampling by `scale`, and `Y_M = R^T X` with a
   known SRF `R` (B x M).  Noise is additive Gaussian with a shared standard
   deviation `sigma` on both observations (calibration regime).

3. **Identifiable-rank ground truth.** `r_id = rank(R^T U_r)`, computed by
   SVD with a `1e-6` relative threshold, equals the rank of the noise-free
   HR-MSI spectral subspace.

## Estimator assumptions

4. **Noise tail.** The estimator recovers `sigma` from the LR-HSI trailing
   singular values; this requires `B - r` to be comfortably non-zero (a
   handful of noise directions).  For near-saturated intrinsic rank
   (`r ~ B`) a wider spectral acquisition (more bands) is used — e.g. the
   rank sweep runs at B=64.

5. **Spatial sampling.** `hw >= r` (more LR pixels than spectral directions)
   so the LR-HSI's spectral space spans `U_r`.

6. **Spatial modes below the LR Nyquist.** Generator modes use frequencies
   below `1/(2*scale)` cycles/pixel so blur+downsampling preserves their
   linear independence (no aliasing rank collapse).

## Out of scope (deferred to later papers / P2 extensions)

- Unknown `D` or `R` (this is P1/P4 territory): the estimator is *non-blind*
  in the sense that the threshold uses the shared noise level, not an oracle
  on the operators themselves; `R` appears only through the MSI it generates.
- Spectral *diversity* of the scene (materials) — deferred to the P2 paper's
  spectral-diversity sweep and to P4.
- Degradation mismatch and registration error — deferred to the robustness
  section of the P2 paper.
- Adaptive `sigma` estimation for the MSI alone (currently shared-noise
  assumption); robustness analysis planned.

## Protocol note

The synthetic generator is the P2 "controlled falsification" tool: it fixes
`r`, `r_id`, `sigma`, band count, SRF overlap, and reports `|r_hat - r_id|`
as the primary metric.  All numbers above are reproducible on CPU via
`python -m proposal2.rankest.selfcheck`.