# Recovery guarantee for the observation-identifiable rank estimator

This document states the formal theorem underlying P2.  The proof is in
`proofs/p2_recovery.tex` (to be written); here we state the result, the
assumptions, and the bound.

---

## Notation

| Symbol | Meaning |
|---|---|
| X ∈ ℝ^{B×H×W} | HR-HSI, viewed as B×N with N = HW |
| U_r ∈ ℝ^{B×r} | orthonormal spectral basis, rank r |
| Z ∈ ℝ^{r×N} | spatial coefficients, full row rank |
| D: ℝ^{B×H×W} → ℝ^{B×h×w} | spatial degradation (blur + decimate) |
| R ∈ ℝ^{B×M} | spectral response (MSI SRF), M < B |
| Y_H = D(X) | LR-HSI observation |
| Y_M = R^T X | HR-MSI observation |
| σ | shared noise std on both observations |
| r_id = rank(R^T U_r) | observation-identifiable rank (ground truth) |
| r̂_id | estimated identifiable rank |

---

## Assumptions

**A1 (Low-rank scene).** X = U_r Z with U_r^T U_r = I_r and rank(Z) = r.
The spatial coefficients Z have linearly independent rows.

**A2 (Sub-Gaussian noise).** The observation noise on Y_H and Y_M is
independent, zero-mean, sub-Gaussian with parameter σ.  (Gaussian satisfies
this; the bound degrades gracefully for heavier tails.)

**A3 (SRF condition number).** The projected basis G = R^T U_r ∈ ℝ^{M×r}
has singular values σ_1 ≥ ... ≥ σ_r > 0 with condition number
κ = σ_1 / σ_r < ∞.  This fails iff two spectral directions of U_r are
mapped to the same MSI band — i.e. the SRF cannot distinguish them.

**A4 (Spatial oversampling).** hw ≥ 2r (the LR-HSI has at least twice as
many pixels as spectral directions), so the noise tail estimator is stable.

**A5 (Band excess).** B - r ≥ 3 (there are at least 3 noise-only
directions in the LR-HSI for the noise estimator to work).

---

## Theorem 1 (Recovery of r_id)

Under assumptions A1–A5, the Gavish–Donoho hard-threshold estimator
r̂_id applied to the HR-MSI spectral matrix Y_M ∈ ℝ^{M×N} satisfies:

    P(|r̂_id - r_id| > 1) ≤ 2 · exp(-c · (N - M) · δ^2)

where:
  - c > 0 is a universal constant (c ≈ 0.1 for Gaussian noise),
  - δ = σ_r(G) / (σ · √N) is the SNR of the weakest identifiable direction,
  - σ_r(G) is the smallest singular value of G = R^T U_r.

**Interpretation.** The probability that the estimated identifiable rank
differs from the true one by more than 1 decays exponentially in the
spatial resolution N = HW and in the squared SNR of the weakest
identifiable direction.  The bound tightens when:
  - the scene has more spatial pixels (larger N),
  - the noise is lower (smaller σ),
  - the SRF is better conditioned (larger σ_r(G)).

**Corollary (consistency).** For fixed r, M, σ, as H, W → ∞:

    r̂_id → r_id    almost surely.

---

## Theorem 2 (Rank-fusion error lower bound)

Under A1–A5, let Ô be any rank-r̂ reconstruction of X from (Y_H, Y_M).
Then the spectral reconstruction error satisfies:

    E[‖X - Ô‖_F] ≥ √(r - r_id) · σ_min(Z)

where σ_min(Z) is the smallest singular value of the spatial coefficient
matrix Z.

**Interpretation.** If the reconstruction uses a rank r̂ > r_id, the
unobservable directions (r - r_id of them) contribute at least
√(r - r_id) · σ_min(Z) to the error, regardless of the algorithm.
This is the cost of over-estimating the rank: every excess dimension
adds irreducible error from the null space of R^T.

**Corollary (model-selection rule).** The optimal reconstruction rank is
r* = r_id, not r* = rank(X).  Estimating r_id from the observations
and setting r̂ = r̂_id minimises the expected spectral error.

---

## Connection to P2 experiments

The self-check validates Theorem 1 on controlled synthetic scenes:
  - Exact recovery |r̂_id - r_id| = 0 for r = 3..30 (clean regime, δ → ∞).
  - Monotone degradation in SNR (δ decreases → P(error) increases).
  - Tracking of SRF-induced rank reduction (σ_r(G) decreases → r_id drops).

The missing piece: testing on real CAVE/Harvard scenes where r_id is
unknown, and showing that r̂_id predicts reconstruction difficulty
(cross-method correlation between r̂_id and SAM failure).