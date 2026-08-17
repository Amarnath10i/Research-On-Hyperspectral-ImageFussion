# Observation-identifiable rank — definition and estimator

## Equivalence classes

Define the observation-identifiability equivalence on HR-HSI reconstructions:

    X_1 ~_A X_2   <=>   A(X_1) = A(X_2),      A = [D; R].

All members of the class `[X]_A` produce identical observations.  The fusion
problem is therefore not `Y -> X` but `Y -> [X]_A`; the identifiable content
of the inverse problem is the *quotient*, and the ambiguous content is the
kernel of `A`.  This is the conceptual foundation shared by P1, P2 and P4.

## r_id for the low-rank model

Under `X = U_r Z` with full-rank spatial coefficients `Z`, the HR-MSI is

    Y_M = R^T X = (R^T U_r) Z.

Its spectral subspace is the column space of `R^T U_r` (M x r), so the number
of spectral degrees of freedom observable at HR resolution is

    r_id = rank(R^T U_r).

For a noise-free observation, `rank(Y_M) = r_id`.  Under noise the *estimated*
identifiable rank is the number of MSI spectral singular values above the
optimal hard threshold.

## Estimator (non-neural, observation-only)

1. **Noise level** — from the LR-HSI's spectral matrix (B x hw): its trailing
   singular values (beyond an energy-0.9 cap) lie in the Marchenko-Pastur
   noise bulk, so

       sigma_hat = median( s[r_cap:] ) / sqrt(hw).

   In the noise-free case the tail is ~0 and sigma_hat ~ 0.

2. **Threshold** — for the MSI spectral matrix (M x HW), n = HW, beta = M/n,
   the Gavish-Donoho optimal hard threshold is

       tau = omega(beta) * sigma_hat * sqrt(n),
       omega(beta) = 0.56 beta^3 - 0.95 beta^2 + 1.43 beta + 1.43.

   In the noise-free case tau degrades to a small relative floor
   `floor_rel * s_max` so the structurally significant directions are counted
   exactly.

3. **r_id_hat** = number of MSI singular values >= max(tau, floor_rel * s_max).

The intrinsic rank estimate `r_hat` uses the same machinery on the LR-HSI's
spectral matrix (B x hw).

## Validated behaviour (proposal2/rankest/selfcheck.py)

- **Rank sweep (clean, B=64, M=48):** exact recovery `|r_hat - r| = 0` and
  `|r_id_hat - r_id| = 0` for r in {3,5,8,12,20,30}.
- **Cap:** `r_id_hat <= M` always.
- **Noise sweep (r=12, M=16):** r_id_hat is non-increasing with decreasing SNR
  (12, 10, 6, 6, 5); sigma_hat within ~5% of the injected sigma.
- **Band-count sweep (clean, r=12):** `|r_id_hat - r_id| = 0` for M in {4,8,16,31}.
- **SRF-overlap sweep (clean, r=12, M=16):** r_id tracks the reduction caused
  by overlapping SRF columns (12, 12, 11, 6) exactly.