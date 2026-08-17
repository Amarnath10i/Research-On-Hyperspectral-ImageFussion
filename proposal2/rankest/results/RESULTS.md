# P2 — Identifiable Spectral Complexity: results (scaffold)

Status: **non-neural estimator self-checks ALL PASS** (CPU).  This is the
foundation deliverable per the Q1 restructure: first prove `r_id` can be
estimated on controlled synthetic scenes, before any network is built.

## Selfcheck (proposal2/rankest/selfcheck.py)

```
[1] rank sweep  r=3..30  |r_hat-r|=0, |r_id_hat-r_id|=0   (all)  PASS
[2] cap         r_id_hat <= M                               (M in {4,8,16,31}) PASS
[3] noise trend r_id_hat 12 -> 10 -> 6 -> 6 -> 5 (SNR inf->5)     PASS
[4] band sweep  |r_id_hat-r_id|=0 for M in {4,8,16,31}            PASS
[5] srf sweep   |r_id_hat-r_id|=0, r_id 12->12->11->6             PASS
[6] noise est   sigma_hat/sigma in [0.8,1.2] (1.03-1.05)          PASS
```

## Headline metric: |r_hat - r_id|

| intrinsic rank r | r_id_true | r_hat | r_id_hat | |r_hat-r| | |r_id_hat-r_id| |
|---|---|---|---|---|---|---|
| 3 | 3 | 3 | 3 | 0 | 0 |
| 5 | 5 | 5 | 5 | 0 | 0 |
| 8 | 8 | 8 | 8 | 0 | 0 |
| 12 | 12 | 12 | 12 | 0 | 0 |
| 20 | 20 | 20 | 20 | 0 | 0 |
| 30 | 30 | 30 | 30 | 0 | 0 |

## Phase diagram (noise sweep, r=12, M=16, decaying spectrum)

| SNR (dB) | sigma | sigma_hat | r_hat | r_id_hat |
|---|---|---|---|---|
| inf | - | 2.35e-08 | 12 | 12 |
| 30 | 2.09e-03 | 2.19e-03 | 12 | 10 |
| 20 | 6.61e-03 | 6.93e-03 | 10 | 6 |
| 10 | 2.09e-02 | 2.15e-02 | 8 | 6 |
| 5 | 3.72e-02 | 3.78e-02 | 6 | 5 |

## Spectral sampling (band-count sweep, clean, r=12)

| MSI bands M | r_id_true | r_id_hat | |r_id_hat-r_id| |
|---|---|---|---|---|
| 4 | 4 | 4 | 0 |
| 8 | 8 | 8 | 0 |
| 16 | 12 | 12 | 0 |
| 31 | 12 | 12 | 0 |

## SRF overlap (sweep, clean, r=12, M=16)

| SRF width | r_id_true | r_id_hat | |r_id_hat-r_id| |
|---|---|---|---|---|
| 0.02 | 12 | 12 | 0 |
| 0.06 | 12 | 12 | 0 |
| 0.15 | 11 | 11 | 0 |
| 0.50 | 6 | 6 | 0 |

## Claims validated

- The observation-identifiable rank `r_id = rank(R^T U_r)` is recoverable from
  the observations alone to within +/-1 (exact in the clean regime) across
  intrinsic ranks 3..30.
- `r_id_hat` respects the MSI band-count bound and tracks the loss of
  identifiability caused by (a) noise and (b) SRF overlap — i.e. the
  estimator answers "how many spectral degrees of freedom do the observations
  support?", not "how many does the scene have?".
- Noise is estimated from the data within ~5% (no oracle).

## Next (P2 paper / Kaggle)

- Sweep spectral diversity, degradation mismatch, registration error; build
  the identifiability phase diagram `r_id = f(N_bands, SRF, SNR, scale, ...)`.
- Adaptive complexity: rank-regularised reconstruction `min_{U,Z,r} L(UZ) +
  lambda*C(r)` using `r_id_hat` to set the model's spectral width.
- Protocol evaluation of reconstruction error vs spectral complexity.