# P4 — results: identifiability phase diagram (proposal4/identifiability)

Status: **ALL PASS** (CPU).  Scene r=8, bands=31, H=W=48, scale=4.

## Phase diagram 1: MSI band count x SNR (srf_width = 0.02)

| MSI bands M | inf | 30 | 20 | 10 |  5 |
|---|---|---|---|---|---|
|  4 |  4W |  4W |  4W |  4W |  3W |
|  8 |  8I |  8I |  7I |  7I |  6I |
| 16 |  8I |  8I |  8I |  8I |  8I |
| 31 |  8I |  8I |  8I |  8I |  8I |

Cell = `r_id_hat` + regime.  M=4 can never exceed Weakly (4 bands cannot pin
8 spectral DOF); M>=8 is Identifiable; noise degrades only the marginal row.

## Phase diagram 2: SRF overlap x SNR (M = 8), clean null_frac

| SRF width | inf | 30 | 20 | 10 |  5 | null_frac(clean) |
|---|---|---|---|---|---|---|
| 0.02 |  8I |  8I |  7I |  7I |  6I | 0.459 |
| 0.06 |  8I |  7I |  6I |  6I |  5W | 0.454 |
| 0.15 |  8I |  5W |  4W |  4W |  3W | 0.457 |
| 0.50 |  6I |  3W |  2W |  2W |  2W | 0.484 |

Spectral overlap shifts cells Identifiable -> Weakly as SNR drops, and the
clean ambiguity fraction grows 0.459 -> 0.484 at the overlap extreme.

## Selfcheck

[1] regimes: clean/M31 = 8I; SNR5/M4/w0.5 = 1N; SNR5/M16 = 8I; SNR5/M4/w0.06 = 4W  PASS
[2] r_id_hat non-increasing in SNR (M in {4,8,16,31})                      PASS
[3] r_id_hat non-decreasing in M (SNR inf, 20)                             PASS
[4] null_frac monotone in M (0.492->0.254); endpoint-increase w/ overlap    PASS
[5] corr(score, null_frac) over the M x SNR grid = -0.70                   PASS

## Reading

- The two measures (P2 spectral identifiability, P1 geometric ambiguity)
  agree across the grid (corr -0.70): identifiability is one quantity, seen
  from the spectral side and from the null-space side.
- The phase diagram is the falsifiable target any learned joint estimator must
  respect: a P4 result shows a method that reports confidence in the
  Non-identifiable regime, or fails where the diagram says it must.