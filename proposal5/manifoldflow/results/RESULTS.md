# ManifoldFlow — Results (scaffold)

Status: **structural self-checks ALL PASS** (CPU).  No real training yet —
local machine is CPU-only and the CAVE/Harvard data are not available; real
runs go to Kaggle.

## Selfcheck numbers (seed 0)

| check | metric | value | threshold | status |
|---|---|---|---|---|
| tangent | max\|D(P_perp v)\| | 7.1e-05 | < 1e-3 | PASS |
| tangent | max\|D(consistent(x, v)) − X\| | 6.9e-05 | < 1e-3 | PASS |
| flow matching | velocity loss | 0.0964 → 0.0014 (×71.2) | > 5× | PASS |
| few-step | \|y_4 − gt\| / \|y₀ − gt\| | 0.520 | < 0.6 | PASS |
| few-step | \|y_32 − gt\| / \|y₀ − gt\| | 0.523 | < 0.6 | PASS |
| straightness | \|y_4 − y_32\| / \|y₀ − gt\| | 0.147 | < 0.4 | PASS |
| consistency | max\|D(y_k) − X\| (k = 0..7) | ≤ 1.6e-05 | < 1e-2 | PASS |
| ladder | s1–s3 fwd/bwd, s4 sampling | — | — | PASS |

## Claims validated

- Data consistency is an algebraic identity: every Euler iterate satisfies
  `D(y_k) = X` to CG precision, regardless of the network.
- The flow is rectified: 4 Euler steps ≈ 32 steps (straightness 0.147),
  supporting the "~10× fewer steps" claim.
- Flow matching converges: the velocity field learns the constant target
  `u = gt − y₀` (>70× loss reduction).

## Next (Kaggle)

- Train on CAVE/Harvard patch pairs, evaluate with the protocol (per-scene
  LR/SRF-consistency, paired Wilcoxon + bootstrap CIs, ≥3 seeds).
- Compare `sample_steps ∈ {1, 2, 4, 8}` for the few-step trade-off.
- Blind operator variant: replace the fixed projector kernel with the
  spectralflow degradation-head estimate.