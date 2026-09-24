# NSP results

Status: **do not cite**.

## Selfcheck (CPU, torch 2.13.0+cpu)
| check | result |
|-------|--------|
| div(g grad u) self-adjoint | rel. 0.0 |
| PDE -> fusion residual (diffusion off) | 9.40e+01 -> 1.03e-01, monotone |
| scale transfer (same weights) | s=2: 8.10e+01 -> 2.16e+01; s=4: 9.47e+01 -> 2.16e+01 |
| ladder smoke (s1..s4) | ALL PASS |

## Ranked comparisons (CAVE/Harvard)
Not yet available. Requires Kaggle training (no data locally). When produced,
follow `PROTOCOL_AUDIT.md`: SAM/ERGAS headline, per-scene LR/SRF-consistency,
paired Wilcoxon, bootstrap CIs, >= 3 seeds.