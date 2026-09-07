# GraphDIP results

Status: **do not cite**.

## Selfcheck (CPU, torch 2.13.0+cpu)
| check | result |
|-------|--------|
| superpixels (seeded k-means) | full, non-empty 8-partition |
| linear mixing is linear | rel. 2.3e-07 |
| nonlinear / attention mixing | linearity violation 0.58 / 0.57 |
| physics-only DIP (synthetic scene) | 3.19 -> 0.034 (ratio 0.011) |
| ladder smoke (s1..s4) | ALL PASS |

## Ranked comparisons (CAVE/Harvard)
Not yet available. Per-scene DIP runs on Kaggle. When produced, follow
`PROTOCOL_AUDIT.md`: SAM/ERGAS headline, per-scene LR/SRF-consistency,
paired Wilcoxon, bootstrap CIs, >= 3 seeds.