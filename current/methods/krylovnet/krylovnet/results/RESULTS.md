# KrylovNet results

Status: **do not cite**.

## Selfcheck (CPU, torch 2.13.0+cpu)
| check | result |
|-------|--------|
| adjoint D/D^T, S/S^T | 9.5e-07 / 3.2e-07 |
| exact solve (32x32 SPD, m=32) | residual 6.6e-13, rel err 8.5e-15 |
| preconditioner cond(A) 2.48 -> cond(P^-1 A) 1.08 | PASS |
| 6 stages vs 1 (residual 1.73 vs 4.73); Krylov vs Richardson (1.73 vs 9.76) | PASS |
| ladder smoke (s1..s5) | ALL PASS |

## Ranked comparisons (CAVE/Harvard)
Not yet available. Requires Kaggle training (no data locally). When produced,
follow `PROTOCOL_AUDIT.md`: SAM/ERGAS headline, per-scene LR/SRF-consistency,
paired Wilcoxon, bootstrap CIs, >= 3 seeds.
