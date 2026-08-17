# P1 scaffold — results (proposal1/ambiguity/selfcheck.py)

Status: **ALL PASS** (CPU, tiny synthetic scene: B=6, M=3, 16x16 HR, scale 2,
250 block-CG steps, ridge 1e-8).  No network anywhere; every check is against
the operator algebra of `A=[D;R]`.

## Checks

| # | claim | value | verdict |
|---|-------|-------|---------|
| 1 | `<A x, y> = <x, A^T y>` | rel err 3.3e-07 | PASS |
| 2 | `A(consistent(y, v)) = y` for arbitrary v | max rel 8.7e-06 | PASS |
| 3 | `X = X_obs + X_amb`, `A(X_amb) = 0` | 2.3e-05 / 7.5e-07 | PASS |
| 4 | H base/half/full = 1.000/0.500/0.000; `E_obs` spread | 4.0e-06 | PASS |
| 5 | `corr(U(x,y), E_amb(x,y))` for null-ignorant baseline | 1.0000 | PASS |

## What these numbers mean for P1

- The joint `A=[D;R]` projector is exact to solver precision: adding an MSI
  sensor does not break the consistency identity that the spatial-only
  projector already satisfied.
- The hallucination metric behaves as specified (monotone with null-space
  accuracy), and the observable-component error is *structurally* invariant to
  null-space perturbations (spread 4e-6) — the claim the paper's comparison
  ladder depends on.
- The ambiguity map `U(x,y) = ||P_N X||` exactly predicts where a
  null-space-ignorant reconstruction errs on this controlled scene
  (corr = 1.0 by construction: the baseline's ambiguous error IS `P_N X`).

## Caveat / next

The correlation is 1.0 here because the baseline drops all null content, so
its per-pixel error literally equals the ambiguity map.  The interesting
(later, empirical) questions — does a learned `M_theta` beat `A^T Y + P_N F_theta`
at reducing `H`, and does `U` still predict `E_amb` when the null-space guess
is a learned function rather than zero — require the network stage on Kaggle.