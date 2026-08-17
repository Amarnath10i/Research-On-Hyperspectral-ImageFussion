# proposal5 — null-space generative models (P1's learned stage groundwork)

**Question:** can a generative model fill the ambiguous component of the
fusion problem without breaking data consistency, and be *shown* to do so?

## Present work — `manifoldflow/` (selfcheck ALL PASS)

The learned-stage groundwork for the P1 paper's admissible ambiguity manifold:
a rectified flow whose velocity field lives on the null space of the (estimated)
observation operator, so every Euler step stays on the consistent set.

- `model.py` — `VelocityNet` (t as scalar or batch tensor; first conv
  `in_ch = 2·bands+1` after the MSI→band projection).
- `engine.py` — velocity projected with `RangeNullProjector` at every step.
- `selfcheck.py` — ALL PASS: tangent `D(P_perp v) ≈ 7e-5`; flow loss
  0.0964 → 0.0014 (×71.2); 4-step vs 32-step reconstruction error 0.520 vs
  0.523; straightness 0.147; consistency `D(y_k)=X ≤ 1.6e-5` every step.

## Core reusable — `spectralflow/nullspace.py`

`RangeNullProjector` (`D_pinv = D^T (DD^T)^-1`, `P_perp = I − D_pinv D`, one CG
solve in LR space) is the operator primitives the whole program reuses — the
P1 paper extends it to the combined `A=[D;R]` in `proposal1/ambiguity`.

## Legacy — `spectralflow/` (prior Q1 architecture, selfchecked)

Score-based posterior sampling on the null space of the estimated observation
operator (`SpectralFlow`).

```powershell
$env:PYTHONPATH="common;proposal5"; python -c "import manifoldflow; manifoldflow._selfcheck.run_all()"
$env:PYTHONPATH="common;proposal5"; python -c "import spectralflow; spectralflow._selfcheck.run_all()"
```