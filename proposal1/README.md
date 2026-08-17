# proposal1 — P1 paper: admissible ambiguity manifold (priority 2)

**Question:** what does an HSI–MSI fusion method actually know from the data,
and what does it invent?  **Object:** a learned, physically admissible
ambiguity manifold `M_theta subset N(A)`.

## Present work — `ambiguity/` (non-neural scaffold, selfcheck ALL PASS)

The falsifiable base the learned manifold is built on, extending
`proposal5/spectralflow`'s spatial-only `RangeNullProjector` to the joint
operator `A = [D; R]`:

- `operator.py` — `CombinedOperator`: exact adjoint, block normal operator
  `(A A^T)(z_H,z_M)`, one block-CG solve in observation space; projectors
  `project_range` / `project_null` / `consistent`.
- `metrics.py` — observable/ambiguous split, hallucination
  `H = ||P_N(X̂−X)||/(||P_N X||+ε)`, `E_obs`/`E_amb`, ambiguity map
  `U(x,y) = ||P_N X||` per pixel, per-band `U(λ)`.
- `selfcheck.py` — adjoint 3.3e-7; joint consistency `A(consistent)=Y` 8.7e-6;
  decomposition 2.3e-5; `H` monotone 1.000/0.500/0.000 with `E_obs` spread
  4e-6; ambiguity map predicts null-ignorant failure corr = 1.0000.
- `docs/ARCHITECTURE.md`, `results/RESULTS.md`.

**The structural identity:** `A(X_obs + X_amb) = Y` for any `X_amb` — data
consistency is algebraic, and the model only ever fills the ambiguous
component.  `M_theta` (the network stage) is built on this scaffold.

```powershell
$env:PYTHONPATH="common;proposal1;proposal5"; python -c "import proposal1.ambiguity as a; a._selfcheck.run_all()"
```

## Legacy (prior Q1 architectures, selfchecked, superseded)

| package | architecture | mechanism |
|---|---|---|
| `daetf/` | DAETF-Net | projective spectral embedding + range/null consistency |
| `slt/` | SLT | self-learning transformer |

`docs/` holds the historical architecture notes (`Q1_REDESIGN.md`,
`ARCHITECTURE_v1_original.md`, `ANALYSIS.md`, `GAP_ANALYSIS.md`).