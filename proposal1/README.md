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
- `auditor_selfcheck.py` — **the differentiated P1 contribution: a
  method-agnostic *ambiguity auditor*.** Given *any* fusion output, `H` reports
  what it invented and `U` localises where.  Proof: `H(oracle)=0.000`,
  `H(partial)=0.500`, `H(range_only)=1.000`, `H(wrong)=1.215`; `corr(H, SAM)`
  across methods `= 0.973` (H is a usable proxy for spectral error);
  `corr(U_map, true null per pixel) = 1.000` (U localises the unobservable
  content).  This is what makes P1 a general diagnostic, not another fusion net.
- `docs/ARCHITECTURE.md`, `results/RESULTS.md`.

**The structural identity:** `A(X_obs + X_amb) = Y` for any `X_amb` — data
consistency is algebraic, and the model only ever fills the ambiguous
component.  `M_theta` (the network stage) is built on this scaffold.

> **Repositioning note.** The original `daetf/` fusion net scored *below*
> bicubic/GSA, so P1's contribution is reframed as the auditor above (apply
> `H`/`U` to *existing* SOTA), not as a competing fusion method. See
> `docs/Q1_REDESIGN.md`.

```powershell
$env:PYTHONPATH="common;proposal1;proposal5"; python -c "import proposal1.ambiguity as a; a._selfcheck.run_all()"
$env:PYTHONPATH="common;proposal1;proposal2;proposal5"; python -c "import proposal1.ambiguity.auditor_selfcheck as au; au.run_all()"
```

## Legacy (prior Q1 architectures, selfchecked, superseded)

| package | architecture | mechanism |
|---|---|---|
| `daetf/` | DAETF-Net | projective spectral embedding + range/null consistency |
| `slt/` | SLT | self-learning transformer |

`docs/` holds the historical architecture notes (`Q1_REDESIGN.md`,
`ARCHITECTURE_v1_original.md`, `ANALYSIS.md`, `GAP_ANALYSIS.md`).