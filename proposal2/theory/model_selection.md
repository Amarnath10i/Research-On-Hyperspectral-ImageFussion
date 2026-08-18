# P2 theory note — model-selection lower bound on spectral error

## Setup
Scene `X ∈ R^(B×N)`, spectral basis `U ∈ R^(B×r)`, spectral coefficients
`Z = UᵀX ∈ R^(r×N)`.  Sensor `R ∈ R^(M×B)`, `M ≤ r`.  Define the
**scene-aware observable projector**

```
G := RᵀU ∈ R^(M×r),   P_row := Gᵀ(GGᵀ)⁻¹G,   P_null := I_r − P_row.
```

`P_row` is the orthogonal projector onto the row space of `G` (the spectral
directions the observations can resolve); `P_null` projects onto the **spectral
null space** — the directions of `X` that no observation can constrain.

> **Why `G = RᵀU`, not `R` alone.** `R` alone measures SRF capacity; the
> scene enters through `U`.  A direction `u_k` is observable only if `Rᵀu_k` is
> non-degenerate.  Using `Rᵀ` (without `U`) conflates sensor capacity with
> scene identifiability and over-states what the data can pin down.  This is the
> bug that the original range-null `‖(I−R†R)X‖` formulation had.

## Theorem (optimality floor)
Let `A = [D; R]` be the forward operator and let `X̂` be *any* reconstruction
that reproduces the observations (`A X̂ = A X`) — i.e. an estimator that invents
nothing beyond what the data forces.  Then

```
‖X̂ − X‖_F / ‖X‖_F  ≥  ‖U P_null Z‖_F / ‖X‖_F .
```

The bound is **achieved** by `X̂_id = U P_row Z` (the observable part of `X`),
which is the minimum-norm observation-consistent reconstruction.

### Proof
`A X̂ = A X` ⟹ `X̂ − X ∈ ker A`, so in the spectral basis
`P_row(Ẑ − Z) = 0` (the observable coefficients are fixed) and only
`P_null Z` is free.  Hence `X̂ − X = U P_null(Ẑ − Z)` and

```
‖X̂ − X‖² = ‖U P_null(Ẑ − Z)‖² ≥ ‖U P_null Z‖²   (since the choice
            Ẑ = 0 minimises the norm over the free component),
```

with equality at `Ẑ = 0`, i.e. `X̂ = U P_row Z`. ∎

The floor is reported by `model_selection.spectral_null_fraction`.

## Model-selection rule
Two rank estimates are available:
- `r_hat` — the true scene rank (estimated from the HSI trailing spectrum);
- `r_id_hat = rank(G)` — the observation-identifiable rank (P2's core object).

Any method that reconstructs `min(r_hat, r_id_hat)` spectral directions cannot
recover the remaining `max(0, r_hat − r_id_hat)` directions, and the minimum
error it can incur equals the floor above.  Hence we set

```
r* := min(r_hat, r_id_hat)
```

and the achievable floor is a property of `(U, R)` only — independent of the
fusion architecture.  Methods that claim to recover beyond `r_id_hat` are
necessarily hallucinating the null-space component; `proposal1`'s hallucination
metric `H` quantifies exactly how much.

## Self-check evidence
`model_selection_selfcheck.py`: for a rank-12 scene with `r_id_hat = 8`,
`r* = 8`, the rank-`r*` reconstruction achieves `err = floor = 0.5712`
(exactly the bound); an oracle that knows the true null recovers `err = 0`;
an over-ranked method using a wrong null prior has `err = 0.803 > floor`; an
under-ranked method has `err = 0.805 > floor`.
