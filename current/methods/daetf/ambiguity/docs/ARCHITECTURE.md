# P1 scaffold — identifiability-aware fusion (ambiguity package)

The P1 (priority-2) paper restructure: the novelty is **a learned physically
admissible ambiguity manifold `M_theta subset N(A)`** that fills only the
component the observations leave free.  This package is the non-neural,
falsifiable scaffold that the manifold is built on.

## The object: combined operator A = [D; R]

`CombinedOperator` extends `proposal5`'s spatial-only `RangeNullProjector` to
the joint observation operator

    A(X) = [ D(X) ; R^T X ],          A^T[Y_H; Y_M] = D^T Y_H + R Y_M

- `D` = per-band blur + decimate (reused `DegradationOperator`, exact adjoint,
  zero padding so `<Dx,y> = <x,D^T y>` holds exactly);
- `R` (bands x msi_bands) = spectral response; `R^T` maps `(B,H,W) -> (M,H,W)`,
  `R` maps back;
- `D` (spatial, per-band) and `R` (spectral, per-pixel) **commute**, so the
  normal operator is the clean block map

    (A A^T)(z_H, z_M) = ( D D^T z_H + D R^T z_M ,  R D^T z_H + R R^T z_M )

  and the projector needs one block-CG solve in observation space
  (`block_cg`), never on the HR cube.

## Decomposition (structural identity, provable)

    X_obs = A^T (A A^T)^-1 Y          pinned by the data
    X_amb = P_N v = (I - A^T (A A^T)^-1 A) v    genuinely free

with `A(X_obs + X_amb) = Y` for ANY `v`.  Consequences (all verified in the
selfcheck):

- Data consistency is an identity, not a penalty (check 2);
- `X = X_obs + X_amb` and `A(X_amb) = 0` (check 3);
- the observable component `X_obs` is invariant to any null-space perturbation:
  `E_obs` is identical across reconstructions with different null content
  (check 4, spread 4e-6).

## Metrics (P1 headline quantities)

- `hallucination`: `H = ||P_N(X_hat - X)|| / (||P_N X|| + eps)` — normalized
  error on the genuinely-ambiguous component.  0 = null content reproduced,
  ~1 = fabricated/dropped wholesale.  Verified monotone 1.0 -> 0.5 -> 0.0 for
  zero / half / full null-space guesses.
- `error_split`: `E_obs`, `E_amb` — observable vs ambiguous error.
- `uncertainty_map`: `U(x,y) = ||P_N X||` per pixel, `U(lam)` per band — where
  the fusion must guess.
- Verified: `corr(U(x,y), E_amb(x,y)) = 1.0` — the ambiguity map predicts where
  a null-space-ignorant reconstruction fails (check 5).

## Next stage (M_theta, not this package)

The learned manifold `M_theta` is built on this scaffold: a network generates
`v` and is re-projected with `P_N` at every step (as ManifoldFlow already does
with spatial D), while `A^T(A A^T)^-1 Y` is added in closed form.  The
hallucination metric and ambiguity map above are the measurements that decide
whether `M_theta` actually reduces ambiguity or merely memorizes it.