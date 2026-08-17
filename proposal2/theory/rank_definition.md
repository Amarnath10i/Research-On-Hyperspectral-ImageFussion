# Rank definitions for P2

The fusion problem writes the HR-HSI as a low-rank spectral model `X ≈ U_r Z`
with `U_r` the (B x r) spectral basis and `Z` the (r x HW) spatial coefficients.
"Rank" is ambiguous; the paper distinguishes four notions.

## 1. Matrix rank — rank(X)

The exact linear-algebraic rank of `X` viewed as a B x HW matrix.  Degenerate
for real data: real scenes are never exactly low rank, and this notion carries
no information about *observability*.

## 2. Numerical rank — r_epsilon

The number of singular values needed to capture a `1-epsilon` fraction of the
singular-value energy:

    r_epsilon = min{ r : sum_{i=1}^{r} sigma_i^2 / sum_i sigma_i^2 >= 1 - epsilon }.

Used internally as a conservative cap (e.g. energy 0.9-0.995) but not the
headline quantity: it is scale- and SNR-dependent and, under noise, drifts
upwards with the noise floor.

## 3. Material rank

The number of independent endmember/material spectra present in the scene.
Physically meaningful but not directly observable; requires a spectral
unmixing assumption (linear mixing model) and is tied to P4.

## 4. Observation-identifiable rank — r_id (the novelty)

The number of spectral degrees of freedom of the HR-HSI that the observation
pair actually supports:

    r_id = rank( R^T U_r ),

i.e. the rank of the *projected* spectral basis through the MSI's spectral
response R.  Equivalently, the rank of the HR-MSI's spectral subspace
`Y_M = R^T X`.  Directions of `U_r` in the kernel of `R^T` are observable only
at the LR-HSI's spatial resolution; they cannot be resolved at HR, so they are
**not** counted as identifiable.

`r_id` is bounded by `min(r, M)` and degrades with SRF overlap (P2
experiments), with the MSI band count, and with noise (the *estimated*
identifiable rank).

### Why r_id is the right object

For the combined observation operator `A = [D; R]`, two reconstructions are
observationally indistinguishable iff `A(X_1) = A(X_2)` (equivalence relation,
see identifiable_rank.md).  The number of spectral degrees of freedom that are
*pinned down by the observations* is exactly the dimension of the projected
subspace `R^T U_r`.  Reporting `|r_hat - r_id|` (not `|r_hat - rank(X)|`)
separates "what the scene is" from "what the sensors can see".