# NSP — Neural Spectral PDE — Proposal 3 (Q1 redesign)

## Status
Scaffold complete. Structural claims verified on CPU (see `selfcheck`). No
trained checkpoints yet; real training and rankings run on Kaggle.

## What it is
Fusion as the steady state of a **diffusion–reaction PDE** over the spectral
volume:

    du/dt = div(D(u) grad u) - lam1 D^T(D u - X) - lam2 S^T(S u - M)

- `div(D(u) grad u)` is the diffusion term with a **learned cross-spectral
  tensor**: per-pixel, per-band in-plane diffusivity `gxy` plus a band-axis
  diffusivity `gz` that couples adjacent spectral bands (the whole HSI cube is
  treated as a 3D diffusion medium).
- The observation terms are soft **penalties** (gradient of the least-squares
  energies), so the steady state minimises
  `E(u) = int g |grad u|^2 + lam1/2 ||D u - X||^2 + lam2/2 ||S u - M||^2`.
  This is the honest limitation of the design: consistency is in the penalty
  sense, not hard constraints, so residuals are small but not zero.
- The PDE is a continuous-space model; the discretisation step `dt` is the only
  scale-dependent quantity (explicit Euler, `dt *= 1/scale` when
  `scale_adaptive_dt`). The learned tensor and penalties do not encode scale.
- Finite differences: forward-difference gradients with zero-flux (Neumann)
  boundaries and the exact adjoint divergence, so the diffusion operator is
  exactly self-adjoint (`<div(g grad u), v> == <u, div(g grad v)>`, verified to
  machine precision).

## Ladder
| stage | dynamics | learned pieces |
|-------|----------|----------------|
| s1 | pure penalty gradient descent | none (baseline) |
| s2 | + learned scalar dt, lam1, lam2 | scalars |
| s3 | + cross-spectral diffusion tensor | + CNN tensor (3D coupling) |
| s4 | + scale-adaptive discretisation step | as s3, dt ∝ 1/scale |

## Verified claims (CPU selfcheck)
1. **Adjointness**: the discretised `div(g grad .)` is exactly self-adjoint
   (rel. error 0.0).
2. **PDE → fusion**: with diffusion off, the penalty dynamics drive the
   normal-equation residual from 9.4e+01 to 1.0e-01 (monotone, no violations).
3. **Scale transfer**: the same learned weights reduce the residual at scale 2
   (8.1e+01 → 2.2e+01) and scale 4 (9.5e+01 → 2.2e+01) with only `dt`
   adapting.
4. **Ladder smoke**: every stage builds, forwards and backprops; gradients reach
   the tensor net and the learned dt/penalty scalars.

## Honest limitations
- Soft-penalty consistency only: the steady state balances observation errors
  against the diffusion prior; residuals are small but never zero. If the task
  demands hard consistency, the PDE must be replaced by a constrained solve
  (that is KrylovNet's lane, P2).
- Explicit Euler is only conditionally stable; too large `dt` (or too strong a
  learned tensor) diverges. `pde_steps` is a capacity/compute knob and the
  forward is `pde_steps` operator evaluations, so it is comparatively slow.
- The tensor net is re-evaluated every step; it has no memory of the scene
  beyond the current iterate.

## Run
    python -c "import nsp; nsp.selfcheck()"     # structural checks
    python -c "import nsp; nsp.train(cfg)"      # on Kaggle