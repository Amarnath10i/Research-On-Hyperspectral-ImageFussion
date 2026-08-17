# KrylovNet — Proposal 2 (Q1 redesign)

## Status
Scaffold complete. Structural claims verified on CPU (see `selfcheck`). No
trained checkpoints yet; real training and rankings run on Kaggle.

## What it is
Fusion posed as the normal equation of the two observation models and solved by
a **differentiable unrolled GMRES**:

    A x = b,   A = D^T D + S^T S + rho I,   b = D^T X + S^T M

- `D`: LR-HSI operator (blur + decimate), `S`: SRF-to-MSI operator. Both use
  zero padding so `D^T` / `S^T` are **exact adjoints** (`conv_transpose` /
  `einsum` transpose) — verified to ~1e-7.
- The unrolled solver grows the Krylov basis one orthonormalised vector per
  stage (Arnoldi, per-sample modified Gram-Schmidt) and re-solves the
  residual-minimising combination over the growing subspace (pseudo-inverse
  least squares on the small Hessenberg system). The subspace grows
  monotonically, so the residual is non-increasing.
- Learned pieces sit *inside* the solver, not in a spatial encoder:
  - `Blend`: an attention over basis-vector norms that blends the pure GMRES
    combination with a learned one (`alpha` starts near zero).
  - `SpectralPreconditioner`: a GNN over the **spectral band graph** (nodes =
    HSI bands, kNN edges from per-band LR-HSI statistics) that emits a positive
    per-band scale; used as a left preconditioner (`P^-1 A`).
  - `Hypernet` (optional, stage 5): reads a conditioning proxy (initial/current
    residual ratio) and gates the blend strength per stage.

## Ladder
| stage | solver | learned pieces |
|------|--------|----------------|
| s1  | Richardson fixed point | none (baseline) |
| s2  | Krylov (GMRES) | none |
| s3  | Krylov | attention blend |
| s4  | Krylov | blend + spectral preconditioner |
| s5  | Krylov | blend + preconditioner + hypernet |

## Verified claims (CPU selfcheck)
1. **Adjointness**: `D/D^T` and `S/S^T` satisfy `<Dx,y>=<x,D^Ty>` to 1e-7.
2. **Exact solve**: on a 32x32 SPD normal-equation system the unrolled GMRES
   reaches residual 6.6e-13 (float64) with a monotone non-increasing residual
   and solution error 8.5e-15.
3. **Preconditioner helps**: after ~150 steps of fitting a diagonal target, the
   spectral GNN reduces `cond(P^-1 A)` from 2.48 to 1.08 (< 0.5 x cond(A)).
4. **Stages help**: 6 stages beat 1 (normal-equation residual 1.7 vs 4.7) and
   Krylov beats the fixed-point baseline at equal stages (1.7 vs 9.8).
5. **Ladder smoke**: every stage builds, forwards and backprops; gradients reach
   the blend and the preconditioner.

## Honest limitations
- The GMRES residual is measured in the (preconditioned) operator norm; the
  physics/spectral *energy* terms only decrease in tandem once the normal
  equation is near-solved, and are not individually guaranteed monotone.
- The preconditioner and blend are small (no spatial capacity); on hard scenes
  they may underfit. The old UnfoldFusion encoder is intentionally absent —
  capacity must come from the solver.
- Per-sample Arnoldi orthonormalisation is memory-light but the solver cost
  grows linearly in stages; stage count is a capacity/compute knob.
- Zero padding for D/S deviates from `blur_downsample`'s reflect padding at
  image borders (needed for an exact adjoint); effect is boundary-only.

## Run
    python -c "import krylovnet; krylovnet.selfcheck()"     # structural checks
    python -c "import krylovnet, proposal2.krylovnet.experiments as e; ..."
    python -c "import krylovnet; krylovnet.train(cfg)"      # on Kaggle
