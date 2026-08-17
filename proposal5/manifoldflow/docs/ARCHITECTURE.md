# ManifoldFlow — Proposal 5 (Q1 redesign)

Rectified **flow matching** for HSI-MSI fusion on a linear manifold of
consistent reconstructions.  The innovation over plain diffusion flow
matching is the *tangent-space (null-space) constraint*: the velocity field is
projected onto the null space of the observation operator at every step, so
every iterate stays on the consistent set **by construction** (an algebraic
identity, not a penalty).

## Manifold constraint (why the math is exact)

Given the (downsample-and-blur) operator `D` with the range projector
`D_pinv = Dᵀ(DDᵀ)⁻¹`, the **consistent set** is

    { y : D(y) = X }  =  { D_pinv(X) + P_perp(v) }      (any v)

with `P_perp = I − D_pinv D`.  `RangeNullProjector` (shared with the P5
spectralflow degradation head, `proposal5/spectralflow/nullspace.py`) realises
`D_pinv` as one small CG solve in the low-resolution space and `P_perp` as a
linear application.  Because `D(P_perp v) ≡ 0`:

- `D(consistent(x, v)) = X` exactly — data consistency is an **identity**,
  independent of the network output.
- The **tangent space** of the consistent set at any consistent point is
  exactly `null(D)`.  Constraining `dy/dt ∈ null(D)` keeps the entire
  trajectory consistent, so the fused cube is never re-projected ad hoc.

## Flow formulation

- `y₀ = D_pinv(X)` (deterministic, consistent start).
- `y₁ = gt`; because training pairs are synthesised with `X = D(gt)`, the
  target is itself consistent, so `u = y₁ − y₀ ∈ null(D)`.
- Interpolation `y_t = y₀ + t·u` (straight line, the rectified path).
- Loss: `E_t ‖P_perp(v_θ(y_t, m, t)) − u‖²` — flow matching on the tangent
  space (with `t` and the MSI guide `m` as conditioning).
- Sampling: `N` Euler steps `y ← y + (1/N)·P_perp(v_θ(y, m, t))`.

Because the flow is **rectified** (target velocity is constant in `t`), the
field is pushed to be ~constant along the trajectory, and few Euler steps
suffice — the "~10× fewer steps than diffusion" claim.  The straightness
metric `|y_4steps − y_32steps| / |y₀ − gt|` quantifies this directly.

## Ladder (config flags)

| stage | toggle | description |
|---|---|---|
| s1 | `tangent=False` | unconstrained velocity field (plain flow matching) |
| s2 | `tangent=True` | velocity projected onto `null(D)` (consistency preserved) |
| s3 | `straightness_reg=True` | extra `t`-invariance loss `‖v(y_t, t) − v(y_t, t′)‖²` |
| s4 | `sample_steps=4` | rectified few-step Euler sampling |

## Self-checks (CPU, no data)

`python -c "import manifoldflow; manifoldflow.selfcheck()"` verifies:

1. **Tangent**: `max|D(P_perp v)|` and `max|D(consistent(x, v)) − X|` < 1e-3.
2. **Flow matching**: velocity loss drops (>5×) after training on synthetic
   pairs; 4-step and 32-step sampling both land closer to `gt` than `y₀`
   (`err/g₀ < 0.6`).
3. **Straightness**: `|y_4 − y_32| / |y₀ − gt| < 0.4` (few ≈ many steps).
4. **Consistency during sampling**: `max|D(y_k) − X| < 1e-2` for every Euler
   iterate `k` (the tangent constraint holds at every step).
5. **Ladder smoke**: s1–s3 forward/backward, s4 sampling.

## Files

- `config.py` — experiment config (flow, optimisation, observation model).
- `model.py` — `VelocityNet` (conditioned on `y, m, t`) and `ManifoldFlow`
  (`training_step` for flow matching, `sample` for rectified Euler).
- `losses.py` — `FlowLoss` (flow matching + straightness + auxiliary recon).
- `engine.py` — flow-matching training, `evaluate_dataset`, `tiled_inference`.
- `experiments.py` — protocol-compliant comparison tables (wraps shared stats).
- `selfcheck.py` — the five structural self-checks above.
- `results/RESULTS.md` — selfcheck numbers and status.