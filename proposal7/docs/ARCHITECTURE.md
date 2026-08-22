# NullFusion — Architecture

## The decomposition the network is built on
`proposal1.ambiguity.operator.CombinedOperator` gives `A = [D; R]` with an exact
adjoint and the joint range/null projector.  For any cube `X`,

```
X = A^T (A A^T)^-1 A X          +   (I - A^T(A A^T)^-1 A) X
    \________________________/        \________________________/
    X_obs  (range / observable)       X_amb  (null / ambiguous)
```

and for *any* `v`,  `A(X_obs + P_N v) = A X_obs = A X` — the null component is
invisible to the sensors.  NullFusion makes the network predict only `P_N v`.

## Modules
```
yH (LR-HSI) ─┐
             ├─ hsi_stem ─┐
yM (HR-MSI) ─┘            ├─ fuse ─ enc(ResBlocks, LR grid) ─ upsample ──┐
             msi_stem ────┘                                            │
yM ─ msi_detail ───────────────────────────────────────────────────────┤
Xobs = pinv(yH,yM) ────────────────────────────────────────────────────┤
                                                                         ├─ concat
                                                                         │
                                            f_theta (ResBlocks + attn) ──┤
                                                                         │
                                            spectral rank bottleneck ────┤
                                                                         │
                                            P_N (project_null) ──────────┘
out = Xobs + P_N( f_theta( cond ) )
```

### Range component `Xobs = pinv(yH, yM)`
One block-CG solve in observation space (`CombinedOperator.pinv`).  Exact, no
parameters.  This is the part every other method *learns* and gets wrong.

### Conditioning
- `hsi_stem` + `msi_stem` (pooled to the LR grid) → `fuse` → `enc` ResBlocks on
  the LR grid → transposed-conv upsample to HR.  Carries the hyperspectral
  context where it lives (the LR grid).
- `msi_detail`: full-resolution MSI features (the high-frequency spatial guide).
- `Xobs` itself is concatenated so `f_theta` sees the data-determined base and
  only refines the residual.

### Null-space prior `f_theta`
Residual conv blocks + optional spatial self-attention on the HR grid, predicting
`v` (B, bands, H, W).  `v` is then forced into the null space with
`project_null` and added to `Xobs`.

### P2 spectral low-rank bottleneck
If `cfg.rank < bands`, `v` is squeezed through a `1x1` `bands → rank → bands`
bottleneck.  **Crucially** the null space has `bands - r_id` dimensions, not
`r_id`, so the SOTA run leaves `rank = bands` (full capacity).  The bottleneck is
an *optional* P2-regularised variant (`rank = bands - r_id`) used in the
ablation ladder; the `r_id` estimate is instead the auditable bound reported via
the H metric.

### P3 sensor conditioning
The SRF is a buffer on `A`; calling `set_srf(nikon_d700_srf(bands))` (or an
estimated/domain-shifted response) re-derives the exact consistent set with no
code change.  Cross-domain robustness is then a property of the operator, not a
retraining step.

## Exactness proof sketch (verified numerically)
`out = A^T(A A^T)^-1 Y + (I - A^T(A A^T)^-1 A) v`
`A(out) = A A^T(A A^T)^-1 Y + (A - A A^T(A A^T)^-1 A) v = Y + 0 = Y`.
The `selfcheck` confirms `D-err, S-err ~ 1e-5` on CPU.

## Training
- Loss: `L1(out, GT) + w_ssim·SSIM(out,GT) + w_sam·SAM(out,GT)`.  No
  data-consistency term (it is free).
- The hallucination auditor: `H = ||P_N(out - GT)|| / ||P_N GT||` is reported
  every eval — it is the paper's quantitative claim that NullFusion invents less
  than bicubic/GSA/transformer baselines.
- For SOTA: configure for CAVE ×4 Nikon (see `SOTA_COMPARISON.md`), set
  `rank` from P2, train on Kaggle (T4 ×2) with the budget in
  `proposal2/notebooks/krylovnet_SOTA_CAVE_Nikon.ipynb`.
