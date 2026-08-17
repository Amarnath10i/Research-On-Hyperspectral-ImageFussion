# SpectralFlow: Null-Space-Consistent Score-Based HSI-MSI Fusion

> **Status: implemented and self-checked.** Every mechanism in this document
> exists in `../spectralflow/` and is verified numerically by `selfcheck.py`
> (a few seconds on CPU).  The checks verify the *algebraic* claims — most
> importantly that `D(Y_hat) = X` holds after sampling even with an untrained
> network.  What is **not** yet claimed is a ranking: no full CAVE/Harvard
> training run has produced comparable numbers yet, so the ranking questions
> at the end of this document are still open.

## 1. The problem, restated

HSI-MSI fusion is the inversion of two coupled observation models:

```
X = D(Y)      LR-HSI:   blur then decimate            (spectral identity)
M = S(Y)      HR-MSI:   spectral response projection  (spatial detail)
```

The benchmark in this repository shows that ten published methods hold spatial
quality while *spectral* fidelity collapses under domain shift (SAM 2–7 deg
in-domain to 8–58 deg cross-domain, ERGAS up to two orders of magnitude).
Root causes, in order:

1. **Ill-conditioning.** Many HR cubes explain the same LR-HSI.  A network that
   is asked to predict the whole cube spends most of its capacity reproducing
   the part the data already determines, and the rest is unconstrained by the
   observation.
2. **The learned model is the domain.**  A discriminative regressor trained on
   synthetic CAVE has to generalise its prediction error to real Harvard.  When
   the sensor statistics shift, that error shifts with it — there is nothing in
   a supervised regressor that stops it violating the target observation.
3. **Intensity masks spectral error.**  Per-image intensity differences dominate
   pixelwise metrics, which is why PSNR *improves* while SAM degrades.

## 2. The proposed formulation (the contribution)

Treat the LR observation as the *constraint set* and the HR-MSI as the *guide*,
and solve fusion as posterior sampling restricted to the null space of the
observation operator:

```
Y_hat = D_pinv(X)  +  P_perp( Y_draw )
        \______/        \______________/
       fixed by          drawn from a score-based
       the data          spectral-spatial prior,
       (closed form)     guided by M and d
```

* `D_pinv = D^T (D D^T)^{-1}` — the range component, computed exactly in
  closed form by CG in low-resolution space.  **Never learned.**
* `P_perp = I - D_pinv D` — the orthogonal projection onto the null space of D.
* `Y_draw` — a sample from a multispectral-guided score-based model of the
  HR spectral-spatial manifold, conditioned on the HR-MSI guide `M` and a
  degradation code `d` estimated by a blind operator encoder.

Because `D P_perp = 0`, the reconstruction satisfies `D(Y_hat) = X` **exactly,
for any sample the generative model produces.**  Data consistency is an
algebraic identity; the sampler re-asserts it at every reverse step by mapping
each estimate back onto the consistent set `D_pinv(X) + P_perp(·)`.

Three properties follow:

1. **Consistency by construction, not by penalty.**  A diffusion model that
   generates freely and hopes the observation is satisfied, or a residual
   network that penalises mismatch in its loss, both *negotiate* with the data.
   SpectralFlow cannot violate it: the architecture gives the sample no way to
   express a violation.  `selfcheck` verifies this with an *untrained* network.
2. **The generative model only learns what the data does not determine.**  The
   dimension of the generative problem is cut by the observation: the prior
   never has to reproduce the component that is computed in closed form.  This
   is the reduction that makes a modest score network sufficient.
3. **Domain robustness is structural.**  Whatever the score prior learned on
   CAVE, the *range* component on a Harvard scene is taken from Harvard's own
   measurement.  A wrong prior can only distort the null-space detail; it
   cannot corrupt the spectral identity the sensor actually saw.

## 3. How it differs from the closest existing work

Positioning must be precise, or a reviewer will (correctly) call this
incremental:

| Existing idea | What it does | What SpectralFlow adds |
|---|---|---|
| Physics **loss** terms (this repo's P1, and many TGRS papers) | penalise `D(ŷ) ≈ X` | enforces `D(ŷ) = X` in the **sampler**, algebraically; the network never predicts the range |
| Residual/back-projection fusion | learn `ŷ = up(X) + r` | splits along the *null space*; the residual can never overwrite the measurement |
| DDNM (ICLR 2023), null-space diffusion | range/null split for generic inverse problems with a **known** operator | the two-observation HSI-MSI setting with an **estimated and per-scene refined** operator, a *multispectral-guided* spectral prior, and the claim is *spectral* fidelity under domain shift |
| Diffusion pansharpening / HSI diffusion SR | unconditional or soft-guided generation | hard per-step consistency projection; only the null-space component is sampled |

The falsifiable hypothesis, stated so it can lose:

> Compared with (a) an equally sized unconstrained residual fusion, (b) a
> matched physics-loss model, and (c) the same score network without the
> projection, null-space-projected score-based fusion reduces LR-consistency
> error to solver tolerance while improving SAM and ERGAS — especially when
> blur/noise/SRF shift between source and target sensors.

## 4. Architecture

```
LR-HSI X ──► Degradation Head ──► (code d, kernel D_hat)
    │                                   │
    ├──────────────────────────────► D_pinv(X) = range        (closed form, CG)
    │                                                            │
    └──► score network  eps_theta(y_t, M, d, t)                 │
HR-MSI M ──► concatenated at the stem (spatial detail)          │
                    │                                           │
              reverse diffusion, every step:                    │
                  x0_est = (y_t - sqrt(1-a_t) eps)/sqrt(a_t)    │
                  x0     = D_pinv(X) + P_perp(clamp(x0_est))    │
                    │                                           │
                    └──────────────────────────────────► Y_hat = x0
```

Components:

* **`score.py`** — `SpectralScoreNet`, a compact FiLM-conditioned U-Net
  (GroupNorm + SiLU residual blocks, channel-multipliers `(1,2,2,2)`).
  Conditions: time embedding, degradation code, and the HR-MSI concatenated at
  the stem.  Input channels = `bands + msi_bands`.
* **`sampler.py`** — `DiffusionSampler`, deterministic DDIM reverse (`eta=0`,
  `sample_steps` configurable) with the per-step null-space projection.  The
  projector runs one small CG solve in LR space per step; the range component
  `D_pinv(X)` is computed once per scene and cached.  The `use_projection`
  switch (off ⇒ plain DDIM) is what makes the critical Stage 1 vs Stage 2
  experiment runnable **on a single checkpoint**: projection is an inference
  behaviour, not part of training, so one trained score net serves both arms.
* **`nullspace.py`** — `DegradationOperator` (exact adjoint, zero padding),
  `RangeNullProjector`, CG solver.  Same formulation as P1 but self-contained.
* **`losses.py`** — denoising score matching (MSE on `eps`), plus a
  degradation-regression term so the code is physical, plus a light
  SRF-consistency term so the MSI channel is load-bearing.
* **`engine.py`** — `train` (score matching only, no sampling), `SamplingModel`
  (the protocol-facing wrapper: sampling *is* the forward pass), the
  `set_projection()` / `set_msi_guide()` toggles that drive the ladder, and
  `evaluate_dataset` with per-scene **LR-consistency** reporting and optional
  per-scene **operator refinement**.

### Blind setting: joint operator estimation and refinement

The degradation head estimates the blur kernel from `(X, M)`.  Because the
consistency identity holds with respect to whichever `D` is supplied, an
*erroneous* estimate still gives `D_hat(Y_hat) = X` — but against the *true*
sensor the guarantee weakens.  SpectralFlow therefore refines the operator per
scene against the physics of its own sample:

```
min_{params}  || D_k(Y) - X ||^2  +  w_spec || S(Y) - M ||^2
```

alternating sample → refine → re-sample (`refine_rounds`, off by default).
This is the generative analogue of test-time adaptation: no target labels, only
the target scene's own observations.

## 5. Protocol and required experiments

Same rules as the rest of the repository: one scale factor, one metric
implementation with `data_range=1.0`, one fixed eval degradation, full scenes
via Hann-tiled inference, per-scene rows retained for paired tests.

| Stage | Model | Question answered |
|---|---|---|
| 0 | Bicubic, GSA, HySure/CNMF | Is the protocol correct? (run these first) |
| 1 | Score net without projection (plain DDIM) | Does the generative prior learn the manifold? |
| 2 | Stage 1 + per-step null-space projection | Does the projection help, hurt, or neither? |
| 3 | Stage 2 + MSI conditioning off | Is the guide load-bearing? |

Stages 1–3 are implemented as `STAGE_LADDER` in `experiments.py`: the two
switches (`use_projection`, `use_msi_guide`) flip inference behaviour on the
**same** checkpoint, so the ladder costs one training run.  `run_stage_ladder`
produces a paired stage-vs-stage comparison with Wilcoxon / Cohen's d / bootstrap
CIs; the roadmap decision gates are Stage 2 vs Stage 1 (SAM ↓≥2°, and
LR-consistency falling ~1e-2 → ~1e-6) and Stage 3 vs Stage 2 (guide must earn
its channel).  If projection does not beat plain DDIM on SAM/ERGAS, the correct
output is a negative-result paper about the limits of data-consistency
projection — still publishable, different framing.

For every stage, report per-scene PSNR, SSIM, SAM, ERGAS, LR-consistency error
`‖D(ŷ) − X‖` (expected ~1e-6 with the projection, ~1e-2 without), and
SRF-consistency error.  Paired Wilcoxon tests, bootstrap CIs, ≥3 seeds for the
final contenders.  Cross-domain protocol is inductive: source-train /
target-test with no scene overlap.

## 6. Claims that are allowed (and only after the experiment succeeds)

1. **Consistency is algebraic.**  `‖D(ŷ) − X‖` at solver tolerance for the
   supplied operator; separately for the *estimated* operator under blind
   reconstruction.  This is the one claim the selfchecks already support.
2. **Spectral fidelity.**  SAM and ERGAS lower than matched unconstrained and
   unprojected baselines over per-scene paired comparisons and three seeds.
3. **Domain robustness.**  Cross-domain SAM/ERGAS gap no worse than the
   discriminative baselines, and the LR-consistency error is bounded *on the
   target domain* by construction — which no supervised regressor can claim.
4. **Efficiency.**  Parameters, GFLOPs, full-scene latency, peak memory under
   one hardware/precision/size.

Never claim SOTA from the mechanism checks alone.  The selfchecks prove the
algebra; the ranking is an empirical outcome this repository does not yet have.

## 7. Known limitations (state them in the paper)

* **Sampling cost.**  Reverse diffusion at inference is 10–50 score forward
  passes per tile; tiled full-scene evaluation is slower than a single-pass
  regressor.  Mitigations: DDIM `sample_steps=10`, the null-space reduction
  shrinks what must be generated, and the companion proposal
  `proposal6/consistentflow/` distills the sampler into a **one-step** map
  (same guarantee, ~regressor cost) — the speed story lives there.
* **The prior is trained on a domain.**  SpectralFlow is robust to shift through
  the projection, not through the prior; a poor prior yields plausible-but-wrong
  null-space detail (e.g., hallucinated texture), which no consistency check can
  catch.
* **SRF ambiguity.**  With a jittered/synthetic MSI the operator refinement can
  fit `S` to the sample's own error.  The refinement is deliberately small
  (`refine_steps=3`) and is an optional stage, not the default.
* **Clamp-then-reproject** keeps values in [0,1] at the cost of one extra CG
  solve per step; the range component can sit marginally outside [0,1] for
  pathological scenes.

## 8. Paper outline if the evidence succeeds

1. Introduction: spectral fidelity under sensor/domain shift.
2. The null-space formulation of fusion as constrained posterior sampling.
3. The score-based spectral prior and the MSI/degradation conditioning.
4. Blind operator estimation and per-scene refinement.
5. Unified protocol and results (consistency, spectral, cross-domain).
6. Ablations, failure cases, cost, limitations.

This is a Q1-credible direction *only if* the Stage 2 vs Stage 1 comparison is
a win.  If projection does not beat plain DDIM on SAM/ERGAS, the correct output
is a negative-result paper about the limits of data-consistency projection —
still publishable, but with a different and harder framing.
