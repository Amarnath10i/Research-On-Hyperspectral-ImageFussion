# ConsistentFlow: One-Step Null-Space-Consistent Generative Fusion

> **Status: implemented and self-checked.** The algebraic claim — `D(Y_hat) = X`
> after a **single** consistency-map forward pass — is verified by
> `python -c "import consistentflow; consistentflow.selfcheck()"` (CPU seconds).
> No full CAVE/Harvard training run has produced comparable numbers yet; as with
> P5, the ranking is an empirical outcome this repository does not yet have.

## 1. Why this proposal exists

Proposal 5 (SpectralFlow) makes data consistency an algebraic identity by
projecting onto the null space of the observation operator at every reverse
step. Its one structural weakness is **cost**: a DDIM run is 10–50 score
forward passes per tile, which the roadmap and the P5 docs both flag as the
main objection a reviewer will raise.

ConsistentFlow removes the loop. The contribution is not a new generative
architecture — it is the observation that the P5 sampler can be **distilled
into a single-step map** without losing the consistency guarantee, because the
guarantee lives in the projection, not in the number of steps.

```
Y_hat = D_pinv(X) + P_perp( f_phi(y_T, T, M, d) )
        \______/        \_______________________/
      closed form          ONE forward pass of a
      from the LR-HSI      consistency map + projection
```

## 2. Mechanism

A **consistency map** `f_phi(y_t, t, M, d)` maps any point on a noise
trajectory straight to the clean endpoint. Trained by **consistency
distillation** (Song et al., ICLR 2023), it is pushed toward its own EMA target
along teacher trajectories:

```
L_CD = | f_phi(y_{t+1}, t+1) - f_phi^-(y_t, t) |^2
```

with `y_t` reached by one teacher (P5 score net) DDIM step from `y_{t+1}`.
Both sides are mapped onto the consistent set `D_pinv(X) + P_perp(·)` *during*
training, so the map learns to output points that re-explain the observation —
it does not negotiate with it.

At inference:

1. `y_T ~ N(0, I)`.
2. `f_phi(y_T, T, M, d)` gives the clean estimate in **one** forward pass.
3. `Y_hat = D_pinv(X) + P_perp(clamp(·))` re-asserts `D(Y_hat) = X` exactly.

A few-step variant (standard stochastic consistency sampling: re-noise between
steps) exists for the speed/quality trade-off study.

## 3. Positioning against P5

| | SpectralFlow (P5) | ConsistentFlow (P6) |
|---|---|---|
| Consistency | algebraic, per-step projection | algebraic, single post-map projection |
| Inference cost | 10–50 score passes / tile | **1** score pass / tile (~regressor) |
| Prior | score net (denoising score matching) | same score net, **distilled** into a one-step map |
| Blind operator | estimated + per-scene refinement | estimated (same deg head); refinement optional |
| New claim | projection beats plain DDIM (Stage 1 vs 2) | **one-step** consistency beats multi-step at a fraction of the cost |

Falsifiable hypothesis:

> Compared with (a) the multi-step projected sampler (P5) and (b) the same
> one-step map *without* the projection, one-step null-space-consistent
> sampling matches or improves SAM/ERGAS while reducing LR-consistency error
> to solver tolerance at ~10–50× lower inference cost. If one-step projection
> is worse than multi-step projection, the correct output is a
> speed-vs-quality Pareto study — still publishable, different framing.

## 4. Required experiments (same protocol as the repository)

| Stage | Model | Question answered |
|---|---|---|
| 0 | Bicubic, GSA, Subspace-LS | protocol audit |
| 1 | One-step map, **no** projection | does the distilled prior learn the manifold? |
| 2 | Stage 1 + null-space projection | does the projection help at one step? |
| 3 | Stage 2 vs P5 multi-step DDIM | how much quality does one-step cost? |
| 4 | Stage 3 at sample_steps = 2/4 | the speed/quality Pareto curve |
| 5 | Stage 3 + blind operator refinement | does refinement help the one-step map? |

Report per scene: PSNR, SSIM, SAM, ERGAS, **LR-consistency** (headline: ~1e-6
at one step with the projection), and per-tile latency (headline: 1 forward
pass vs 10–50). Paired Wilcoxon, bootstrap CIs, ≥3 seeds, mean ± std.

## 5. Implementation notes

- **Reuses P5 components deliberately.** `SpectralScoreNet`,
  `RangeNullProjector`, and the linear schedule are imported from
  `spectralflow`. The proposal's contribution is the distillation and the
  one-step sampler, not a new prior — importing the prior keeps the comparison
  honest (identical capacity) and the codebase small.
- `Config.sample_steps` defaults to **1**. `use_projection` gives the Stage 1
  control arm without re-training.

## 6. Known limitations (state in the paper)

- **Distillation cost.** One-time: a trained P5 score net (or the same
  schedule self-distilled) is needed to generate teacher trajectories. Amortised
  over inference it is negligible.
- **One-step quality ceiling.** A single pass cannot recover from a bad map;
  the few-step variant (steps 2–4) is the safety valve.
- **The prior is still trained on a domain.** Consistency is structural; the
  *detail* in the null space is as good or as wrong as the prior was — the
  projection cannot invent spectral truth that was never in the training data.
- **SRF/blur ambiguity in the blind setting** carries over from P5.

## 7. Why Q1

The roadmap's decision matrix says P5's blocker is speed. ConsistentFlow turns
that blocker into its headline: algebraic data consistency at regressor speed.
That is the cleanest possible answer to the "physics loss is a penalty, not a
guarantee" objection *and* the "diffusion is too slow" objection in one paper.
