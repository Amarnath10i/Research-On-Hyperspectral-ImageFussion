# UnfoldFusion: a deep-unfolded variational solver for HSI-MSI fusion

**Proposal 2 of four.** Implemented in [`unfoldfusion/`](../unfoldfusion/),
runnable via [`notebooks/unfoldfusion_Kaggle_GPU.ipynb`](../notebooks/unfoldfusion_Kaggle_GPU.ipynb).

---

## 1. The one-sentence difference

DAETF-Net (proposal 1) is a feed-forward network *told* about the physics
through two loss terms. UnfoldFusion has no free-form backbone at all: it is an
optimisation algorithm whose iterations have been unrolled into layers, so the
observation model is **structural** rather than encouraged.

## 2. The formulation

The observation model for fusion is

```
Y_h = B(X)     low-resolution HSI  = blur + decimate of the truth
Y_m = R(X)     multispectral image = spectral projection of the truth
```

Recovering `X` is ill-posed, so the classical variational formulation is

```
min_X  ||B(X) − Y_h||²  +  ||R(X) − Y_m||²  +  λ·φ(X)
```

with `φ` a regulariser encoding what natural hyperspectral images look like.
Half-quadratic splitting introduces an auxiliary `V` and alternates:

| step | what it does |
|---|---|
| `V ← prox_φ(X)` | project onto the prior — a **learned** denoiser |
| `X ← argmin ‖B(X)−Y_h‖² + ‖R(X)−Y_m‖² + ρ‖X−V‖²` | re-impose the observations — **solved**, not learned |

The second step is linear least squares. We solve it with a few conjugate-gradient
iterations using only matrix–vector products; `B`, `Bᵀ`, `R`, `Rᵀ` are all cheap
and known, so no matrix is ever formed.

## 3. What is actually learned

Deliberately little:

- **`prox_φ`** — a small CNN denoiser, *shared across all stages*, conditioned on
  the current `ρ` through a sinusoidal embedding and FiLM. This is the only
  free-form component.
- **`ρ` per stage** — so the solver learns its own schedule for how much to trust
  the prior versus the data as it converges.
- **the blur kernel** — *estimated per input* by a small head, not assumed. A
  sensor with a different blur is handled by re-estimating `B`, not by retraining.

The spectral basis is **not** learned: it is taken from an SVD of the LR-HSI at
inference, so the CG runs in an `r`-dimensional coefficient space rather than
over all 31 bands. That is what makes unrolling affordable.

## 4. Why it should transfer

Every stage re-imposes agreement with the actual observations. A feed-forward
network that has memorised CAVE statistics has nothing forcing it to re-explain
a Harvard observation. Here, a solution that does not satisfy `B(X)=Y_h` is
driven out at every stage regardless of which dataset it came from.

Because only the prior is free-form, the surface available for domain-specific
overfitting is the smallest of the four proposals.

## 5. Verified properties

| Property | Why it matters | Measured |
|---|---|---|
| `⟨Bx,y⟩ = ⟨x,Bᵀy⟩` | if `Bᵀ` is not the true adjoint, CG solves an inconsistent system | rel err **1.0e-06** |
| `⟨Rx,y⟩ = ⟨x,Rᵀy⟩` | same, for the spectral operator | rel err **8.6e-08** |
| CG reduces the residual | the data step must actually solve something | 1.3e+02 → 1.4e-05 over 16 steps |
| gradients reach every parameter | no dead components | 23/23 tensors |
| depth adjustable at inference | shared weights make it an algorithm, not a deep net | 3-stage-trained model runs 8 stages |

### The bug this caught

The first implementation used **reflect** padding in `B` and **zero** padding in
`Bᵀ`. Reflect padding is a linear operator whose adjoint is not reflect padding,
so the adjoint identity failed at rel err 4.5 — and CG was solving an
inconsistent system. The unrolling would have silently degenerated into a stack
of denoisers **while still training and producing plausible loss curves**. Zero
padding on both sides fixes it exactly.

This is the reason the adjoint check runs before anything else.

## 6. Deep supervision

Every unrolled stage is supervised, not just the last, with weights increasing
toward the final stage. Without it the early stages receive almost no gradient
and the unrolling collapses into "one useful stage plus decoration" — which also
destroys the adjustable-depth property.

## 7. Cost

Roughly 0.09 M parameters at the tested configuration — the smallest of the four
proposals, because the algorithm carries the structure that a network would
otherwise have to learn. Compute is dominated by `stages × cg_steps` operator
applications, so the accuracy/compute trade is explicit and tunable *after*
training.

## 8. Honest limitations

- The CG step count is fixed, not adaptive; a residual-based stopping rule would
  be better and is not implemented.
- The spectral basis is a plain SVD of the LR-HSI. A scene whose spectra are not
  well approximated at rank `r` will be capped by that choice, and `r` is not
  currently selected per scene.
- The kernel estimator predicts a single spatially-invariant blur. Real optics
  vary across the field.
- **No results on real data yet.** All numbers above are correctness checks, not
  performance claims.
