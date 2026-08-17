# Analysis: the three claims behind PSE + range/null fusion

This file is the mathematical spine of the Q1 paper. It is not a survey of the
code; it is the argument a reviewer will probe, written so each step can be
verified numerically by the selfchecks or the ladder.

---

## 1. Why per-pixel intensity masks spectral error (the PSNR paradox)

**Observation (the benchmark).** Across the ten re-run baselines, PSNR
frequently *improves* on the harder dataset (five of nine score higher on
Harvard than on CAVE) while SAM degrades by 4–57 degrees and ERGAS rises by up
to two orders of magnitude.

**Mechanism.** PSNR at a constant `data_range=1.0` is

```
PSNR = 10 log10 ( 1 / mean((ŷ - y)²) ).
```

For a dark scene whose spectra sit near 0.1 (per-image intensity is low),
a constant absolute error of 0.01 gives `PSNR ≈ 20 dB` — *independent of
whether the error is spectral or scalar*. Per-image maximum normalisation makes
it worse: rescaling a dark scene to unit max turns a constant error into a
near-zero *relative* error, inflating PSNR arbitrarily.

SAM, by contrast, is a *direction* on the spectral sphere:

```
SAM(p, r) = arccos( <p, r> / (‖p‖ ‖r‖) ).
```

It ignores intensity entirely, so it cannot be fooled by rescaling — and it is
precisely the error a network trained on pixelwise L1/L2 has no incentive to
reduce. A model can hold PSNR by matching intensities while its spectral
*directions* drift, which is exactly what the benchmark shows.

**Consequence.** Any method whose training objective is pixelwise L1/L2
optimises a proxy that can be satisfied while the metric that fails (SAM)
deteriorates. PSE removes the intensity channel from the learned objective by
construction, so training can no longer "win" by reproducing intensity.

**Why not just add a SAM loss term?** A SAM term is intensity-invariant, but:

- its gradient vanishes when `p` and `r` are anti-parallel (the denominator
  `‖p‖‖r‖` and the numerator have no directional information at exactly 180°),
  so the worst-case spectral error is the least corrected;
- it is a *scalar* penalty: it says nothing about how the representation should
  be organised, so it does not transfer what is learned about direction across
  spectra.

PSE replaces the scalar penalty with a coordinate system — optimising L2 in the
embedding *is* optimising spectral angle, with well-behaved gradients
everywhere on the sphere (verified: `check_manifold_predicts_sam`, held-out
Pearson r and fitted-MAE statistics).

---

## 2. The range/null decomposition is correct, not heuristic

**Setup.** The observation operator `D = blur-and-decimate` maps an HR cube to
the LR-HSI. `D` is linear and rank-deficient: many HR cubes explain the same
LR cube exactly. Write its pseudo-inverse as

```
D† = Dᵀ (D Dᵀ)⁻¹            (computed by CG in LR space),
P_perp = I − D† D            (orthogonal projection onto the null space of D).
```

The reconstruction

```
ŷ = D† X + P_perp( r ),      r = any residual
```

satisfies, by linearity and the two identities below,

```
D ŷ = D D† X + D P_perp r = X + 0 = X.        (∗)
```

**The two identities, stated and verified:**

1. `D D† = I` on the range of `D`. Since `D† = Dᵀ (D Dᵀ)⁻¹`, `D D† =
   D Dᵀ (D Dᵀ)⁻¹ = I`. Therefore the range component `D† X` is the exact
   minimum-norm explanation of the measurement: `D(D† X) = X`.
   Verified: `nullspace.check_consistency` → max error 1.9e-06.

2. `D P_perp = 0`. `D P_perp = D − D D† D = D − D = 0`, using (1). Therefore the
   learned residual `r` is *invisible* to the observation: it cannot overwrite
   the measurement, whatever the network does.
   Verified: `nullspace.check_null_annihilation` → max error 1.6e-06.

Both hold with respect to *whichever* `D` is supplied — a fixed evaluation
kernel or a blind estimate. In the blind case `D̂(ŷ) = X` remains an identity;
against the *true* sensor the guarantee weakens to `D(ŷ) ≈ X`, which is why the
estimated operator is refined per scene against the physics of the sample.

**Why this reduces the learning problem.** An unconstrained residual network
has `C·H·W` output degrees of freedom. Because `D†X` is computed in closed form,
the network is only ever asked to supply the *null-space* component
`P_perp(r)` — the degrees of freedom the measurement genuinely leaves free. The
dimension of what must be learned is reduced by the rank of `D`, and the part
that is learned is exactly the part the data cannot falsify. The stress test
`check_network_consistency` makes this concrete: multiplying the reconstruction
head weights by 50 still leaves `D(ŷ) = X` at 7e-04 relative error — a physics
*loss* would degrade with the network; an identity cannot.

**Why a penalty is not the same thing.** A loss term `‖D(ŷ) − X‖²` negotiates
with the data: at any finite weight it can be traded against other losses, and
at evaluation time a model that drifted will violate the observation. The
projection is not a term in the objective; it is a restriction on the output
space. There is no hyperparameter that trades consistency away.

---

## 3. The projective spectral embedding is metric-aligned

**Definition.** For a spectrum `s ∈ ℝ^C`, the embedding is

```
φ(s) = MLP( s / ‖s‖ ),        intensity kept aside as a scalar.
```

**Intensity invariance.** For any `α > 0`, `φ(αs) = φ(s)` because the
normalisation removes the scale. Verified: `max|φ(s) − φ(5s)| = 3e-08`. This is
the formal statement of "illumination cannot hide spectral error": exposure,
illumination, and sensor gain act as per-pixel scalar multipliers, which are
not expressible in the representation.

**Metric alignment.** The MLP is trained so that Euclidean distance in the
embedding tracks the chord distance on the unit sphere,

```
‖φ(a) − φ(b)‖  ≈  ‖ a/‖a‖ − b/‖b‖ ‖.
```

The chord distance is a monotone function of spectral angle (the sphere
chord `c` and the angle `θ` satisfy `c = 2 sin(θ/2)`), so L2 in the embedding is
an order-preserving proxy for SAM. Verified: after calibration on a train set,
`check_manifold_predicts_sam` reports the held-out Pearson correlation and the
MAE of the fit `SAM ≈ slope·‖φ(a)−φ(b)‖ + intercept`.

**The training equivalence.** The manifold term

```
w_embed · ‖φ(ŷ) − φ(y_gt)‖²
```

minimises spectral angle *without* ever computing an arccos — no anti-parallel
singularity, and the representation stays a genuine coordinate system rather
than a scalar penalty. Intensity is neither learned nor penalised: the part the
observation determines is carried exactly by the range component `D†X`.

---

## 4. Why the two mechanisms compose

- The **range/null split** guarantees `D(ŷ) = X` — the spectral identity the
  sensor actually measured is preserved *whatever the network does*. Its
  weakness is that it says nothing about the null-space component.
- The **PSE** manifold fixes exactly that: it makes the learned (null-space)
  component honest to the metric that fails. The null-space detail is
  free in the sense that the measurement cannot see it — so the only thing
  that can make it right is the prior (the training data), and the only thing
  that makes the prior right for the *right reason* is training on the metric
  that matters.

So the decomposition says "the data decides the observable part", and PSE says
"the prior decides the unobservable part, and it is trained on the metric that
fails under shift." Each addresses the other's blind spot.

---

## 5. Known limits of the argument (state these in the paper)

- **Linear degradation assumed.** Tone mapping, non-linear sensor response, or
  saturation break `X = D(Y)`. Under such operators the identity `D(ŷ) = X`
  holds only for the *modelled* `D`, which no longer describes the sensor.
- **Estimated operator.** In blind settings the projection uses `D̂`; an
  erroneous kernel still gives `D̂(ŷ) = X` but weakens the claim against the
  true sensor. Per-scene refinement mitigates but does not remove this.
- **Intensity invariance is per-pixel scaling.** Spatially varying illumination
  (e.g. soft shadows) is a per-pixel scalar — still handled. Non-Lambertian
  surfaces, where the observed spectrum depends on view angle as well as
  illumination, violate the assumption that a single spectral direction is the
  ground truth.
- **Anti-parallel spectra** (SAM = 180°) are the natural failure of any
  direction metric; the embedding is better behaved than a raw SAM term but
  still degenerate at exactly antipodal points.
