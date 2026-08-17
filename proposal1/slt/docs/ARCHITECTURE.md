# Spectral Lie Transport (SLT): fusion as geodesic transport on the spectral manifold

> **Status: implemented and self-checked.** The algebraic/structural claims
> below are verified numerically by `selfcheck.py` (seconds on CPU).  The
> headline result: under a 1.7x illumination change the manifold network's SAM
> moves 0.03 deg (float32 floor) while a matched Euclidean-residual control
> moves ~8-10 deg.  No full CAVE/Harvard ranking exists yet — the ladder runs
> on Kaggle.

## 1. The problem, restated

The P1 benchmark (`proposal1/results/RESULTS.md`) shows DAETF-Net at or below
Bicubic/GSA, and the cross-domain story is worse: SAM degrades to double digits
under sensor shift.  The root cause is that **per-image intensity differences
dominate pixelwise metrics while masking spectral error**: a network that
predicts absolute intensities can satisfy a PSNR-driven loss while rotating
every spectrum by several degrees, because L2 distance in absolute value
space is not the metric the protocol reports.

P1's Projective Spectral Embedding already established the theory half: spectral
angle is the meaningful distance, and manifold geometry predicts SAM (Pearson
r = 0.74 on held-out spectra).  SLT operationalises that finding — it makes the
network *incapable* of intensity masking.

## 2. The manifold and the maps

Normalised spectra (per-pixel L2) live on the unit sphere S^{B-1}.  On that
sphere the spectral angle is exactly the geodesic distance:

```
SAM(p, q) = arccos(<p, q>) = d_S(p, q)
```

`manifold.py` implements the differentiable geometry:

* `tangent_projection(v, p)` — the tangent plane at p: `v - <v,p>p`.
* `exp_map(p, v)` — geodesic transport: `cos(|v|) p + sinc(|v|) v`.  The
  transported displacement has magnitude `|v|` = the SAM distance travelled.
* `log_map(p, q)` — the inverse: the tangent vector whose image is q, with
  `|Log_p(q)| = SAM(p, q)`.  Because exp/log is an isometry, squared MSE in the
  tangent space *bounds* the geodesic (SAM) error — the training objective is
  metric-aligned by construction.

## 3. The network parameterisation

```
Y_0 = bicubic(LR)                      (absolute, from the observation)
I_0 = ||Y_0|| ,  dir0 = Y_0 / I_0      (intensity + direction, factored)
v   = tangent_projection(head(enc), dir0)     (learned tangent vector)
dir_out = Exp_{dir0}(v)                       (geodesic transport)
Y_hat   = I_0 * dir_out                       (intensity re-attached)
```

Two properties are **by construction**:

1. **Illumination equivariance.**  The encoder sees only scale-invariant inputs
   (`dir0`, and the normalised MSI guide when enabled).  Scaling the observation
   by `c` scales `I_0` by `c` and leaves `dir_out` untouched, so
   `Y_hat(c·X) = c·Y_hat(X)` — the spectrum (SAM) never moves.  The Euclidean
   residual control (`manifold=False`) has no such guarantee and loses ~8-10 deg
   SAM under the same change: that gap is the falsifiable claim.
2. **Metric alignment.**  `SAM(Y_hat, Y) = d_S(dir_out, dir_gt)` — the error the
   network incurs in the tangent space is exactly the headline metric.

Intensity is deliberately carried by the observation (`fix_intensity=True`): the
network's only degrees of freedom are geodesic displacements, so it cannot trade
spectral fidelity for a better L2 score.

## 4. Ladder (what the experiments decide)

| Stage | Model | Question answered |
|---|---|---|
| 0 | Bicubic, GSA | protocol check |
| 1 | Euclidean residual CNN (same capacity, no manifold) | is transport better than regression? |
| 2 | naive chordal transport (`dir0 + v`, renormalise) | does *any* manifold help? |
| 3 | geodesic exponential map | does the *correct* geometry help? |
| 4 | Stage 3 + MSI-guided tangent | does the guide earn its channel? |
| 5 | (future, not scaffolded) spectral group equivariance | high-risk; band-permutation equivariance |

Report per scene: PSNR, SSIM, SAM, ERGAS, LR-consistency, and the manifold
geodesic.  Paired Wilcoxon, bootstrap CIs, >= 3 seeds.  Decision gate: Stage 3
vs Stage 2 (geodesic vs chordal) and Stage 4 vs Stage 3 (guide).

## 5. Claims that are allowed (only after the ladder succeeds)

1. **Intensity cannot mask spectral error** (selfcheck-supported already: 0.03
   deg vs 8-10 deg under 1.7x illumination).
2. **SAM/ERGAS improvement** over the Euclidean residual at equal capacity, and
   especially cross-domain (CAVE-train, Harvard-test), over per-scene paired
   comparisons and three seeds.
3. **The geometry is the contribution** — the geodesic (exp/log) beats the
   naive chordal renormalisation, isolating the metric-aligned construction from
   a mere parameterisation trick.

Never claim SOTA from the mechanism checks; the ranking is the empirical
outcome the ladder produces.

## 6. Known limitations

* **Fixed intensity.**  `fix_intensity=True` caps PSNR at the bicubic intensity
  quality; a `fix_intensity=False` variant (learned log-intensity multiplier,
  still scale-equivariant) is the documented fallback if PSNR lags.
* **`max_angle_rad` clamp.**  Very large angular corrections are capped (default
  1.4 rad); a scene whose LR direction is far from the truth needs more stages,
  not a bigger clamp.
* **Scale-equivariance ≠ cross-domain robustness.**  Illumination invariance is
  one failure mode; sensor blur/SRF shift is handled by the same
  degradation-conditioning machinery as the rest of the repo, not by the
  manifold itself.
* **Spectral group equivariance (band permutation etc.) is deferred.**  It is
  the exotic, months-of-risk extension in the redesign brief; the manifold
  transport is the falsifiable core and is implemented.

## 7. Relationship to the other proposals

* P1's PSE established the theory (manifold geometry predicts SAM); SLT is the
  constructive form of the same idea.
* The null-space proposals (P5/P6) guarantee `D(Y_hat) = X`; SLT guarantees
  the *metric* — spectral error cannot be masked.  They are complementary:
  SLT for the metric, null-space for the observation.
