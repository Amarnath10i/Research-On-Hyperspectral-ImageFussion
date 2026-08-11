# ContinuumFusion: arbitrary-scale HSI-MSI fusion by implicit representation

**Proposal 3 of four.** Implemented in [`continuumfusion/`](../continuumfusion/),
runnable via [`notebooks/continuumfusion_Kaggle_GPU.ipynb`](../notebooks/continuumfusion_Kaggle_GPU.ipynb).

---

## 1. The one-sentence difference

Proposals 1 and 2 both produce a fixed output grid and are welded to one scale
factor. ContinuumFusion never represents the image as a grid: it learns a
continuous function and *samples* it wherever an output pixel is wanted, so the
scale factor becomes a **query parameter** rather than an architectural constant.

## 2. Why this is the right gap to attack

The benchmark in [`existing/`](../../existing/results/BENCHMARK.md) is unusable
for cross-method claims precisely because its ten methods ran at ×4, ×8, ×16 and
×32. **That is not sloppy bookkeeping — it is a property of grid-based
architectures**, each welded to the factor it was trained for. Re-running them
all at one factor means retraining all of them.

A model that handles any factor from one set of weights makes the comparison
well-posed: every method can be evaluated at every factor, and the factor becomes
a reported axis instead of a hidden confound.

It also matters physically. A real sensor's resolution ratio is whatever the
optics give you — rarely a power of two.

## 3. The architecture

```
LR-HSI ─┐
        ├─► Encoder ──────────► latent on the LR grid   (spectra live here)
MSI ────┘

MSI ──────► DetailEncoder ────► features at FULL resolution (detail lives here)

for each queried coordinate (x, y):
    gather the 4 nearest latent cells
    for each: MLP([latent, detail(x,y), Δcoord, cell_size, band_embedding]) → radiance
    blend the 4 by area weights
```

Four decisions carry the design:

**Latent on the LR grid.** The spectra were only measured there, so that is where
spectral reasoning belongs. Putting the latent at HR would force the encoder to
hallucinate spectra before it has any reason to.

**Detail read at full resolution.** Spatial high-frequency information is sampled
from the HR MSI feature map at the query point, not upsampled from the LR grid.

**The band index is a coordinate.** It enters through a learned embedding, so one
MLP serves all bands, the parameter count does not grow with band count, and the
representation is continuous along wavelength — a band the sensor never sampled
can be queried by interpolating its embedding.

**Local ensemble.** Each output point is decoded from its four neighbouring latent
cells and blended by area weights. Decoding from the single nearest cell leaves
visible blocking at cell boundaries.

The MLP predicts a **residual** over bicubic, not the full radiance, which trains
substantially faster.

## 4. Why it should transfer

The decoder sees relative coordinates and cell sizes, never absolute image size,
so it is scale-agnostic by construction. Training with randomised scale factors
(`train_scales`) turns the resolution ratio into just another nuisance variable
the model has learned to be robust to — the same argument as randomising the
blur, applied to geometry.

## 5. Verified properties

| Property | Measured |
|---|---|
| Coordinates are exact pixel centres | `[-0.75, -0.25, 0.25, 0.75]` for n=4, symmetric |
| Arbitrary output resolution | 32², 64², 96², 128², and non-square 48×80 |
| Arbitrary scale factor, one set of weights | ×2, ×3, ×4, ×5, ×8 |
| Gradients reach the band embedding | yes — the spectral coordinate is learned |
| One MLP serves all bands | decoder output dim is 1, not `bands` |
| Tiled inference | full scenes at 16 GB |

Note ×3 and ×5 in that list: non-power-of-two factors that a PixelShuffle-based
upsampler cannot express at all.

## 6. Cost

The MLP is evaluated once per output pixel **per band**, which is the honest
downside: inference is heavier than a convolutional decoder at a single fixed
scale. The trade buys scale-freedom, and it is reported rather than hidden.
Batching the band dimension inside the decoder keeps it tractable.

## 7. Honest limitations

- Evaluating at ×16 or ×32 with a model trained on ×2–×8 is *extrapolation*, and
  should be reported as such rather than as in-distribution performance.
- The continuous-along-wavelength claim is structural. Querying an unsampled band
  is possible, but this repository has no ground truth for such bands, so the
  claim is **not empirically validated** here.
- The local ensemble costs 4 MLP evaluations per output point.
- **No results on real data yet.** Everything above is a correctness property,
  not a performance claim.
