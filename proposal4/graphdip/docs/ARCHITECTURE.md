# GraphDIP — Proposal 4 (Q1 redesign)

## Status
Scaffold complete. Structural claims verified on CPU (see `selfcheck`). No
full-dataset runs yet; real runs go to Kaggle.

## What it is
Per-scene, fully **self-supervised** fusion with a GNN prior (deep image prior):

1. **Superpixel graph** — the HR MSI guide is segmented (seeded k-means over
   position + colour) into `n_seg` nodes. Node features are the cluster centres.
2. **GNN prior** — message passing over the kNN band/scene graph refines node
   representations; each node emits a C-band spectrum. `head` gathers node
   outputs to the HR grid via the hard superpixel labels (differentiable in the
   node outputs). The graph is the inductive bias that replaces a learned
   data-driven prior with a scene-specific one.
3. **Physics-only objective** — for a single scene, the GNN is optimised against
   `||D(y) - X||^2 + ||S(y) - M||^2` (optionally + TV), with **no ground truth
   and no training set**.

Message passing is the "mixing" mechanism. The mixing stage is selectable:

| stage | mixing | property |
|-------|--------|----------|
| s1 | `linear` (bias-free) | genuinely linear map (verified) |
| s2 | `nonlinear` (relu) | non-linear mixing (verified) |
| s3 | `attention` (softmax q.k) | non-linear, adaptive weights (verified) |
| s4 | `attention` + TV prior | physics-only DIP with spatial regulariser |

## Verified claims (CPU selfcheck)
1. **Superpixels**: seeded k-means yields a full, non-empty `n_seg` partition.
2. **Mixing**: the linear stage satisfies `out(tx) = t out(x)` to 2e-7; the
   nonlinear and attention stages violate linearity strongly (0.58 / 0.57) —
   the non-linear mixing is real, not cosmetic.
3. **Physics-only DIP**: on a synthetic scene (superpixel-constant ground truth,
   observed through D and S), optimising the GNN against the physics objective
   alone drops the objective from 3.19 to 0.034 (ratio 0.011) with zero
   supervision. Node recovery corr. 0.50 (the physics is under-determined).
4. **Ladder smoke**: all mixing stages build, forward and backprop.

## Honest limitations
- No supervision means the physics under-determination persists: many HR cubes
  fit the observations. The superpixel prior reduces ambiguity but does not
  remove it (node corr. 0.50 even on the synthetic case).
- Piecewise-constant output over superpixels: spatial detail within a segment
  is flat unless the TV term / segmentation is fine-grained.
- DIP per scene is expensive (fresh optimisation per scene), and the objective
  is non-convex in the GNN weights; results depend on restarts and `dip_steps`.
- The graph is built once from the MSI guide (fixed during optimisation); it is
  not adapted to the fused estimate.

## Run
    python -c "import graphdip; graphdip.selfcheck()"   # structural checks
    python -c "import graphdip; graphdip.train(cfg)"    # per-scene DIP