# ZeroFusion: self-supervised per-scene fusion, with no training set

**Proposal 4 of four — and the control arm for the whole study.** Implemented in
[`zerofusion/`](../zerofusion/), runnable via
[`notebooks/zerofusion_Kaggle_GPU.ipynb`](../notebooks/zerofusion_Kaggle_GPU.ipynb).

---

## 1. The one-sentence difference

Proposals 1–3 learn from a training split and are then asked to generalise.
ZeroFusion has **no training split at all**: given a single pair (LR-HSI, MSI) it
optimises a small network from scratch on that pair, using only losses that need
no ground truth.

## 2. Why this is the most important of the four

It is the reference line. Every cross-domain number the other three report is an
attempt to answer *"how much does training on CAVE hurt you on Harvard?"*.
ZeroFusion never trains on CAVE, so it **cannot suffer domain shift** — its
Harvard score is by construction the same kind of quantity as its CAVE score.

That makes the comparison decisive:

- If a trained proposal **beats** ZeroFusion cross-domain, the training genuinely
  transferred something useful.
- If it **does not**, then whatever that model learned on the source dataset was
  worth less than nothing on the target, and the honest conclusion is that
  per-scene optimisation is simply the better method — not that our model is good.

Very few fusion papers include this comparison. It is easy to look strong
cross-domain until someone asks whether a method that ignores the training set
entirely would have done better.

## 3. The model

Physically grounded rather than a generic denoiser — linear spectral unmixing:

```
X = E · softmax(A) · s

  E : [bands, k]   endmember spectra, initialised from an SVD of the LR-HSI
  A : [k, H, W]    abundance logits, predicted from BOTH observations
  s : [1, H, W]    per-pixel brightness, bounded to ≈[0.37, 2.7]
```

- **Sum-to-one and non-negativity** come free from the softmax, so every
  reconstructed pixel is a convex combination of plausible spectra. This is a
  strong prior that costs nothing and rules out the spectral hallucination that
  shows up as large SAM.
- **Endmembers start from the scene's own subspace.** Random initialisation makes
  the optimisation slow and unstable; the leading singular vectors of the observed
  cube are already close.
- **Brightness factor.** A strict convex combination cannot represent illumination
  change, so a shaded region of a uniform material would be unreachable. A bounded
  positive scale restores that freedom while keeping the prior on spectral *shape*.
- **Capacity is the regulariser.** ~0.03 M parameters. With no training set, a
  large network simply memorises the observations, noise included.

## 4. The objective

Only terms computable without ground truth:

```
‖Down(X) − Y_h‖  +  ‖SRF(X) − Y_m‖        physics
+ w·L½(A)  +  w·TV(A)  +  w·range(X)       priors
```

Early stopping is on the **self-supervised** objective, so no ground truth is
consulted at any point — this remains legitimate on a genuinely unlabelled scene.

## 5. Three bugs this design caught

All three left training running and loss curves looking plausible while the model
learned nothing. They are documented because each is a trap the other proposals
could fall into.

**`clamp` in the forward pass kills gradients.** `out.clamp(0,1)` has exactly zero
gradient outside its range, so once the estimate saturated, optimisation froze.
The symptom was byte-identical results across 200, 600 and 1500 steps *and* across
k = 8, 16, 31, with two random initialisations differing by exactly 0.0. Clamping
now happens only at evaluation, with a soft range penalty in the loss instead.

**Entropy is a degenerate sparsity prior.** Under a sum-to-one softmax, minimising
entropy has a free minimum at one-hot that is reachable regardless of how badly
the data is fitted. Measured: entropy fell 2.08 → 0.00 while the reconstruction
went spatially constant. Replaced with an L½ quasi-norm, which still has to
explain the observations.

**An unbounded brightness head runs away.** `exp(scale_head)` with a gradient of
0.42 on the bias saturated the output within a few steps. Bounded to ≈[0.37, 2.7].

After the fixes, quality improves monotonically with step count (18.8 → 22.1 →
24.3 PSNR at 200/600/1500 steps on the synthetic harness) and was **still
improving at 1500**, which is why that is the default and described as a floor.

## 6. Cost — the honest headline

| | ZeroFusion | Proposals 1–3 |
|---|---|---|
| Training | **none** | hours, once |
| Per scene at inference | seconds to a minute | fractions of a second |

This trade is stated rather than buried. It is the reason ZeroFusion is a
reference line and a deployment option for one-off scenes, not a drop-in
replacement for a trained model in a throughput setting.

## 7. Honest limitations

- The linear mixing model has a representational ceiling. On the synthetic
  harness an *oracle* fit (abundances optimised directly against ground truth)
  reached 30.27 dB while bicubic reached 30.84 — so **that harness cannot rank
  this method fairly**, and its real standing is unknown until it runs on CAVE
  and Harvard.
- The fitted region is capped by `max_side` so a 1040×1392 Harvard scene stays
  affordable; large scenes are therefore scored on a crop.
- `k` (endmember count) is fixed rather than selected per scene.
- **No results on real data yet.**
