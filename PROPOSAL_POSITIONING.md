# Cross-cutting positioning: how all four proposals differ from prior work

## The unifying narrative

> **We do not build a new fusion model.** We study the *identifiability*
> of HSI-MSI fusion: what can be recovered from a given sensor pair,
> and what is irrecoverably ambiguous.  Our four contributions form a
> chain:
>
> 1. **P2** estimates the observation-identifiable rank r_id — the number
>    of spectral degrees of freedom pinned down by the observation pair.
> 2. **P1** decomposes any reconstruction into observable (E_obs) and
>    ambiguous (E_null) components, showing where methods hallucinate.
> 3. **P3** proves a generalization bound for continuous spectral fields
>    that explains when zero-shot sensor transfer succeeds or fails.
> 4. **P4** derives the phase transition formula that predicts, given a
>    scene and sensor, whether identification is possible at all.
>
> The common thread: uncertainty and ambiguity quantification in HSI
> fusion, with formal guarantees and practical tools.

## How we differ from each category of prior work

### vs. New fusion architectures (Fusformer, PGU-Net, U2K, etc.)
- They compete on PSNR/SSIM on benchmark datasets.
- We don't compete on PSNR — we provide diagnostic tools (P1 auditor),
  estimation methods (P2 r_id), and theoretical guarantees (P3–P4).
- Our tools work *on top of* their models, not instead of them.

### vs. Subspace methods (Chanussot, Zhang, Bioucas-Dias)
- They assume the rank is known or estimate it heuristically.
- We prove the rank estimate r_id is the correct operating rank (P2),
  and derive the phase transition that determines when identification
  is possible (P4).

### vs. Implicit neural representations (NeSSR, SINR, INR-HSISR)
- They optimise a neural field per scene, no transfer.
- We prove when transfer is possible (P3 bound) and provide a hybrid
  field that provably satisfies the bound.

### vs. Rank estimation (Bioucas-Dias & Nascimento 2008, Zhang et al. 2020)
- They estimate the *intrinsic* rank of the scene (property of X alone).
- We estimate the *observation-identifiable* rank (property of X and the
  sensor pair), which is the correct rank for fusion.

### vs. Hallucination detection (GAN-based quality metrics)
- They measure perceptual quality, not spectral hallucination.
- We measure the cosine similarity between E_null and E_obs, distinguishing
  hallucination from noise.

### vs. Phase transitions in compressed sensing (Donoho & Tanner 2005)
- They study ℓ₁ recovery of sparse signals.
- We study spectral identification in HSI-MSI fusion, a different
  mathematical problem with different phase boundaries.

## The Q1 novelty checklist

| Criterion | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| New mathematical object | E_obs/E_null decomposition | r_id = rank(R^T U_r) | Δ_sensor bound | M*(r) phase boundary |
| Formal theorem/lemma | Claims 1–4 | Theorems 1–2 | Generalization bound | Phase transition theorem |
| Not another CNN | ✓ (auditor, not a model) | ✓ (estimator, not a model) | ✓ (field, not a CNN) | ✓ (analysis, not a model) |
| Real-data experiments needed | ✓ (audit existing methods) | ✓ (r̂_id on CAVE/Harvard) | ✓ (sensor shift on real data) | ✓ (phase diagram on real scenes) |
| Ablation showing component matters | ✓ (E_obs invariance) | ✓ (r_id vs intrinsic rank) | ✓ (L_F vs EMD) | ✓ (phase boundary vs grid) |
| Practical tool delivered | AmbiguityAuditor | select_rank() | rank_adaptive_subspace | phasediagram.py |