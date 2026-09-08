# P4: How we differ from prior work

## The positioning paragraph (for the paper)

> **Identifiability analysis in HSI fusion** has been studied informally
> (Chang 2003, "Hyperspectral Data Exploitation"; Bioucas-Dias et al. 2012)
> but never formalised as a phase transition.  Our contribution is:
> (1) a precise definition of identifiable, weakly identifiable, and
> non-identifiable regimes; (2) an analytical formula for the phase
> boundary; (3) a practical classification tool that works on real scenes;
> (4) a capstone connecting P1–P3 through the identifiability lens.

## What already exists

| Method | Reference | What it does | Our difference |
|---|---|---|---|
| Intrinsic identifiability | Bioucas-Dias et al. 2012 | Discusses when endmembers can be recovered | Scene-level, not fusion-level; no phase diagram |
| Virtual dimensionality | Chang & Du 2004 | Estimates effective dimensionality via noise subspace | Noise estimator, not a phase diagram; no I/W/N classification |
| Rank analysis for fusion | Various (implicit in subspace methods) | Assumes rank is known or estimates it empirically | We derive the phase boundary analytically and validate empirically |
| Phase transitions in compressed sensing | Donoho & Tanner 2005 | Phase transitions for sparsity-based recovery | They study ℓ₁ recovery; we study spectral identification in HSI-MSI fusion |

## What is genuinely new in P4

1. **Phase transition formula.** M*(r) = min M such that rank(R^T U) = r
   is a new analytical result.  It predicts, given a scene and sensor,
   whether identification is possible.

2. **I/W/N classification.** The three-regime classification with formal
   definitions is new.  Prior work talks about identifiability informally.

3. **SRF-dependent boundary.** The fact that the phase boundary depends
   on the SRF conditioning (κ) is new.  Two sensors with the same M but
   different SRFs have different phase boundaries.

4. **Capstone unification.** The phase diagram connects P1–P3:
   - P2 determines which regime a scene is in
   - P1 measures the null component in the W regime
   - P3 provides the reconstruction in the I regime