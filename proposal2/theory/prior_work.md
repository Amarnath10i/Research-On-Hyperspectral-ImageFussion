# P2: How we differ from prior work

## The positioning paragraph (for the paper)

> **Spectral rank estimation** for HSI data is a studied problem
> (Bioucas-Dias & Nascimento 2008, IEEE 9591999; Zhang et al. 2020,
> "Intrinsic rank estimation").  These methods estimate the *intrinsic*
> spectral rank of the scene — i.e. the number of independent endmember
> spectra — which is a property of the scene alone and does not depend on
> the sensor.  Our observation-identifiable rank r_id is different: it
> counts the spectral degrees of freedom that are *pinned down by the
> observation pair* (LR-HSI + HR-MSI).  Two scenes with the same intrinsic
> rank can have different r_id if their SRFs have different overlap or
> their MSI band counts differ.  The consequence for fusion is that
> intrinsic rank overestimates how many spectral directions can be
> reconstructed at HR resolution; r_id gives the correct operating rank.

## What already exists

| Method | Reference | What it estimates | Our difference |
|---|---|---|---|
| Intrinsic rank | Bioucas-Dias & Nascimento 2008; Zhang et al. 2020 | rank(X) = number of endmembers | Scene-only, ignores sensor; r_id = rank(R^T U_r) depends on SRF |
| Virtual dimensionality | Chang & Du 2004 | effective number of bands via noise subspace | Noise-level estimator, not a rank of the fusion problem |
| Gavish–Donoho threshold | Gavish & Donoho 2014 | optimal rank for low-rank + noise recovery | We apply it to the MSI spectral matrix, not the HSI; our threshold uses the LR-HSI noise estimate |
| Matrix completion for fusion | Various | rank assumption in completion formulations | They assume rank is known; we estimate it from observations |

## What is genuinely new in P2

1. **r_id as a new object.** rank(R^T U_r) has not been studied as the
   relevant rank for HSI-MSI fusion.  Existing work uses either the
   intrinsic rank (ignoring the sensor) or an oracle rank (assumes
   ground truth is available).

2. **Observation-only estimation.** r̂_id is computed from (Y_H, Y_M)
   alone — no oracle, no ground truth, no known SRF.  The noise level
   is estimated from the LR-HSI trailing singular values; the threshold
   is Gavish–Donoho applied to the MSI spectral matrix.

3. **Model-selection rule.** r* = r̂_id auto-selects the reconstruction
   rank, preventing spectral distortion from over-ranked methods.  This
   is the practical payoff: feed r̂_id into subspace-LS, PGU-Net, U2K,
   or any rank-parameterised method and it automatically uses the right
   spectral complexity.

4. **Formal guarantees.** Theorem 1 (recovery) and Theorem 2 (error
   lower bound) provide the theoretical foundation that existing rank
   estimators lack in the fusion context.