# P1: How we differ from prior work

## The positioning paragraph (for the paper)

> **Range/null decomposition applied to HSI-MSI fusion** has not been
> studied as a diagnostic tool.  Prior work on range-space methods
> (Ruitenbeek et al. 2007, Foucart & Rauhut 2013) uses it for *reconstruction*
> — building a new algorithm.  We use it for *auditing* — understanding
> where existing algorithms hallucinate.  Our contribution is a practical
> tool (the AmbiguityAuditor) that works on top of any trained model and
> a set of provable properties (E_obs invariance, H calibration) that
> existing hallucination metrics lack.

## What already exists

| Method | Reference | What it does | Our difference |
|---|---|---|---|
| Range-space reconstruction | Ruitenbeek et al. 2007; Foucart & Rauhut 2013 | Reconstructs using the range subspace of the degradation | They build a new algorithm; we audit existing algorithms |
| Null-space regularization | Various (generic in inverse problems) | Penalises the null component during optimisation | They use it in training; we measure it post-hoc on any output |
| Hallucination metrics for GANs | Blau & Michaeli 2018 (perceptual quality) | Measures perceptual distortion, not spectral hallucination | They measure image quality; we measure spectral direction invention |
| Spectral angle mapper (SAM) | Chang 2003 | Measures angle between ground-truth and reconstructed spectra | SAM requires ground truth; H works without it |
| Uncertainty quantification in HSI | Various (Bayesian deep learning) | Produces uncertainty estimates from a trained model | They need a trained model; we audit any model including training-free |

## What is genuinely new in P1

1. **Auditing, not reconstruction.** The decomposition is applied to
   understand *existing* methods, not to build a new one.  This is the
   core novelty: we don't compete on PSNR, we compete on diagnosis.

2. **Algorithm invariance of E_obs.** The observable component is the
   same for all algorithms.  This is a structural identity (Claim 2),
   not an empirical finding.

3. **Hallucination vs noise.** The cosine similarity between E_null and
   E_obs distinguishes hallucination (correlated with observed) from
   noise (uncorrelated).  Existing metrics don't make this distinction.

4. **Practical tool.** The AmbiguityAuditor is a drop-in diagnostic for
   any (lr_hsi, msi) → {"out": ...} model.  It doesn't require retraining,
   doesn't need ground truth, and works on any scene size.