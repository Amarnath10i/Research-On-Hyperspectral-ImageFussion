# P1: Formal claims — ambiguity decomposition theorems

## The positioning paragraph (for the paper)

> **We do not build a new fusion model.** We show that any reconstruction
> produced by an HSI-MSI fusion method can be decomposed into an
> *observable* component (determined by the observation pair) and an
> *ambiguous* component (determined by the algorithm's choice within the
> null space of the degradation).  The observable component is invariant to
> the choice of algorithm; the ambiguous component is the hallucination.
> We formalise this decomposition, prove a calibration bound relating the
> ambiguity score to reconstruction error, and provide a practical auditor
> that works on top of any existing method (Fusformer, PGU-Net, U2K, etc.).

---

## Notation

| Symbol | Meaning |
|---|---|
| X ∈ ℝ^{B×H×W} | ground-truth HR-HSI |
| D: ℝ^{B×H×W} → ℝ^{B×h×w} | spatial degradation |
| R ∈ ℝ^{B×M} | spectral response (SRF) |
| Y_H = D(X) + n_H | noisy LR-HSI |
| Y_M = R^T X + n_M | noisy HR-MSI |
| A = [D; R^T] | combined degradation operator |
| N(A) | null space of A (rank-r complement) |
| R(A^T) | row space of A (observable subspace) |
| Ô | any rank-r̂ reconstruction from (Y_H, Y_M) |
| E_obs = P_R(Ô) | observable component |
| E_null = Ô - E_obs | ambiguous component |

---

## Claim 1 (Structural identity: observable decomposition)

**Statement.** For any reconstruction Ô = f(Y_H, Y_M) that lies in the
rank-r subspace spanned by the LR-HSI basis, the decomposition

    Ô = E_obs + E_null

where E_obs ∈ R(A^T) and E_null ∈ N(A), is unique and satisfies:

    E_obs = P_{R(A^T)}(Ô)    (row-space projection)
    E_null = P_{N(A)}(Ô)     (null-space projection)

**Proof sketch.** R(A^T) ⊕ N(A) = ℝ^{B×N} (fundamental theorem of
linear algebra).  The projections commute with the rank restriction,
so the decomposition is unique.

**Why this matters.** The decomposition is *not* a property of the
algorithm — it's a property of the operator A.  Every algorithm that
produces a rank-r̂ reconstruction gets the same E_obs for the same
observations.  The algorithms differ only in how they fill E_null.

---

## Claim 2 (E_obs invariance to null perturbation)

**Statement.** Let Ô be any rank-r̂ reconstruction.  Then for any
Δ ∈ N(A) (any null-space perturbation):

    E_obs(Ô + Δ) = E_obs(Ô)

**Proof.** P_{R(A^T)}(Δ) = 0 since Δ ∈ N(A) ⊥ R(A^T).  Therefore
P_{R(A^T)}(Ô + Δ) = P_{R(A^T)}(Ô) + P_{R(A^T)}(Δ) = P_{R(A^T)}(Ô).

**Empirical hypothesis (to test on real data).** If E_obs is invariant
to the choice of algorithm, then:

  - E_obs computed from Fusformer ≈ E_obs computed from PGU-Net ≈
    E_obs computed from subspace-LS, for the same (Y_H, Y_M).

  - The differences are bounded by the noise level σ.

---

## Claim 3 (Ambiguity score correlates with reconstruction error)

**Statement.** Define the ambiguity score:

    H(Ô) = ‖E_null‖_F / ‖Ô‖_F

Then the spectral reconstruction error satisfies:

    ‖X - Ô‖_SAM ≤ arcsin(H(Ô)) + noise terms

where the noise terms depend on σ and the SRF conditioning.

**Proof sketch (informal).** The SAM error measures the angle between
the ground-truth and reconstructed spectra.  The null component E_null
adds a spectral direction that is orthogonal to the observable subspace;
its angular contribution is bounded by arcsin(H).  The noise terms come
from n_H and n_M.

**Empirical hypothesis (to test on real data).** H(Ô) correlates with
SAM error across methods and scenes:

  - If we audit Fusformer, PGU-Net, U2K, and subspace-LS on the same
    scenes, the correlation between H and SAM should be ≥ 0.6.

  - Higher H → worse SAM (the method is hallucinating more).

---

## Claim 4 (Calibration bound for uncertainty map)

**Statement.** The per-pixel uncertainty map u(i) = ‖E_null[:, i]‖₂
satisfies:

    E[|X(i) - Ô(i)|²] ≤ u(i)² + σ² · c(κ)

where c(κ) is a constant depending on the SRF condition number κ.

**Interpretation.** The uncertainty map is a *lower bound* on the
reconstruction error: where u is high, the method is unreliable.
The bound is tight when the noise is small and the SRF is well-conditioned.

---

## What is genuinely new in P1

1. **The decomposition itself.** While range/null decomposition is known
   in linear algebra, applying it to the *fusion* problem (D; R^T) to
   separate observable from ambiguous components is new.

2. **Algorithm invariance of E_obs.** The claim that E_obs is the same
   for all algorithms is a structural identity, not an empirical finding.

3. **Auditor as a tool.** The AmbiguityAuditor is a practical diagnostic:
   feed any fusion model's output through it and get the decomposition.
   This is the "ambiguity auditing" contribution — we don't compete on
   PSNR, we compete on understanding *why* methods fail.

4. **Hallucination metric.** The cosine similarity between E_null and
   E_obs quantifies whether the null component is correlated with the
   observed component (hallucination) or uncorrelated (noise).  This
   distinction is new.