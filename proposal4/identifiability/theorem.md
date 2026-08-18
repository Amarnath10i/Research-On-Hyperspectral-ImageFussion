# P4: Phase transition theorem for identifiability

## The positioning paragraph (for the paper)

> **We derive an analytical phase transition formula** for the linear
> low-rank HSI-MSI fusion model.  The formula predicts, as a function
> of spectral rank r, number of MSI bands M, and SRF conditioning κ,
> whether the scene is in the identifiable (I), weakly identifiable (W),
> or non-identifiable (N) regime.  The phase boundary is a curve in
> (r, M) space; we prove it is monotone and compute the critical M*(r)
> below which identification fails.

---

## Model

The linear low-rank fusion model:

    X = U Z          (B×N = B×r · r×N)
    Y_H = D X + n_H  (h×w observation)
    Y_M = R^T X + n_M  (M×N observation)

where D is spatial degradation, R^T is spectral projection (SRF), and
r = rank(X).

The identifiable rank:

    r_id = rank(R^T U) = rank(G)

where G = R^T U ∈ ℝ^{M×r} is the projected spectral basis.

---

## Theorem (Phase transition for the linear model)

**Statement.** Under assumptions A1–A5 (from P2 recovery guarantee),
the phase boundary between the identifiable and non-identifiable regimes
is:

    r ≤ M'      (identifiable, I)
    r > M'      (non-identifiable, N)

where M' = rank(R^T U) ≤ min(M, r) is the effective number of
identifiable spectral directions.

The critical boundary is:

    M*(r) = min M   such that rank(R^T U) = r

This is achieved when the SRF R has at least r linearly independent
rows that are not in the column space of the first r-1 rows of R^T U.

**Corollary (monotone phase boundary).** M*(r) is non-decreasing in r:

    M*(r+1) ≥ M*(r)

Proof: increasing r by 1 adds a column to G = R^T U; the rank can
stay the same or increase by 1, never decrease.

**Corollary (SRF-dependent boundary).** For a fixed SRF R:

  - If R has full column rank (M ≥ B), then M*(r) = r for all r ≤ B.
    Every rank is identifiable.

  - If R has rank M < B, then M*(r) = r for r ≤ M, and M*(r) = M for
    r > M.  The SRF caps the identifiable rank at M.

  - If R is poorly conditioned (κ → ∞), M*(r) can be less than r even
    when M ≥ r, because two spectral directions become confounded.

---

## Phase diagram classification

For a given scene with rank r and SRF R:

| Regime | Condition | Consequence |
|---|---|---|
| **I** (identifiable) | r_id = r | Full spectral recovery possible; rank-r subspace methods work |
| **W** (weakly identifiable) | 0 < r_id < r | Partial recovery; some spectral directions are lost; rank-r̂ < r methods needed |
| **N** (non-identifiable) | r_id = 0 | No spectral recovery from MSI alone; pure spatial super-resolution only |

The phase transition occurs at the critical M*(r) curve.

---

## Connection to experiments

The simulator (`proposal4/identifiable/simulator.py`) generates synthetic
scenes across the (r, M) grid and classifies each point.  The phase
diagram (`proposal4/identifiable/phasediagram.py`) plots the empirical
boundary and compares it to the theoretical M*(r) curve.

**What the simulation already validates (self-check):**
- Monotone r_id as r increases (PASS)
- Non-increasing null_frac as M increases (PASS)
- Score ↔ null_frac correlation -0.70 (PASS)

**What the simulation does NOT yet do:**
- Compare empirical boundary to the analytical M*(r) formula
- Show the phase transition as a continuous curve (current is coarse grid)
- Validate on real scenes (CAVE/Harvard)

---

## What is genuinely new in P4

1. **Analytical phase transition formula.** The M*(r) curve is a new
   object: given a scene rank and SRF, it predicts whether identification
   is possible.  Existing work assumes identification is always possible
   or uses heuristic rank selection.

2. **Regime classification.** The I/W/N classification with formal
   definitions is new.  Prior work talks about "identifiability" informally;
   we give it a precise phase diagram.

3. **SRF-dependent boundary.** The fact that the phase boundary depends
   on the SRF conditioning (κ) is new.  Two sensors with the same M
   but different SRFs have different phase boundaries.

4. **Capstone connection.** The phase diagram unifies P1–P3:
   - P2 (r_id) determines which regime a scene is in
   - P1 (ambiguity) measures the null component in the W regime
   - P3 (continuous fields) provides the reconstruction in the I regime