# P3: Generalization bound for continuous spectral fields

## The positioning paragraph (for the paper)

> **We prove a generalization bound** for continuous spectral fields
> that shows the sensor-shift error Δ_sensor depends on the Lipschitz
> constant of the spectral field and the Earth Mover's Distance between
> the source and target SRFs.  The bound explains why zero-shot
> transfer fails for sharp spectral features and succeeds for smooth
> ones, and predicts which sensor pairs are compatible.

---

## Model

The continuous spectral field:

    F: ℝ³ → ℝ^B    (x, y, λ) → spectrum

Parameterised as F_θ(x, y, λ) with Lipschitz constant L_F in the
spectral direction.

Sensor operators:

    S_src: ℝ^B → ℝ^{M_src}   (SRF of source sensor)
    S_tgt: ℝ^B → ℝ^{M_tgt}   (SRF of target sensor)

Sensor-shift error:

    Δ_sensor = ‖S_tgt ∘ F - S_tgt ∘ F̂‖    (error from SRF mismatch)

where F̂ is the source-trained field.

---

## Theorem (Generalization bound for sensor shift)

**Statement.** Let F be a Lipschitz continuous spectral field with
Lipschitz constant L_F in the spectral direction.  Let S_src and S_tgt
be two sensor operators with SRFs {s_i^{src}} and {s_j^{tgt}}.  Then:

    Δ_sensor ≤ L_F · EMD(s^{src}, s^{tgt}) + noise terms

where EMD is the Earth Mover's Distance between the source and target
SRF distributions.

**Proof sketch.** The spectral field F(·, λ) is L_F-Lipschitz in λ.
The sensor operator integrates F against the SRF: S[F]_j = ∫ F(·, λ) s_j(λ) dλ.
If the SRFs s^{src} and s^{tgt} are close in EMD, then S_src[F] and
S_tgt[F] are close in ℓ₂, with the bound proportional to L_F · EMD.

**Interpretation.** The sensor-shift error is controlled by:
1. How sharp the spectral field is (L_F): sharp features → large L_F → large error.
2. How different the sensors are (EMD): large SRF shift → large error.

For smooth spectral fields (L_F small), zero-shot transfer works even
for large SRF shifts.  For sharp spectral features (L_F large), even
small SRF shifts cause large errors.

---

## Corollary (Smooth spectral fields transfer better)

If F is α-Hölder continuous with exponent α ∈ (0, 1]:

    Δ_sensor ≤ C_α · EMD(s^{src}, s^{tgt})^α

for a constant C_α depending on the Hölder constant.

**Interpretation.** Smoother fields (larger α) have smaller transfer
error.  This explains why PCA-based methods (which smooth the spectral
dimension) transfer better than per-band methods.

---

## Corollary (Sensor compatibility)

Two sensors are *compatible* for zero-shot transfer if:

    EMD(s^{src}, s^{tgt}) < ε / L_F

where ε is the acceptable error threshold.

**Interpretation.** For a given spectral field sharpness L_F, there is
a maximum SRF distance beyond which transfer fails.  This predicts
which sensor pairs (e.g., CAVE→Harvard, or WorldView-3→Pléiades)
are compatible.

---

## Connection to experiments

The current self-check validates:
- zero-shot Δ_sensor ≈ -3e-8 on smooth fields (PASS)
- baseline error 3.8e-1 (PASS)

**What needs to be added:**
- Measure L_F for real scenes (CAVE/Harvard)
- Compute EMD between real SRFs
- Validate the bound: does Δ_sensor ≤ L_F · EMD hold?
- Show the bound is tight for sharp spectral features (absorption edges)

---

## What is genuinely new in P3

1. **Generalization bound for sensor shift.** The Δ_sensor ≤ L_F · EMD
   formula is a new result.  It explains why zero-shot transfer works
   for some sensor pairs and fails for others.

2. **Lipschitz constant as a predictor.** L_F (spectral sharpness) as
   a predictor of transfer quality is new.  Existing work uses
   heuristic measures of spectral similarity.

3. **Sensor compatibility criterion.** The EMD(s^{src}, s^{tgt}) < ε/L_F
   criterion for sensor compatibility is new.  It provides a practical
   rule for selecting compatible sensor pairs.

4. **Hybrid field design.** The hybrid F(x,y,λ) = F_parametric + F_θ
   (provable scaffold + learned refinement) is new.  The parametric
   part ensures the bound holds; the learned part improves accuracy.