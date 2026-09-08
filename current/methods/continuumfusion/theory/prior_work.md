# P3: How we differ from prior work

## The positioning paragraph (for the paper)

> **Continuous spectral fields for HSI fusion** have been studied
> (NeSSR IJCV 2024/25, CLoRF 2024, SINR 2023, INR-HSISR TGRS 2022)
> but none provide generalization bounds for sensor shift.  Our
> contribution is: (1) a formal bound Δ_sensor ≤ L_F · EMD relating
> transfer error to spectral sharpness and SRF distance; (2) a hybrid
> field F_parametric + F_θ that provably satisfies the bound while
> learning scene-specific details; (3) a sensor compatibility criterion
> that predicts which sensor pairs are compatible.

## What already exists

| Method | Reference | What it does | Our difference |
|---|---|---|---|
| NeSSR | IJCV 2024/25 | Neural implicit spectral representation for HSI reconstruction | No sensor-shift analysis; no generalization bound |
| CLoRF | 2024 | Continuous low-rank representation for HSI fusion | Linear model; no neural refinement; no sensor-shift bound |
| SINR | 2023 | Spectral implicit neural representation | Per-scene optimisation; no transfer; no bound |
| INR-HSISR | TGRS 2022 | Implicit neural representation for HSI super-resolution | No sensor-shift analysis; no generalization bound |
| SIREN | CVPR 2020 | Sinusoidal representation networks | General architecture; not HSI-specific; no sensor-shift bound |

## What is genuinely new in P3

1. **Generalization bound.** Δ_sensor ≤ L_F · EMD is a new result.
   Existing continuous field methods lack formal transfer guarantees.

2. **Hybrid field design.** F_parametric + F_θ combines a provable
   scaffold (parametric, satisfies the bound) with learned refinement
   (neural, improves accuracy).  Existing methods use either pure
   parametric or pure neural.

3. **Sensor compatibility criterion.** EMD(s^{src}, s^{tgt}) < ε/L_F
   is a new practical rule for selecting compatible sensor pairs.

4. **Lipschitz constant as predictor.** L_F (spectral sharpness) as a
   predictor of transfer quality is new.  Existing work uses heuristic
   spectral similarity measures.