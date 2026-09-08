# ASON Architecture

## Design Principles

1. **Rectified Flow on Consistent Set**: Instead of diffusion, we use rectified flow matching (ODE) constrained to the set of signals consistent with the observations: {y | D(y) = X, S(y) = M}.

2. **Degradation-Aware Conditioning**: A DegradationCode encoder captures the specific degradation characteristics of each input pair, enabling adaptive fusion.

3. **Null-Space Projection**: At each ODE step, the velocity is projected onto the tangent space of the consistent set, ensuring the trajectory never leaves the feasible region.

## Module Details

### DegradationCode
- Input: LR-HSI [B, C_lr, H_lr, W_lr] + MSI [B, M, H_lr, W_lr]
- Architecture: dual-path CNN encoder → MLP → 64-dim code
- Purpose: encodes degradation type, noise level, blur characteristics

### VelocityNet
- Input: current state y [B, C, H, W] + MSI [B, M, H, W] + time t + code [B, 64]
- Architecture: concat → Conv blocks with FiLM conditioning on code
- Output: velocity field v [B, C, H, W]
- FiLM: γ, β from code → scale/shift intermediate features

### RangeNullProjector
- Purpose: project velocity v to null-space of degradation operator D
- Operation: v_proj = v - D^T(D D^T)^{-1} D v
- Simplified: v_proj = v - blur(D(v)) (iterative refinement)

## Complexity

- Parameters: ~0.3M (hidden=24, blocks=3)
- FLOPs: depends on input size and sample_steps (4 default)
- Memory: ~2GB for 128-band 256×256 patch on T4
