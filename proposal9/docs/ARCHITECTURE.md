# DACF Architecture

## Design Principles

1. **Degradation-Aware**: Estimates degradation from observations, not from metadata
2. **Flow-Based**: Normalizing flow for flexible density estimation + generation
3. **Self-Supervised Adaptation**: No paired GT needed at test time
4. **Band-Agnostic**: Same architecture for 31, 103, 128, or any band count

## Modules

### DegradationEncoder
- Input: LR-HSI [B, C, H_lr, W_lr] + MSI [B, M, H, W]
- Architecture: dual-path CNN → MLP → 64-dim code
- Purpose: captures blur kernel, noise level, SRF characteristics
- Key: first conv handles any channel count

### ConditionalFlow
- Architecture: 8 affine coupling layers with FiLM conditioning
- Each coupling layer: split channels → predict scale+shift for other half
- FiLM: γ, β from degradation code → modulate coupling features
- Sampling: rectified-flow-style interpolation (4 steps default)

### NullProjector
- Enforces D(fused) = LR-HSI (blur+downsample consistency)
- Operation: v_proj = v - D^T(DD^T)^{-1}Dv
- Simplified: v_proj = v - blur(D(v))

## Complexity

- Parameters: ~0.5M (encoder: 50K, flow: 450K)
- FLOPs: depends on input size and flow_steps
- Memory: ~1GB for 103-band 64×64 patch on T4
- Adaptation time: ~0.5s per scene (100 iterations)

## Why It Works

The degradation encoder learns to extract degradation features from the observations. During training with randomized degradations, the flow network learns to adapt its behavior based on the code. At test time:
1. Encoder sees real degradation → produces code
2. Flow conditions on code → adapts fusion behavior
3. Phase 2 fine-tunes encoder to better match real degradation

This is more efficient than UTAL (which adapts the entire network) and more flexible than fixed-degradation methods.
