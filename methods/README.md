# Methods

Each folder is a self-contained package with its own README (method, status, how to run).

| Folder | Method | Idea | Status |
|---|---|---|---|
| [puformer](puformer/) | **PUFormer**, Physics-Unfolded Transformer | Exact data-fidelity steps + Restormer spectral-attention prior; consistency / mismatch / noise evaluation | **Active**: Chikusei ×4 SOTA attempt |
| [nullfusion](nullfusion/) | NullFusion (P7) | `X = pinv(y) + P_N(f(y))`, so the network only fills the null space and cannot contradict the observations | Trained (CAVE Nikon 50.31 dB) |
| [krylovnet](krylovnet/) | KrylovNet / KrylovNet-P (P2) | Unrolled preconditioned GMRES; identifiable spectral rank `r_id = rank(Rᵀ U_r)` | Trained |
| [daetf](daetf/) | DAETF-Net (P1) | Admissible-ambiguity manifold; hallucination metric H | Trained |
| [continuumfusion](continuumfusion/) | ContinuumFusion (P3) | Continuous scene field `F(x,y,λ)`; sensor-shift bound | Self-checks pass |
| [zerofusion](zerofusion/) | ZeroFusion (P4) | Identifiability phase diagram | Self-checks pass |
| [spectralflow](spectralflow/) | SpectralFlow / ManifoldFlow | Null-space generative models | Self-checks pass |
| [dacf](dacf/) | DACF (P9) | Degradation-adaptive conditional flow | Code only |
| [ason](ason/) | ASON (P8) | Adaptive spectral operator network | Code only |
| [consistentflow](consistentflow/) | ConsistentFlow (P6) | Consistency-distilled one-step sampler | Superseded by NullFusion |

The shared data, degradation, SRF and metric code lives in [`../common/hsifusion`](../common/hsifusion).
