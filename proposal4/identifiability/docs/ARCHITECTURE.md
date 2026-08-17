# P4 — joint scene/sensor identifiability (capstone, theory-first)

The restructured program's capstone: not another fusion network, but a
**controlled study of when HSI-MSI fusion is identifiable at all**.  The
advisor's guidance was explicit: theory/identifiability conditions first,
identifiability phase diagram, a simulator controlling (E, A, D, R); full joint
recovery of (X, D, R, E, A) only after P1-P3.  This package is that study's
non-neural core.

## The simulator

`simulate()` builds a scene with known intrinsic spectral rank `r` (RankScene,
orthogonal low-frequency spatial modes, so no aliasing-driven rank collapse)
and observes it through a combined operator `A=[D;R]` (P1's
`CombinedOperator`) with controlled knobs:

- `rank`      - scene spectral complexity
- `msi_bands` - spectral sampling (M)
- `srf_width` - spectral overlap of the MSI response
- `snr_db`    - shared observation noise

Two complementary identifiability measures, one from P2 and one from P1:

    score     = r_id_hat / r     (spectral DOF the observations pin down)
    null_frac = ||P_N X|| / ||X|| (fraction of the scene left ambiguous)

Regime by score: `I` >= 0.75, `N` < 0.25, `W` between.

## Phase diagram (`python -m proposal4.identifiability.phasediagram`)

**MSI band count x SNR** (r = 8, srf 0.02): M=4 is Weakly at best (4 bands
cannot pin 8 spectral DOF, even noise-free); M>=8 is Identifiable; noise only
hurts the marginal M=8 row.

**SRF overlap x SNR** (M = 8): narrow SRF stays Identifiable; overlap pushes
cells Weakly as SNR drops, and the clean null_frac grows from 0.459 (width
0.02) to 0.484 (width 0.50).  Both axes of identifiability move the way the
theory requires.

## Verified properties (selfcheck)

| # | claim | result |
|---|-------|--------|
| 1 | one robust cell per regime (I/W/N) | clean/M31 -> 8I; SNR5/M4/w0.5 -> 1N; SNR5/M16 -> 8I; SNR5/M4/w0.06 -> 4W |
| 2 | r_id_hat non-increasing in SNR (all M) | PASS |
| 3 | r_id_hat non-decreasing in M (fixed SNR) | PASS |
| 4 | null_frac monotone in M; endpoint-increase with SRF overlap | 0.492->0.254; 0.459->0.484 |
| 5 | corr(score, null_frac) over M x SNR grid | -0.70 (agree) |

## Next stage (the actual P4 papers)

- Vary scene spectral diversity, degradation mismatch, registration error, and
  spatial scale; extend the phase diagram to the joint (X, D, R, E, A) recovery
  question - i.e. which parameters are identifiable under which (rank, M, SRF,
  SNR) conditions, not just the scene.
- The simulators here ARE the ground truth for any learned joint estimator that
  P1-P3 produce; P4's evaluation is to show a learned joint recovery respects
  the regime boundaries (fails where the diagram says Non-identifiable).