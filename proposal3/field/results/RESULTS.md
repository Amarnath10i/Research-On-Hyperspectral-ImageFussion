# P3 scaffold — results (proposal3/field/selfcheck.py)

Status: **ALL PASS** (CPU).  Non-neural scaffold: `SceneField` (6 bumps, 4
modes) observed by A (scale 2, 8 bands), B (scale 4, 3 bands), hold-out C
(scale 2, 6 bands, different SRF centres/width).  "Fitting" = one
least-squares solve.

## Checks

| # | claim | value |
|---|-------|-------|
| 1 | `O_s` linear for A, B, C | max rel err 2.6e-07 |
| 2 | fit A,B -> A,B and C zero-shot | E_seen 6.96e-06, E_unseen 6.93e-06, delta -3.1e-08 |
| 3 | nearest-band copy of A on C | E = 3.79e-01 |
| 4 | under-specified family (4 bumps, 2 modes) | E_unseen 9.49e-01, delta +1.26e-01 |

## Headline: delta_sensor

- Correct family (6,4): `delta_sensor ~ -3e-8` — the hold-out sensor C is
  predicted to solver precision, no re-fitting.  Sensor-independent.
- Under-specified family (4,2): `delta_sensor = +0.126`, `E_unseen = 0.95` —
  the model is NOT sensor-independent; it cannot extrapolate the spectral /
  spatial content it never saw.

## Headline: vs non-field baseline

Nearest-band copy of sensor A (band matching by SRF centre, same spatial
scale) predicts C with relative error 3.8e-1 — five orders of magnitude worse
than the field's 7e-6.  The shared-lambda-continuum coupling is what makes
zero-shot possible.

## Caveat / next

These are structural identities on a synthetic, parametric field.  The claims
that matter for the paper — does a *learned* field on real sensor pairs keep
`delta_sensor` small, and does `delta_sensor` track spectral-diversity /
registration degradation — require the network stage on Kaggle.