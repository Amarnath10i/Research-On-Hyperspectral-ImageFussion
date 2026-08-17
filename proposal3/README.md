# proposal3 — P3 paper: sensor-independent continuous scene field

**Question:** can one scene field be fused for *any* sensor?  **Object:** a
continuous field `F(x,y,lambda)` observed through per-sensor operators
`O_s`, with the hold-out-sensor gap `delta_sensor = E_unseen - E_seen` as the
headline metric.

## Present work — `field/` (non-neural scaffold, selfcheck ALL PASS)

Arbitrary resolution is NOT the claim (crowded: CLoRF, IR&ArF, PGU-Net); the
claim is *sensor independence* — one field fitted on sensors A,B renders for
an unseen sensor C zero-shot:

- `field.py` — `SceneField`: Gaussian spectral bumps over `lambda in [0,1]`
  times cosine spatial modes; `render(lambda, hw)`.
- `sensors.py` — `Sensor` (spectral SRF + spatial blur/decimate), a **linear
  operator** on the field's coefficients; `linear_map` (dense, exact),
  `fit_field` (one least-squares solve).
- `metrics.py` — `relative_error`, `delta_sensor`.
- `selfcheck.py` — operators linear 2.6e-7; joint fit + zero-shot
  `E_seen 7e-6`, `E_unseen 7e-6`, `delta_sensor ~ -3e-8`; nearest-band
  baseline `E(C|A) = 3.8e-1` (five orders worse); under-specified family
  `E_unseen 0.95`, `delta_sensor +0.13`.
- `docs/ARCHITECTURE.md`, `results/RESULTS.md`.

```powershell
$env:PYTHONPATH="common;proposal1;proposal3;proposal5"; python -c "import proposal3.field as f; f._selfcheck.run_all()"
```

## Legacy (prior Q1 architectures, selfchecked, superseded)

| package | architecture | mechanism |
|---|---|---|
| `continuumfusion/` | ContinuumFusion | implicit representation, arbitrary scale factor |
| `nsp/` | NSP | learned cross-spectral PDE fusion |