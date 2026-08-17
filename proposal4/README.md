# proposal4 — P4 paper: joint scene/sensor identifiability (capstone)

**Question:** when is an HSI–MSI fusion problem identifiable *at all*?
**Object:** a controlled identifiability phase diagram
(Identifiable → Weakly → Non-identifiable) over the knobs the operator
depends on (scene rank, MSI band count, SRF overlap, noise, spatial scale).
Highest-risk paper: theory first, joint recovery of `(X, D, R, E, A)` only
after P1–P3.

## Present work — `identifiability/` (non-neural scaffold, selfcheck ALL PASS)

- `simulator.py` — `simulate()`: builds a scene of known spectral rank
  `r` (RankScene) observed through `A=[D;R]` (P1's `CombinedOperator`) with
  controlled knobs; reports two complementary identifiability measures:
  `score = r_id_hat / r` (P2: spectral DOF pinned down) and
  `null_frac = ||P_N X||/||X||` (P1: fraction of the scene left ambiguous);
  regime labels `I` / `W` / `N`.
- `phasediagram.py` — `python -m proposal4.identifiability.phasediagram`:
  (MSI bands × SNR) and (SRF overlap × SNR) tables.
- `selfcheck.py` — one robust cell per regime; `r_id_hat` monotone in SNR and
  in M; `null_frac` monotone in M and endpoint-increase with overlap; the two
  measures anti-correlate across the grid (corr −0.70).
- `docs/ARCHITECTURE.md`, `results/RESULTS.md`.

The phase diagram is the falsifiable target any learned joint estimator must
respect: it must report confidence in the Non-identifiable regime, and fail
where the diagram says it must.

```powershell
$env:PYTHONPATH="common;proposal1;proposal2;proposal4;proposal5"
python -c "import proposal4.identifiability as i; i._selfcheck.run_all()"
python -m proposal4.identifiability.phasediagram
```

## Legacy (prior Q1 architectures, selfchecked, superseded)

| package | architecture | mechanism |
|---|---|---|
| `zerofusion/` | ZeroFusion | per-scene unmixing, no training set |
| `graphdip/` | GraphDIP | superpixel graph deep image prior |