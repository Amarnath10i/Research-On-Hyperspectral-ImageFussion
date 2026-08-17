# proposal2 — P2 paper: observation-identifiable spectral complexity (priority 1)

**Question:** can the intrinsic spectral dimensionality of an HSI–MSI fusion
problem be identified from the observations alone?  **Object:** the
observation-identifiable rank `r_id = rank(R^T U_r)`, with
`|r_hat - r_id|` as the headline metric.

## Present work — `rankest/`, `metrics/`, `theory/`, `experiments/` (selfcheck ALL PASS)

A non-neural estimator that answers "how many spectral degrees of freedom do
the observations support?", not "how many does the scene have?":

- `rankest/generator.py` — `RankScene`: controlled synthetic scenes
  (orthonormal spectral basis, orthogonal low-frequency spatial modes, per-band
  blur+downsample, Gaussian-bump SRF, shared noise); `r_id_true` by SVD.
- `metrics/identifiable_rank.py` — noise estimation from the LR-HSI trailing
  singular values, Gavish–Donoho optimal hard threshold
  (`tau = omega(beta)·sigma·sqrt(n)`) on the MSI singular values, relative
  floor in the noise-free regime; `estimate_ranks` pipeline.
- `metrics/rank_error.py` — `|r_hat - r_id|`, `within_tol`, summariser.
- `theory/` — `rank_definition.md` (four rank notions), `identifiable_rank.md`
  (equivalence `X1 ~_A X2 <=> A(X1)=A(X2)`), `assumptions.md`.
- `experiments/` — four sweep scripts (run via `python -m`):
  `synthetic_rank_sweep`, `noise_sweep`, `band_count_sweep`, `srf_sweep`.

Headline numbers: exact rank recovery `|Δr|=|Δr_id|=0` for r = 3..30;
`r_id_hat` monotone in SNR (12→5) and in MSI band count; SRF-overlap tracking
(12→12→11→6); noise estimated within 3–5%.

```powershell
$env:PYTHONPATH="common;proposal1;proposal2;proposal5"
python -c "import proposal2.rankest as r; r._selfcheck.run_all()"
python -m proposal2.experiments.synthetic_rank_sweep   # etc.
```

## Legacy (prior Q1 architectures, selfchecked, superseded)

| package | architecture | mechanism |
|---|---|---|
| `krylovnet/` | KrylovNet | unrolled preconditioned GMRES fusion |
| `unfoldfusion/` | UnfoldFusion | unrolled variational solver, physics in the architecture |