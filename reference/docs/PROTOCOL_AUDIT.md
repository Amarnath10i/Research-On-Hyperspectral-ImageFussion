# Protocol Audit — one protocol, so every number is comparable

This document is the audit trail that makes any two rows in this repository
comparable. The benchmark (`existing/results/BENCHMARK.md`) is explicit that the
ten published methods were each run under their own protocol — scale factors of
×4, ×8, ×16 and ×32, different normalisations, different metric
implementations. No cross-method ranking survives that. This file fixes the
protocol once and maps every rule to the code that enforces it.

**Status:** the rules below are implemented and enforced by the shared library.
The *Stage 0* audit (classical baselines matching published behaviour) is the
first thing to run when a real dataset is attached; see §5.

---

## 1. The fixed protocol

| Rule | Value | Enforced in |
|---|---|---|
| Scale factor | `scale = 4` for every comparison | `Config.scale` (all proposals) |
| Metric data range | constant `1.0`, **never** the per-image maximum | `daetf/metrics.py`; `hsifusion/metrics.py` |
| Degradation for evaluation | fixed anisotropic Gaussian (`eval_sigma=1.2`) + decimation, zero-padded | `FixedDegradation` (`daetf/degrade.py`, `hsifusion/degrade.py`) |
| SRF | recovered from data by least squares, `min_S ‖HSI·S − RGB‖²` | `estimate_srf` (`daetf/data.py`, `hsifusion/data.py`) |
| Full scenes | Hann-weighted overlapping tiles, no centre crops | `tiled_inference` (`daetf/engine.py`, `hsifusion/engine.py`) |
| Per-scene rows | retained for paired tests | `evaluate_dataset(..., return_rows=True)` |
| ERGAS scale argument | matches the true downsampling factor | `metric_ergas(pred, ref, scale)` |

Anything that does not satisfy every row in this table must be labelled
`protocol-different` and cannot appear in the same table as a protocol-equal row.

---

## 2. The metrics

- **PSNR** — `10·log10(1/mse)`, `data_range=1.0`. Reported, but never headline:
  the benchmark shows PSNR points the wrong way on domain shift (it *improves*
  on harder data while SAM collapses).
- **SSIM** — Gaussian window 11×11, σ=1.5, averaged over bands.
- **SAM** — mean spectral angle in degrees, ignoring degenerate zero pixels.
- **ERGAS** — `100/scale · sqrt(mean((RMSE_b/μ_b)²))` with `scale` the true factor.

Headline metrics are **SAM and ERGAS** (they are the ones that fail under domain
shift). The Q1 ladder and comparison tables lead with them; PSNR is secondary.

## 3. Consistency errors (the structural claims)

Both proposals make algebraic claims about the observation operator, so the
protocol also scores them:

- **LR-consistency** `‖D(ŷ) − X‖ / ‖X‖` — with the range/null projection this
  is solver tolerance (~1e-6); without it, free drift (~1e-2). Reported under
  the **fixed** evaluation operator. The estimated-operator version is reported
  separately in blind runs, where `D̂(ŷ) = X` holds by construction but
  `D(ŷ) ≈ X` against the true sensor does not.
- **SRF-consistency** `‖S(ŷ) − M‖ / ‖M‖` — whether the output re-explains the
  multispectral guide.

These are added to every per-scene row and every ladder table.

---

## 4. Statistics

- **Paired** tests only — every method is scored on the same scenes, so pairing
  is the correct and far more sensitive design on 10–20 scenes.
- **Wilcoxon signed-rank** p-value and **Cohen's d** against every baseline.
- **Bootstrap 95% CIs** on every reported mean.
- **≥3 seeds** for the final contenders, reported as mean ± std.

Implementation: `daetf/experiments.py` (P1) and `spectralflow/experiments.py`
(P5). numpy-only, runs on a bare Kaggle image.

---

## 5. Stage 0 audit — is the pipeline itself correct?

Run the classical baselines under the protocol **before** training anything:

```bash
python - <<'PY'
import daetf, daetf.baselines as B
from daetf.config import Config
cfg = Config.paper_core().resolve()
srf = daetf.estimate_srf(cfg.source_root, "Train", cfg)
B.evaluate_all_baselines(cfg.source_root, cfg, srf, device="cuda")
PY
```

Checklist (from the roadmap's weeks 1–2):

- [ ] **Bicubic** PSNR/SAM/ERGAS are the interpolation floor. If bicubic is
      *better* than a learned method on your own dataset, either the dataset is
      too easy or the learned method is broken — do not paper over it.
- [ ] **GSA** and **Subspace-LS** reproduce classical behaviour: high spatial
      fidelity (PSNR/SSIM), spectral errors visible as SAM/ERGAS when the SRF
      is imperfect.
- [ ] **SRF recovery** is checked against a known synthetic response
      (error ~2e-8 in the selfchecks). Re-verify on the real dataset.
- [ ] Metric sanity: a near-perfect reconstruction scores SAM ≈ 0, PSNR > 50.
- [ ] The numbers must be stable across the two datasets — a method's *ranking*
      may change, but the *pipeline* must not produce nonsense (e.g. ERGAS
      jumping by orders of magnitude solely because of a normalisation bug).

**If the classical baselines do not look sane, stop.** The pipeline is broken;
no learned number below them can be trusted.

---

## 6. The Q1 ladder

### P1 (DAETF-Net / PSE) — `daetf.experiments.Q1_LADDER`

| Stage | Adds | Question answered |
|---|---|---|
| 0 | Bicubic, GSA, Subspace-LS | is the protocol correct? |
| 1 | plain residual fusion (bicubic + residual) | is MSI-guided learning useful? |
| 2 | + physical losses | do constraints help? |
| 3 | + range/null projection | does the split help? |
| 4 | + projective spectral embedding | is PSE load-bearing? |
| 5 | + blind degradation conditioning | does the estimate help? |
| 6 | one optional module at a time | does each earn its cost? |

Run with `Config.paper_core()` and `run_q1_ladder`. **Stop/go:** advance only
when Stage 3 beats Stage 2 on SAM/ERGAS and Stage 4 beats Stage 3. If not,
report the negative result — do not add modules.

### P5 (SpectralFlow) — `spectralflow.experiments.STAGE_LADDER`

| Stage | Switch | Question answered |
|---|---|---|
| 1 | plain DDIM (`use_projection=False`) | does the prior learn the manifold? |
| 2 | + null-space projection (`use_projection=True`) | does the projection help? |
| 3 | + MSI guide ablated (`use_msi_guide=False`) | is the guide load-bearing? |

Crucially, Stage 1 vs Stage 2 use the **same trained checkpoint** — the
projection lives in the sampler, so the comparison is cheap and attributable.

---

## 7. Cross-domain protocol

- **Inductive** split: source-train / target-test with **no scene overlap**.
  Transductive settings (e.g. target seen at train time) must be stated.
- Report **ΔSAM and ΔERGAS** (gap on target minus in-domain), not just the
  target numbers — the claim is that the *gap* is bounded, not that the method
  is best on everything.
- P4 (ZeroFusion) is the control arm: it never trains on a source domain, so
  any learned method must beat it cross-domain or concede that per-scene
  optimisation is the better method.

---

## 8. What is forbidden

- Reusing published numbers from papers with a different scale factor,
  normalisation, crop strategy, or metric implementation. (The
  `SOTA_REFERENCE` table in `daetf/experiments.py` is currently unverified
  placeholder data and must be removed before any submission.)
- Per-image maximum normalisation for PSNR.
- Reporting mean-only numbers without paired statistics and CIs.
- Claiming "consistency" for the *estimated* operator as if it were the *true*
  sensor operator.
