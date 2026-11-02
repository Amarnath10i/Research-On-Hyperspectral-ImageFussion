# Results

| Path | Contents |
|---|---|
| `q1/<job>/` | One folder per job of `methods/puformer/kaggle/jobs.py`: `results.json` (test metrics, plain and self-ensemble, per image), `history.json`, `gaps.json` (consistency, operator swap), `cond_eval.json` (changed observation models), `train.log`. Prefixes: `bench-` equal-budget benchmark, `abl-` ablation, `cond-` operator-family training, `off-`/`rec-` training recipes, `long-` 8 h budget, `arch-`/`sa-`/`spec-`/`scl-` architecture variants, sensors and scales |
| `q1/msi-value-<dataset>/` | Information budget of the HR-MSI (`msi_value.py`) |
| `q1/novelty/` | Architecture facet matrix and semantic similarity (`paper/puformer/novelty_check.py`) |
| `q1/*.kernel.log` | Logs of the Kaggle session kernels |
| `puformer_chikusei_x4_v3/` | Long-trained PUFormer on Chikusei ×4 (58.20 dB): outputs, TIP'26 Table IV ranking, Q2n analysis, robustness study |
| `puformer_chikusei_x4/` | PUFormer first run (57.99 dB): outputs, full TIP'26 Table IV comparison, robustness study |
| `puformer_pavia_x4/` | Long-trained PUFormer on Pavia Centre ×4 (50.69 dB): robustness, paper and SAM analyses |
| `PROPOSAL_COMPARISON.csv` | Every proposal's metrics, protocol and status in one table |
| `kaggle_runs/multidataset/out/` | Final multi-dataset study (CAVE, Harvard, Pavia, Chikusei, cross-domain) |
| `checkpoints/` | UnfoldFusion (KrylovNet) run report and Kaggle checkpoint-dataset stubs |
| `kaggle_runs/` | Kaggle run records of UnfoldFusion and the multi-dataset study |

Model weights are not tracked (see `.gitignore`). Session results are fetched with
`methods/puformer/kaggle/fetch_results.py`, which merges evaluation files by condition instead of overwriting them.
