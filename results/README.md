# Results

| Path | Contents |
|---|---|
| `puformer_chikusei_x4_v3/` | PUFormer v3 (58.20 dB): outputs, TIP'26 Table IV ranking, Q2n analysis, robustness study |
| `puformer_chikusei_x4/` | PUFormer first run (57.99 dB): outputs, full TIP'26 Table IV comparison, robustness study |
| `PROPOSAL_COMPARISON.csv` | Every proposal's metrics, protocol and status in one table |
| `kaggle_runs/multidataset/out/` | Final multi-dataset study (CAVE, Harvard, Pavia, Chikusei, cross-domain) |
| `checkpoints/` | UnfoldFusion (KrylovNet) run report and Kaggle checkpoint-dataset stubs |
| `kaggle_runs/` | Kaggle run records of UnfoldFusion and the multi-dataset study |

Model weights are not tracked (see `.gitignore`); retrain from `experiments/` or `methods/`.
PUFormer training writes to `methods/puformer/out/` (ignored); published runs are copied into `puformer_chikusei_x4*/`.
