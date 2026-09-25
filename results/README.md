# Results

| Path | Contents |
|---|---|
| `puformer_chikusei_x4/` | PUFormer Chikusei ×4 run: outputs, full TIP'26 Table IV comparison, robustness study |
| `PROPOSAL_COMPARISON.csv` | Every proposal's metrics, protocol and status in one table |
| `kaggle_runs/multidataset/out/` | Final multi-dataset study (CAVE, Harvard, Pavia, Chikusei, cross-domain) |
| `checkpoints/`, `archive/` | Logs and result JSONs from earlier runs |

Model weights are not tracked (see `.gitignore`); retrain from `experiments/` or `methods/`.
PUFormer training writes to `methods/puformer/out/` (ignored); published runs are copied into `puformer_chikusei_x4/`.
