# Experiments

| Folder | Contents |
|---|---|
| `scripts/` | Standalone training scripts for our methods: `train_nullfusion_*`, `train_krylovnet_chikusei.py`, `train_diffusion_nullfusion*`, `train_chikusei_sota.py` |
| `notebooks/` | Kaggle notebooks for the runs above; `*_push/` folders hold the exact pushed kernel and its log. `MultiDataset_Fusion_Study.ipynb` produced the multi-dataset results used by the paper |
| `kaggle/` | Kaggle kernel folders (`kernel-metadata.json` + notebook) |

PUFormer's own Kaggle notebook builder is in [`methods/puformer/kaggle/`](../methods/puformer/kaggle/).
