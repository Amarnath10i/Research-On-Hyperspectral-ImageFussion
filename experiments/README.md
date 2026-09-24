# Experiments

| Folder | Contents |
|---|---|
| `scripts/` | Standalone training scripts, e.g. `train_bdt.py`, `train_dspnet.py`, `train_feinfn.py`, `train_ssrnet.py` (baseline reproductions on CAVE ×4) and `train_nullfusion_*`, `train_krylovnet_chikusei.py`, `train_diffusion_nullfusion_*` (our methods) |
| `notebooks/` | Kaggle notebooks for the runs above; `*_push/` folders hold the exact pushed kernel and its log |
| `kaggle/` | Kaggle kernel folders (`kernel-metadata.json` + notebook) |

PUFormer's own Kaggle notebook builder is in [`methods/puformer/kaggle/`](../methods/puformer/kaggle/).
