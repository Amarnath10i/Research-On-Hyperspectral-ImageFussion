# Baselines (third-party code)

Upstream implementations of published methods, used to reproduce comparison numbers. Each
subfolder keeps its original README and license; please cite the original papers.

| Folder | Method | Paper |
|---|---|---|
| [bdt](bdt/) | BDT, Bidirectional Dilation Transformer | IJCAI 2023 |
| [dspnet](dspnet/) | DSPNet, Dual Spatial–Spectral Pyramid Network with Transformer | IEEE TGRS 2023 |
| [3dt-net](3dt-net/) | 3DT-Net, 3D-CNN + Transformer prior | Information Fusion 2023 |
| [mogdcn](mogdcn/) | MoG-DCN, Model-Guided Deep HSI Super-Resolution | IEEE TIP 2021 |
| [ssrnet](ssrnet/) | SSR-NET, Spatial–Spectral Reconstruction Network | IEEE TGRS 2021 |
| [efficient-mif](efficient-mif/) | Efficient multi-source image fusion framework (FeINFN and others) | NeurIPS 2024 and related |

Pretrained weights (`*.pth`, `*.pth.tar`) are not tracked. Get them from the upstream
repositories or retrain with `experiments/scripts/train_<method>.py`.
