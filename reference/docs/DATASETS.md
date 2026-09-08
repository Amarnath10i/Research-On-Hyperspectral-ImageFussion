# Dataset Usage of the IEEE HSI–MSI Fusion Reading List

Companion to `hsim_fusion_ieee_if9.bib`. Legend: **Y** = dataset used,
**N** = not used, **?** = not confirmed from the paper abstract / code / README.

**Answer to "are all of these on CAVE, Harvard, Chikusei, PaviaU?" → No.**
The universal *simulated* benchmarks are CAVE + Harvard (Wald protocol) and the
universal *real* benchmarks are Chikusei + PaviaU. Nearly all classical and
mainstream supervised methods use all four. The exceptions are the
**unregistered / blind / real-scene** methods, which deliberately avoid the
perfectly-registered simulated CAVE/Harvard and instead use Chikusei, PaviaU,
Houston, ICVL, etc.

| # | Key | Paper (short) | CAVE | Harvard | Chikusei | PaviaU | Notes |
|---|-----|---------------|------|---------|----------|--------|-------|
| 1 | qu2025irarf | IR&ArF | ? | ? | ? | ? | Unregistered/arbitrary-res; real-scene method, datasets not stated; likely real Chikusei/PaviaU (unconfirmed) |
| 2 | shi2025vdmu | VDMUFusion | N | N | N | N | General multi-modal framework (IR/visible, medical, multi-exposure); not HSI-MSI/CAVE-Harvard benchmark |
| 3 | chen2025rangenull | UnNull / Range-Nullspace | Y | ? | ? | ? | Code README: CAVE; others unconfirmed |
| 4 | wang2025deeptucker | DTDNML (Deep Tucker) | N | N | Y | Y | PaviaU + Chikusei (+ WaDC, SanDiego) |
| 5 | qu2024regfusionconsistency | Reg-Fusion Consistency | N | N | N | Y | PaviaU (+ Salton Sea, Mississippi Gulfport) |
| 6 | liu2025amsfnet | AMSF-Net | Y | N | Y | N | CAVE, Chikusei, Houston, WorldView-3 |
| 7 | huo2026mian | MIAN | ? | ? | ? | ? | Benchmark list not found |
| 8 | wang2023difiv | DIFIV | N | N | N | N | Real AVIRIS/Sentinel-2 pairs (Ivanpah, Lake Tahoe) |
| 9 | chi2025csgav | CSGAV | N | N | N | Y | Houston, Botswana, PaviaU |
| 10 | xie2022mhfnet | MHF-Net | Y | Y | Y | Y | Standard CAVE/Harvard + real Pavia/Chikusei |
| 11 | ran2023guidednet | GuidedNet | Y | Y | N | N | Code test data: CAVE + Harvard (natural-image HISR) |
| 12 | tian2024hmpnet | HMPNet | ? | ? | ? | ? | Benchmark list not confirmed |
| 13 | fang2024dunet | DUNET | N | N | Y | N | ICVL, Chikusei, Houston |
| 14 | guo2023scanet | SCANet | N | N | Y | Y | PaviaU, Chikusei, PYLake |
| 15 | dong2021modelguided | Model-Guided Deep HSI SR | Y | Y | Y | Y | Standard Wald-protocol sets |
| 16 | dian2024crossfusion | Spectral SR Cross-Fusion | Y | Y | Y | Y | Standard sets |
| 17 | hu2024spectralprior | Exploring Spectral Prior | Y | Y | Y | Y | Standard sets |
| 18 | fang2024cs2dips | CS2DIPs | Y | Y | Y | Y | Standard sets |
| 19 | he2025arbpansharp | Arbitrary-Resolution Pansharpening | Y | Y | Y | Y | Standard sets |
| 20 | yokoya2012cnmf | CNMF | Y | ? | Y | Y | CAVE + real Chikusei/Pavia (Harvard atypical) |
| 21 | simoes2015hysure | HySure | Y | Y | Y | Y | CAVE, Harvard + real Pavia/Chikusei |
| 22 | wei2015sparse | Wei Sparse | Y | Y | Y | Y | CAVE, Harvard + real Pavia/Chikusei |
| 23 | zhou2020integrated | Integrated Reg-Fusion | Y | Y | Y | Y | CAVE, Harvard + real Pavia/Chikusei |

## Patterns
- **All four (CAVE+Harvard+Chikusei+PaviaU):** rows 10, 15–19, 21–23, and
  CNMF mostly (row 20, Harvard ?). These are the "textbook" benchmark sets.
- **Simulated only (CAVE+Harvard, no Chikusei/PaviaU):** GuidedNet (11) — it is a
  natural-image HISR framework.
- **Real only (Chikusei/PaviaU/Houston/ICVL, no CAVE/Harvard):** Deep Tucker (4),
  Reg-Fusion (5), AMSF-Net (6, +Houston/WV3), DUNET (13), SCANet (14), CSGAV (9).
  These are *unregistered / blind / remote-sensing* methods that avoid the
  perfectly-registered simulated pairs.
- **Different real data entirely:** DIFIV (8, AVIRIS/Sentinel-2), VDMUFusion (2,
  other modalities).
- **Unconfirmed (?):** IR&ArF (1), MIAN (7), HMPNet (12) — verify against the
  full paper before citing.
