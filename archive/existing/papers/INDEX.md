# Paper ↔ notebook ↔ resource index

Each benchmarked method maps to one notebook per dataset, named for the method.
PDFs are not tracked in git (see [`.gitignore`](../../.gitignore)); drop them in
this directory using the **Paper file** name so the mapping stays one-to-one.

| Method | Paper file (drop PDF here) | CAVE notebook | Harvard notebook | Kaggle model source |
|---|---|---|---|---|
| AMGSGAN | `AMGSGAN.pdf` | [AMGSGAN_CAVE.ipynb](../notebooks/cave/AMGSGAN_CAVE.ipynb) | [AMGSGAN_Harvard.ipynb](../notebooks/harvard/AMGSGAN_Harvard.ipynb) | `amgsgan-model/frameworks/pytorch` |
| DBIN | `DBIN.pdf` | [DBIN_CAVE.ipynb](../notebooks/cave/DBIN_CAVE.ipynb) | [DBIN_Harvard.ipynb](../notebooks/harvard/DBIN_Harvard.ipynb) | `model-sf16-15k/frameworks/tensorflow2` |
| DHIF-Net | `DHIF-Net.pdf` | [DHIF-Net_CAVE.ipynb](../notebooks/cave/DHIF-Net_CAVE.ipynb) | [DHIF-Net_Harvard.ipynb](../notebooks/harvard/DHIF-Net_Harvard.ipynb) | `model-dhifnet/frameworks/pytorch` |
| Fusformer | `Fusformer.pdf` | [Fusformer_CAVE.ipynb](../notebooks/cave/Fusformer_CAVE.ipynb) | [Fusformer_Harvard.ipynb](../notebooks/harvard/Fusformer_Harvard.ipynb) | `model-fusformer/frameworks/pytorch` |
| IFCASformer | `IFCASformer.pdf` | [IFCASformer_CAVE.ipynb](../notebooks/cave/IFCASformer_CAVE.ipynb) | [IFCASformer_Harvard.ipynb](../notebooks/harvard/IFCASformer_Harvard.ipynb) | `ifcasformer/frameworks/pytorch` |
| LRU | `LRU.pdf` | [LRU_CAVE.ipynb](../notebooks/cave/LRU_CAVE.ipynb) | [LRU_Harvard.ipynb](../notebooks/harvard/LRU_Harvard.ipynb) | trained in-notebook |
| MoG-DCN | `MoG-DCN.pdf` | [MoG-DCN_CAVE.ipynb](../notebooks/cave/MoG-DCN_CAVE.ipynb) | [MoG-DCN_Harvard.ipynb](../notebooks/harvard/MoG-DCN_Harvard.ipynb) | `model-mogdcn/frameworks/pytorch` |
| PSRT | `PSRT.pdf` | [PSRT_CAVE.ipynb](../notebooks/cave/PSRT_CAVE.ipynb) | [PSRT_Harvard.ipynb](../notebooks/harvard/PSRT_Harvard.ipynb) | `model-psrt/frameworks/pytorch` |
| TSFN | `TSFN.pdf` | [TSFN_CAVE.ipynb](../notebooks/cave/TSFN_CAVE.ipynb) | [TSFN_Harvard.ipynb](../notebooks/harvard/TSFN_Harvard.ipynb) | `model-tsfn-epoch500/frameworks/tensorflow2` |
| UTAL | `UTAL.pdf` | [UTAL_CAVE.ipynb](../notebooks/cave/UTAL_CAVE.ipynb) | [UTAL_Harvard.ipynb](../notebooks/harvard/UTAL_Harvard.ipynb) | `model-utal/frameworks/pytorch` |

Model sources are Kaggle slugs under the `nikeshreddypatlolla/` account, as
referenced by the notebooks.

## Datasets

| Dataset | Kaggle slug | Layout | Split | Size |
|---|---|---|---|---|
| CAVE | `nikeshreddypatlolla/cave-dataset-2` | `Data/{Train,Test}/{HSI,RGB}/*.mat` | 20 / 12 | 512×512×31 |
| Harvard | `nikeshreddypatlolla/harvard-hsi-2` | `Data/{Train,Test}/{HSI,RGB}/*.mat` | 30 / 20 | 1040×1392×31 |
| CAVE (CASSI) | `nikeshreddypatlolla/cave-dataset-3` + `casformer-mask` | scene-numbered | 10 test | IFCASformer only |

Method implementations are largely drawn from
[Nikesh0907/hif-benchmarking](https://github.com/Nikesh0907/hif-benchmarking),
which the DBIN notebooks clone directly.

## Adding the PDFs

Paper titles are intentionally left out of this table rather than guessed. When
you add each PDF, record its exact title and citation next to the method here —
that keeps the index usable as the reference list for the manuscript, with no
fabricated bibliography entries to clean up later.
