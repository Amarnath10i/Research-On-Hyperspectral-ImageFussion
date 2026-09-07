"""GraphDIP - Proposal 4 (Q1 redesign).

Per-scene, fully self-supervised fusion: a GNN over a superpixel graph built
from the MSI guide is optimised for a *single* scene against a physics-only
objective (LR-HSI and MSI consistency), deep-image-prior style.  Message passing
implements non-linear spectral mixing across spatial regions.

    import graphdip
    cfg = graphdip.Config().resolve()
    graphdip.selfcheck()          # superpixels, mixing linearity, physics DIP
    hist = graphdip.train(cfg)    # per-scene DIP on the dataset
"""

from . import selfcheck as _selfcheck
from .config import Config
from .engine import evaluate_dataset, set_seed, train
from .model import GraphDIP, MixingLayer
from .superpixels import kmeans_superpixels

__all__ = ["Config", "GraphDIP", "MixingLayer", "kmeans_superpixels",
           "train", "evaluate_dataset", "set_seed"]


def selfcheck(device: str = "cpu") -> bool:
    return _selfcheck.run_all(device)