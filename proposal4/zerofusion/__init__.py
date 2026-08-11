"""ZeroFusion - self-supervised per-scene fusion. Proposal 4.

No training set: each scene is fitted from its own observations using only
losses that need no ground truth. It therefore cannot suffer domain shift, and
acts as the control arm the other three proposals must beat cross-domain to
justify their training.

    import zerofusion as Z
    cfg = Z.Config().resolve()
    mean, rows = Z.evaluate_zeroshot(cfg.target_root, cfg, srf)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from hsifusion import BaseConfig, FusionLoss

from .model import AbundanceNet, ZeroFusion, abundance_priors, build_model

__version__ = "1.0.0"


@dataclass
class Config(BaseConfig):
    name: str = "zerofusion"
    endmembers: int = 12       # spectral basis size for the unmixing model
    width: int = 64
    depth: int = 4
    steps: int = 1500          # optimisation steps per scene; the
                               # objective is still falling at 1500 on
                               # measured data, so this is a floor
    scene_lr: float = 3e-3
    w_sparse: float = 0.005    # abundance L1/2 sparsity
    w_tv: float = 0.02         # abundance total variation
    w_range: float = 1.0       # penalty for leaving [0,1] (replaces a clamp)
    patience: int = 300        # early stop on the self-supervised objective
    max_side: int = 512        # cap the fitted region so Harvard stays affordable
    out_dir: str = "./zero_out"


def build_loss(cfg, srf):
    """Physics only - there is no ground truth to supervise against."""
    import torch

    def extra(out, cfg=cfg, **kw):
        return abundance_priors(out, cfg)

    return FusionLoss(cfg, torch.from_numpy(srf),
                      extra_terms=lambda **kw: extra(**kw))


def fit_scene(lr_hsi, msi, cfg, srf, device: str = "cuda",
              steps: Optional[int] = None, verbose: bool = False):
    """Fit one scene from scratch. This is the whole method."""
    import copy

    import torch

    steps = steps or cfg.steps
    model = build_model(cfg).to(device)
    model.init_endmembers(lr_hsi)
    crit = build_loss(cfg, srf).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.scene_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best, best_state, since = float("inf"), None, 0
    model.train()
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(lr_hsi, msi)
        loss, logs = crit(out, None, lr_hsi, msi, model, supervised=False)
        loss.backward()
        opt.step()
        sched.step()
        # early stopping on the SELF-SUPERVISED objective - no ground truth is
        # consulted, so this remains legitimate on an unlabelled scene
        if logs["total"] < best - 1e-5:
            best, since = logs["total"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                if verbose:
                    print(f"    early stop at {i} (best {best:.5f})")
                break
        if verbose and i % 100 == 0:
            print(f"    step {i:4d}  loss {logs['total']:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(lr_hsi, msi)["out"].clamp(0, 1)
    return pred, model, best


def evaluate_zeroshot(root: str, cfg, srf, split: str = "Test",
                      device: str = "cuda", limit: Optional[int] = None,
                      verbose: bool = True):
    """Fit and score every scene independently.

    Note the asymmetry with the other proposals: there is no train/test split
    to respect, because nothing is carried between scenes.
    """
    import time

    import numpy as np
    import torch

    from hsifusion.data import SceneCache
    from hsifusion.degrade import FixedDegradation
    from hsifusion.io_utils import find_pairs
    from hsifusion.metrics import evaluate_arrays

    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation(cfg.scale, cfg.blur_ksize, cfg.eval_sigma).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": []}
    times = []

    for stem, hp, rp in pairs:
        hsi, rgb = cache.get(stem, hp, rp)
        h = (hsi.shape[1] // cfg.scale) * cfg.scale
        w = (hsi.shape[2] // cfg.scale) * cfg.scale
        # bound the fit region so a 1040x1392 scene stays affordable
        h, w = min(h, cfg.max_side), min(w, cfg.max_side)
        h, w = (h // cfg.scale) * cfg.scale, (w // cfg.scale) * cfg.scale
        gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(device)
        msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(device)
        lr = degrade(gt)

        t0 = time.time()
        pred, _, obj = fit_scene(lr, msi, cfg, srf, device)
        dt = time.time() - t0
        times.append(dt)

        m = evaluate_arrays(pred[0].cpu().numpy().transpose(1, 2, 0),
                            gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
        rows.append({"scene": stem, **m, "seconds": dt})
        for k, v in m.items():
            agg[k].append(v)
        if verbose:
            print(f"  {stem:<22} PSNR={m['psnr']:7.3f}  SSIM={m['ssim']:.4f}  "
                  f"SAM={m['sam']:6.3f}  ERGAS={m['ergas']:8.3f}  ({dt:.1f}s)")
        del gt, msi, lr, pred
        if device == "cuda":
            torch.cuda.empty_cache()

    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    mean["seconds_per_scene"] = float(np.mean(times)) if times else 0.0
    if verbose:
        print(f"  {'MEAN':<22} PSNR={mean['psnr']:7.3f}  SSIM={mean['ssim']:.4f}  "
              f"SAM={mean['sam']:6.3f}  ERGAS={mean['ergas']:8.3f}  "
              f"({mean['seconds_per_scene']:.1f}s/scene)")
    return mean, rows


__all__ = ["Config", "ZeroFusion", "AbundanceNet", "build_model", "build_loss",
           "abundance_priors", "fit_scene", "evaluate_zeroshot", "__version__"]
