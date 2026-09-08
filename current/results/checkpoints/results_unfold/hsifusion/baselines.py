"""Same-protocol reference methods.

The ten deep baselines in `existing/` were each run under their own protocol -
different scale factors, normalisations and metric implementations - so none of
their numbers can be compared directly against ours. Re-running all ten under
one protocol needs their checkpoints and three incompatible frameworks.

These classical methods need no checkpoints and no training, so they can be run
through the *identical* pipeline: same degradation, same scale factor, same
metric module, same scenes. That gives the paper a set of rows that are
genuinely comparable today, and a floor that any learned method must clear.

  bicubic       interpolation only, ignores the MSI entirely - the lower bound
                that reveals how much of a score comes from the HSI alone
  gsa           Gram-Schmidt Adaptive component substitution, the classical
                pansharpening approach with per-band regression gains
  subspace_ls   coupled subspace estimator: a spectral basis from the LR-HSI,
                abundances solved in closed form against the MSI with Tikhonov
                regularisation toward the upsampled HSI

`subspace_ls` is the strongest of the three and is the classical family that
model-based deep unfolding methods descend from, so it is the meaningful
non-learned comparison.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import BaseConfig as Config
from .data import SceneCache
from .degrade import FixedDegradation
from .io_utils import find_pairs
from .metrics import evaluate_arrays


def _upsample(lr: torch.Tensor, scale: int, mode: str = "bicubic") -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode=mode,
                         align_corners=False).clamp(0, 1)


def bicubic(lr_hsi: torch.Tensor, msi: torch.Tensor, srf: torch.Tensor,
            scale: int) -> torch.Tensor:
    """Interpolation only. Deliberately ignores the MSI."""
    return _upsample(lr_hsi, scale)


def gsa(lr_hsi: torch.Tensor, msi: torch.Tensor, srf: torch.Tensor,
        scale: int) -> torch.Tensor:
    """Gram-Schmidt Adaptive component substitution.

    Builds a synthetic low-resolution intensity from the upsampled HSI, then
    injects the detail the MSI carries, band by band, with gains from a
    least-squares regression of each band on that intensity.
    """
    up = _upsample(lr_hsi, scale)                       # [B,C,H,W]
    pan = msi.mean(dim=1, keepdim=True)                 # [B,1,H,W]

    b, c, h, w = up.shape
    x = up.reshape(b, c, -1)
    p = pan.reshape(b, 1, -1)

    # synthetic intensity: least-squares combination of HSI bands matching pan
    xt = x.transpose(1, 2)                              # [B,N,C]
    gram = xt.transpose(1, 2) @ xt                      # [B,C,C]
    rhs = xt.transpose(1, 2) @ p.transpose(1, 2)        # [B,C,1]
    eye = torch.eye(c, device=x.device, dtype=x.dtype)[None] * 1e-6
    coef = torch.linalg.solve(gram + eye, rhs)          # [B,C,1]
    inten = (coef.transpose(1, 2) @ x)                  # [B,1,N]

    det = p - inten
    iv = inten - inten.mean(dim=2, keepdim=True)
    var = (iv * iv).mean(dim=2, keepdim=True).clamp_min(1e-8)
    xv = x - x.mean(dim=2, keepdim=True)
    gain = (xv * iv).mean(dim=2, keepdim=True) / var    # [B,C,1]

    out = (x + gain * det).reshape(b, c, h, w)
    return out.clamp(0, 1)


def subspace_ls(lr_hsi: torch.Tensor, msi: torch.Tensor, srf: torch.Tensor,
                scale: int, rank: int = 8, lam: float = 0.15) -> torch.Tensor:
    """Coupled subspace estimator (closed form).

    Hyperspectral cubes are close to low rank: a handful of spectral basis
    vectors explain almost all the variance. Take that basis E from the LR-HSI
    by SVD, write the high-resolution image as X = E A, and solve for the
    abundances A using the MSI, regularised toward the abundances implied by
    the upsampled HSI:

        min_A  ||(Sᵀ E) A − Y_msi||²  +  λ ||A − A₀||²

    which has the closed-form solution

        A = (MᵀM + λI)⁻¹ (Mᵀ Y_msi + λ A₀),    M = Sᵀ E

    The MSI supplies spatial detail, the LR-HSI supplies spectral truth, and λ
    sets the balance. Without the regulariser the system is underdetermined
    whenever the subspace rank exceeds the MSI band count.
    """
    b, c, _, _ = lr_hsi.shape
    up = _upsample(lr_hsi, scale)                       # [B,C,H,W]
    _, _, h, w = up.shape
    out = torch.empty_like(up)

    for i in range(b):
        y = lr_hsi[i].reshape(c, -1).double()           # [C, n]
        # spectral subspace from the low-resolution cube (uncentred: the mean
        # spectrum is signal here, not a nuisance offset)
        u, _, _ = torch.linalg.svd(y @ y.t(), full_matrices=False)
        e = u[:, :rank]                                 # [C, r]

        s = srf.to(y.dtype).to(y.device)                # [C, m]
        m = s.t() @ e                                   # [m, r]
        ym = msi[i].reshape(msi.shape[1], -1).double()  # [m, N]
        a0 = e.t() @ up[i].reshape(c, -1).double()      # [r, N]

        lhs = m.t() @ m + lam * torch.eye(rank, dtype=y.dtype, device=y.device)
        rhs = m.t() @ ym + lam * a0
        a = torch.linalg.solve(lhs, rhs)                # [r, N]
        out[i] = (e @ a).reshape(c, h, w).to(out.dtype)

    return out.clamp(0, 1)


BASELINES: Dict[str, Callable] = {
    "Bicubic": bicubic,
    "GSA": gsa,
    "Subspace-LS": subspace_ls,
}


@torch.no_grad()
def evaluate_baseline(name: str, root: str, cfg: Config, srf: np.ndarray,
                      split: str = "Test", device: str = "cuda",
                      limit: Optional[int] = None, verbose: bool = True
                      ) -> Tuple[Dict[str, float], List[Dict]]:
    """Run one classical baseline through the identical evaluation pipeline."""
    fn = BASELINES[name]
    pairs = find_pairs(root, split)
    if limit:
        pairs = pairs[:limit]
    cache = SceneCache(cfg.bands, cfg.msi_bands, limit=2)
    degrade = FixedDegradation.from_config(cfg).to(device)
    srf_t = torch.from_numpy(srf).to(device)
    rows, agg = [], {"psnr": [], "ssim": [], "sam": [], "ergas": []}

    for stem, hp, rp in pairs:
        hsi, rgb = cache.get(stem, hp, rp)
        h = (hsi.shape[1] // cfg.scale) * cfg.scale
        w = (hsi.shape[2] // cfg.scale) * cfg.scale
        gt = torch.from_numpy(hsi[:, :h, :w].astype(np.float32))[None].to(device)
        msi = torch.from_numpy(rgb[:, :h, :w].astype(np.float32))[None].to(device)
        lr = degrade(gt)
        pred = fn(lr, msi, srf_t, cfg.scale).float()
        m = evaluate_arrays(pred[0].cpu().numpy().transpose(1, 2, 0),
                            gt[0].cpu().numpy().transpose(1, 2, 0), cfg.scale)
        rows.append({"scene": stem, **m})
        for k, v in m.items():
            agg[k].append(v)
        if verbose:
            print(f"  {stem:<24} PSNR={m['psnr']:7.3f}  SSIM={m['ssim']:.4f}  "
                  f"SAM={m['sam']:6.3f}  ERGAS={m['ergas']:8.3f}")
        del gt, msi, lr, pred
        if device == "cuda":
            torch.cuda.empty_cache()

    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    if verbose:
        print(f"  {name + ' MEAN':<24} PSNR={mean['psnr']:7.3f}  "
              f"SSIM={mean['ssim']:.4f}  SAM={mean['sam']:6.3f}  "
              f"ERGAS={mean['ergas']:8.3f}")
    return mean, rows


@torch.no_grad()
def evaluate_all_baselines(root: str, cfg: Config, srf: np.ndarray,
                           split: str = "Test", device: str = "cuda",
                           limit: Optional[int] = None, verbose: bool = True
                           ) -> Dict[str, Dict]:
    """Every classical baseline on one dataset, under the unified protocol."""
    out = {}
    for name in BASELINES:
        if verbose:
            print(f"\n--- {name} ---")
        mean, rows = evaluate_baseline(name, root, cfg, srf, split, device,
                                       limit=limit, verbose=verbose)
        out[name] = {"mean": mean, "rows": rows}
    return out
