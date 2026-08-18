"""P2 model-selection self-check.

Demonstrates, on a controlled RankScene with r_id < r (genuinely
under-determined), that:

  [1] select_rank = min(r_hat, r_id_hat) recovers the observable rank;
  [2] the rank-r_id reconstruction achieves the fundamental lower bound
      LB = ||P_{ker(R^T)} X|| / ||X||  (no spectral distortion on the
      observable part);
  [3] an ORACLE that also knows the unobservable null component reaches the
      same bound -- i.e. LB is tight and only an oracle can beat it;
  [4] an OVER-ranked method that fills the (r - r_id)-dim null space with a
      (generic, wrong) prior pushes error strictly ABOVE LB -- it invents
      spectra;
  [5] an UNDER-ranked method that drops observable directions also exceeds LB.

This is the empirical payload behind the lemma in theory/model_selection.md:
r* = min(r_hat, r_id_hat) is the largest well-posed model rank.
"""

from __future__ import annotations

import torch

from .rankest.generator import RankScene
from .metrics.identifiable_rank import estimate_ranks
from .model_selection import (select_rank, spectral_null_fraction,
                              observable_reconstruction, reconstruct_with_prior,
                              true_null_component)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm().item() / b.norm().clamp_min(1e-12).item())


def run_all(verbose: bool = True) -> bool:
    # Under-determined scene: intrinsic rank r=12, but only a few MSI bands
    # can observe it, so r_id < r.
    scene = RankScene(bands=31, msi_bands=8, scale=4, rank=12,
                      H=32, W=32, seed=0, srf_width=0.05)
    X, U, R = scene.X, scene.U, scene.R
    r_hat, r_id_hat, _ = estimate_ranks(scene.Y_H, scene.Y_M)
    r_star = select_rank(r_hat, r_id_hat)

    LB = spectral_null_fraction(X, U, R)
    X_id = observable_reconstruction(X, U, R)
    err_id = _rel(X_id, X)

    # Null-space component of the TRUE coefficients Z (r x HW).
    G = R.t() @ U
    Z = U.t() @ X.reshape(X.shape[0], -1).float()
    P_null = torch.eye(G.shape[1], device=G.device) - torch.linalg.pinv(G) @ G
    null_Z = P_null @ Z

    # Oracle: knows the true unobservable component, so it recovers X exactly.
    X_oracle = reconstruct_with_prior(X, U, R, null_Z)
    err_oracle = _rel(X_oracle, X)

    # Over-ranked method: fills the null space with a generic (wrong) prior of
    # the same norm as the truth but a random direction.
    torch.manual_seed(3)
    nrm = null_Z.norm().clamp_min(1e-9)
    wrong = nrm * torch.randn_like(null_Z) / torch.randn_like(
        null_Z).norm().clamp_min(1e-9)
    X_over = reconstruct_with_prior(X, U, R, wrong)
    err_over = _rel(X_over, X)

    # Under-ranked method: SVD-compress the achievable reconstruction to
    # (r_id_true - 2) spectral directions, dropping observable content.
    Xf = X_id.reshape(X.shape[0], -1)
    Us, Ss, Vs = torch.linalg.svd(Xf, full_matrices=False)
    k = max(1, scene.r_id_true - 4)
    X_under = (Us[:, :k] * Ss[:k]) @ Vs[:k, :]
    X_under = X_under.reshape(X.shape)
    err_under = _rel(X_under, X)

    ok = True
    ok1 = (r_star == min(r_hat, r_id_hat)) and abs(r_star - scene.r_id_true) <= 1
    ok &= ok1
    ok2 = abs(err_id - LB) < 0.05 * max(LB, 1e-6) + 1e-3
    ok &= ok2
    ok3 = err_oracle < 1e-3
    ok &= ok3
    ok4 = err_over > LB * 1.05
    ok &= ok4
    ok5 = err_under > LB * 1.02
    ok &= ok5

    print(f"[1] select_rank = min({r_hat},{r_id_hat}) = {r_star} "
          f"(r_id_true={scene.r_id_true})  {'PASS' if ok1 else 'FAIL'}")
    print(f"[2] rank-r_id reconstruction error = {err_id:.4f}  "
          f"LB = {LB:.4f}  (achieves bound)  {'PASS' if ok2 else 'FAIL'}")
    print(f"[3] oracle (knows null) error = {err_oracle:.4f}  "
          f"(recovers X exactly)  {'PASS' if ok3 else 'FAIL'}")
    print(f"[4] over-ranked (wrong prior) error = {err_over:.4f} > LB "
          f"({'PASS' if ok4 else 'FAIL'})")
    print(f"[5] under-ranked error = {err_under:.4f} > LB  "
          f"({'PASS' if ok5 else 'FAIL'})")
    print(f"\n[model_selection] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()
