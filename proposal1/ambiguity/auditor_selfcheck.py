"""P1 ambiguity-auditor self-check.

The differentiated P1 contribution is NOT another fusion network but a
*method-agnostic auditor*: given any fusion output Y_hat, the hallucination
metric H = ||P_N(Y_hat - X)|| / ||P_N X|| and the uncertainty map U say what the
method *invented* versus what the observations pinned down.  This check proves
the auditor works:

  [1] H = 0 for the oracle (recovers the true null content), H = 1 for a
      range-only method that honestly drops it, and H in (0,1) for a partial
      recovery -- H is a faithful hallucination detector;
  [2] H correlates (approx 1) with spectral error (SAM) across methods, so H is
      a usable proxy for "how much did this method make up?";
  [3] the uncertainty map U localises the null content: it correlates with the
      per-pixel magnitude of the true unobservable component.
"""

from __future__ import annotations

import torch

from .operator import CombinedOperator
from .metrics import hallucination, uncertainty_map
from proposal2.rankest.generator import RankScene, make_srf


def _sam(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(a.shape[1], -1)                   # (bands, N)
    b = b.reshape(b.shape[1], -1)
    na = a.norm(2, 0, keepdim=True).clamp_min(1e-9)
    nb = b.norm(2, 0, keepdim=True).clamp_min(1e-9)
    cos = ((a / na) * (b / nb)).sum(0).clamp(-1, 1)
    return float(torch.acos(cos).mean().rad2deg().item())


@torch.no_grad()
def run_all(verbose: bool = True) -> bool:
    bands, msi, rank = 16, 8, 6
    scene = RankScene(bands, msi, 4, rank, 32, 32, 0, srf_width=0.05)
    P = CombinedOperator(4, bands, msi, srf=make_srf(bands, msi, 0.05),
                         cg_steps=120, ridge=1e-6)
    X = scene.X[None]                               # (1, B, H, W)

    x_obs = P.project_range(X)
    true_null = P.project_null(X)
    oracle = x_obs + true_null                       # = X
    partial = x_obs + 0.5 * true_null
    # wrong null: random, same norm as the truth
    torch.manual_seed(7)
    wrong = x_obs + true_null.norm() * torch.randn_like(true_null) / \
        torch.randn_like(true_null).norm().clamp_min(1e-9)

    Hs = {
        "oracle": hallucination(X, oracle, P),
        "partial": hallucination(X, partial, P),
        "range_only": hallucination(X, x_obs, P),
        "wrong": hallucination(X, wrong, P),
    }
    sams = {
        "oracle": _sam(oracle, X),
        "partial": _sam(partial, X),
        "range_only": _sam(x_obs, X),
        "wrong": _sam(wrong, X),
    }

    ok1 = (Hs["oracle"] < 1e-3 and abs(Hs["range_only"] - 1.0) < 1e-2 and
           Hs["partial"] > 0.3 and Hs["partial"] < 0.7)
    # H tracks SAM across the four methods.
    hv = torch.tensor([Hs[k] for k in ("oracle", "partial", "range_only", "wrong")])
    sv = torch.tensor([sams[k] for k in ("oracle", "partial", "range_only", "wrong")])
    rho = torch.corrcoef(torch.stack([hv, sv]))[0, 1].item()
    ok2 = rho > 0.9

    # U localises the null content.
    U = uncertainty_map(X, P)[0]                      # (H, W) per-pixel null
    true_px = true_null.pow(2).sum(1).sqrt()[0]        # (H, W) per-pixel null
    rho_u = torch.corrcoef(torch.stack(
        [U.reshape(-1), true_px.reshape(-1)]))[0, 1].item()
    ok3 = rho_u > 0.5

    ok = ok1 and ok2 and ok3
    print(f"[1] H(oracle)={Hs['oracle']:.3f}  H(partial)={Hs['partial']:.3f}  "
          f"H(range_only)={Hs['range_only']:.3f}  H(wrong)={Hs['wrong']:.3f}  "
          f"({'PASS' if ok1 else 'FAIL'})")
    print(f"[2] corr(H, SAM) across methods = {rho:.3f}  "
          f"({'PASS' if ok2 else 'FAIL'})")
    print(f"[3] corr(U_map, true null per pixel) = {rho_u:.3f}  "
          f"({'PASS' if ok3 else 'FAIL'})")
    print(f"\n[auditor] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()
