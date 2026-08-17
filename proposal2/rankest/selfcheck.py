"""Self-checks for P2 (identifiable spectral rank), CPU-fast, no data required.

Validates the structural claims:
  [1] Rank sweep (clean): r_hat and r_id_hat recover the true ranks within +/-1
      for intrinsic ranks r in {3,5,8,12,20,30} (headline metric |r_hat - r_id|).
  [2] Cap: r_id_hat <= msi band count always.
  [3] Noise sweep: both r_hat and r_id_hat are non-increasing as SNR drops.
  [4] Band-count sweep: r_id_hat tracks the true identifiable rank.
  [5] SRF-overlap sweep: r_id_hat tracks the reduced identifiable rank.
  [6] Observation-only noise estimate: sigma_hat ~ sigma (ratio in [0.8, 1.2]).
"""

from __future__ import annotations

import math

import torch

from proposal2.metrics.identifiable_rank import estimate_ranks
from proposal2.rankest.generator import RankScene


def _scene(**kw) -> RankScene:
    return RankScene(**kw)


def check_rank_sweep() -> bool:
    ok = True
    rows = []
    for r in [3, 5, 8, 12, 20, 30]:
        sc = _scene(bands=64, msi_bands=48, rank=r, seed=0, decay=0.99,
                    srf_width=0.02, snr_db=float("inf"))
        r_hat, rid_hat, _ = estimate_ranks(sc.Y_H, sc.Y_M)
        err = abs(rid_hat - sc.r_id_true)
        rows.append((r, sc.r_id_true, r_hat, rid_hat, err))
        ok &= (abs(r_hat - r) <= 1 and err <= 1)
    for r, rid, rh, ridh, err in rows:
        print(f"[1] r={r:2d} r_id_true={rid:2d} r_hat={rh:2d} "
              f"r_id_hat={ridh:2d} |d|={err} "
              f"{'PASS' if (abs(rh - r) <= 1 and err <= 1) else 'FAIL'}")
    return ok


def check_rank_cap() -> bool:
    ok = True
    for M in [4, 8, 16, 31]:
        sc = _scene(bands=31, msi_bands=M, rank=12, seed=2, decay=0.99,
                    snr_db=float("inf"))
        r_hat, rid_hat, _ = estimate_ranks(sc.Y_H, sc.Y_M)
        cap = rid_hat <= M
        ok &= cap
        print(f"[2] M={M:2d} r_id_hat={rid_hat:2d} <= M "
              f"{'PASS' if cap else 'FAIL'}")
    return ok


def check_noise_trend() -> bool:
    prev_r = prev_rid = None
    ok = True
    for snr in [float("inf"), 30, 20, 10, 5]:
        sc = _scene(bands=31, msi_bands=16, rank=12, seed=1, decay=0.7,
                    snr_db=snr)
        r_hat, rid_hat, _ = estimate_ranks(sc.Y_H, sc.Y_M)
        mono = (prev_r is None) or (r_hat <= prev_r and rid_hat <= prev_rid)
        ok &= mono
        print(f"[3] snr={snr:>4} r_hat={r_hat:2d} r_id_hat={rid_hat:2d} "
              f"non-increasing={mono} {'PASS' if mono else 'FAIL'}")
        prev_r, prev_rid = r_hat, rid_hat
    return ok


def check_band_sweep() -> bool:
    ok = True
    for M in [4, 8, 16, 31]:
        sc = _scene(bands=31, msi_bands=M, rank=12, seed=2, decay=0.99,
                    snr_db=float("inf"))
        r_hat, rid_hat, _ = estimate_ranks(sc.Y_H, sc.Y_M)
        err = abs(rid_hat - sc.r_id_true)
        ok &= err <= 1
        print(f"[4] M={M:2d} r_id_true={sc.r_id_true:2d} r_id_hat={rid_hat:2d} "
              f"|d|={err} {'PASS' if err <= 1 else 'FAIL'}")
    return ok


def check_srf_sweep() -> bool:
    ok = True
    prev = None
    for w in [0.02, 0.06, 0.15, 0.5]:
        sc = _scene(bands=31, msi_bands=16, rank=12, seed=3, decay=0.99,
                    srf_width=w, snr_db=float("inf"))
        r_hat, rid_hat, _ = estimate_ranks(sc.Y_H, sc.Y_M)
        err = abs(rid_hat - sc.r_id_true)
        mono = prev is None or sc.r_id_true <= prev
        ok &= (err <= 1 and mono)
        print(f"[5] width={w:.2f} r_id_true={sc.r_id_true:2d} "
              f"r_id_hat={rid_hat:2d} |d|={err} "
              f"{'PASS' if (err <= 1 and mono) else 'FAIL'}")
        prev = sc.r_id_true
    return ok


def check_noise_estimate() -> bool:
    ok = True
    for snr in [30, 20, 10]:
        sc = _scene(bands=31, msi_bands=16, rank=12, seed=4, decay=0.7,
                    snr_db=snr)
        _, _, shat = estimate_ranks(sc.Y_H, sc.Y_M)
        ratio = shat / sc.sigma if sc.sigma > 0 else 1.0
        good = 0.8 <= ratio <= 1.2
        ok &= good
        print(f"[6] snr={snr} sigma={sc.sigma:.2e} sigma_hat={shat:.2e} "
              f"ratio={ratio:.2f} {'PASS' if good else 'FAIL'}")
    return ok


def run_all(device: str = "cpu") -> bool:
    torch.manual_seed(0)
    checks = [
        ("rank sweep", check_rank_sweep),
        ("rank cap", check_rank_cap),
        ("noise trend", check_noise_trend),
        ("band sweep", check_band_sweep),
        ("srf sweep", check_srf_sweep),
        ("noise estimate", check_noise_estimate),
    ]
    ok = True
    for name, fn in checks:
        try:
            ok &= bool(fn())
        except Exception as e:  # noqa: BLE001
            print(f"[selfcheck] {name}: EXCEPTION {e!r}")
            ok = False
    print(f"[selfcheck] {'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)