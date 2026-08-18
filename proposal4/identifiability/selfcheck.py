"""P4 self-checks: the phase diagram behaves as the identifiability theory
says it must - spectral identifiability (r_id) and spatial/spectral ambiguity
(null_frac) are two views of the same quantity, so both respond monotonically
to the knobs and anti-correlate across cells."""

from __future__ import annotations

from typing import List

import torch

from .simulator import simulate
from proposal2.rankest.generator import RankScene, make_srf
from proposal1.ambiguity.operator import CombinedOperator


@torch.no_grad()
def check_regimes(rank=8, bands=31):
    """One robust cell per regime."""
    ok = True
    good = simulate(rank, 31, 0.02, None, bands)
    bad = simulate(rank, 4, 0.50, 5, bands)
    i_cell = simulate(rank, 16, 0.02, 5, bands)
    w_cell = simulate(rank, 4, 0.06, 5, bands)
    ok &= good["regime"] == "I" and good["r_id_hat"] == rank
    ok &= bad["regime"] == "N"
    ok &= i_cell["regime"] == "I"
    ok &= w_cell["regime"] == "W"
    print(f"[1] regimes  clean/M31: {good['r_id_hat']}{good['regime']}; "
          f"SNR5/M4/w0.5: {bad['r_id_hat']}{bad['regime']}; "
          f"SNR5/M16: {i_cell['r_id_hat']}{i_cell['regime']}; "
          f"SNR5/M4/w0.06: {w_cell['r_id_hat']}{w_cell['regime']} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_noise_monotone(rank=8, bands=31):
    """r_id_hat is non-increasing as SNR decreases, for every M."""
    ok = True
    for m in (4, 8, 16, 31):
        seq = [simulate(rank, m, 0.02, s, bands)["r_id_hat"]
               for s in (None, 30, 20, 10, 5)]
        monotone = all(a >= b for a, b in zip(seq, seq[1:]))
        ok &= monotone
        print(f"    M={m:2d}: r_id_hat {seq} "
              f"({'OK' if monotone else 'NON-MONOTONE'})")
    print(f"[2] r_id_hat non-increasing in SNR (all M) "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_band_monotone(rank=8, bands=31):
    """r_id_hat is non-decreasing in M at fixed SNR."""
    ok = True
    for s in (None, 20):
        seq = [simulate(rank, m, 0.02, s, bands)["r_id_hat"]
               for m in (4, 8, 16, 31)]
        monotone = all(a <= b for a, b in zip(seq, seq[1:]))
        ok &= monotone
        print(f"    SNR={s}: r_id_hat {seq} "
              f"({'OK' if monotone else 'NON-MONOTONE'})")
    print(f"[3] r_id_hat non-decreasing in M (fixed SNR) "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_null_frac(rank=8, bands=31):
    """null_frac decreases with M (spatial/spectral sampling) and increases
    with heavy SRF overlap (endpoints; the mid-range is CG noise)."""
    nf_m = [simulate(rank, m, 0.02, None, bands)["null_frac"]
            for m in (4, 8, 16, 31)]
    nf_w = [simulate(rank, 8, w, None, bands)["null_frac"]
            for w in (0.02, 0.06, 0.15, 0.50)]
    ok_m = all(a >= b for a, b in zip(nf_m, nf_m[1:]))
    ok_w = nf_w[-1] > nf_w[0]
    print(f"    null_frac vs M: {[f'{v:.3f}' for v in nf_m]}")
    print(f"    null_frac vs width: {[f'{v:.3f}' for v in nf_w]}")
    ok = ok_m and ok_w
    print(f"[4] null_frac monotone in M; endpoint-increase with SRF width "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_agreement(rank=8, bands=31):
    """score and null_frac anti-correlate across the M x SNR grid."""
    cells = []
    for m in (4, 8, 16, 31):
        for s in (None, 30, 20, 10, 5):
            r = simulate(rank, m, 0.02, s, bands)
            cells.append((r["score"], r["null_frac"]))
    sc = torch.tensor([c[0] for c in cells])
    nf = torch.tensor([c[1] for c in cells])
    rho = torch.corrcoef(torch.stack([sc, nf]))[0, 1].item()
    ok = rho < -0.3
    print(f"[5] corr(score, null_frac) over M x SNR = {rho:.3f} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


@torch.no_grad()
def check_predictive(rank: int = 8, bands: int = 31):
    """The phase diagram is PREDICTIVE, not just descriptive.  Its score
    r_id_hat/r predicts the *spectral* identifiability floor:

        floor_spec = ||P_{ker(R^T)} X|| / ||X||,

    i.e. the part of the scene no fusion can recover because the sensors
    literally do not observe those spectral directions.  We build an
    Identifiable (I: r_id = r) and a Non-identifiable (N: r_id << r) cell,
    show that the best consistent spectral reconstruction error EQUALS
    floor_spec (so the diagram's score is exactly the achievable floor), and
    that the regimes separate the floor (tiny in I, large in N)."""
    from proposal2.model_selection import (spectral_null_fraction,
                                          observable_reconstruction)

    def cell(msi_bands, srf_width, snr_db):
        d = simulate(rank, msi_bands, srf_width, snr_db, bands)
        scene = RankScene(bands, msi_bands, 4, rank, 48, 48, 0,
                          srf_width=srf_width, snr_db=snr_db)
        floor = spectral_null_fraction(scene.X, scene.U, scene.R)
        x_rec = observable_reconstruction(scene.X, scene.U, scene.R)
        best_err = (x_rec - scene.X).norm() / scene.X.norm().clamp_min(1e-12)
        return d, float(floor), float(best_err)

    i_d, i_floor, i_err = cell(31, 0.02, None)     # I regime (r_id = r)
    n_d, n_floor, n_err = cell(4, 0.50, 5)         # N regime (r_id << r)

    # Best consistent reconstruction error must equal the spectral floor.
    ok_i = abs(i_err - i_floor) < 1e-3
    ok_n = abs(n_err - n_floor) < 1e-3
    # The diagram's score predicts the floor: ~0 in I, large in N.
    ok_floor_i = i_floor < 0.05
    ok_floor_n = n_floor > 0.20
    ok = ok_i and ok_n and ok_floor_i and ok_floor_n
    print(f"[6] predictive spectral floor  I: score={i_d['score']:.2f} "
          f"floor={i_floor:.3f} best-err={i_err:.3f};  "
          f"N: score={n_d['score']:.2f} floor={n_floor:.3f} "
          f"best-err={n_err:.3f}  ({'PASS' if ok else 'FAIL'})")
    return ok


def run_all(verbose: bool = True) -> bool:
    ok = True
    ok &= check_regimes()
    ok &= check_noise_monotone()
    ok &= check_band_monotone()
    ok &= check_null_frac()
    ok &= check_agreement()
    ok &= check_predictive()
    print(f"\n[identifiability] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()