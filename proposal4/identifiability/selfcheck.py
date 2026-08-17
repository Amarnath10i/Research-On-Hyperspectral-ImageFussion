"""P4 self-checks: the phase diagram behaves as the identifiability theory
says it must - spectral identifiability (r_id) and spatial/spectral ambiguity
(null_frac) are two views of the same quantity, so both respond monotonically
to the knobs and anti-correlate across cells."""

from __future__ import annotations

from typing import List

import torch

from .simulator import simulate


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


def run_all(verbose: bool = True) -> bool:
    ok = True
    ok &= check_regimes()
    ok &= check_noise_monotone()
    ok &= check_band_monotone()
    ok &= check_null_frac()
    ok &= check_agreement()
    print(f"\n[identifiability] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return bool(ok)


if __name__ == "__main__":
    run_all()