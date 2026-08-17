"""Phase-diagram sweeps: (MSI bands x SNR) and (SRF width x SNR).

Run:  python -m proposal4.identifiability.phasediagram
"""

from __future__ import annotations

from typing import List, Optional

from .simulator import simulate

SNR_GRID = [None, 30, 20, 10, 5]          # None = clean (SNR=inf)
M_GRID = [4, 8, 16, 31]
W_GRID = [0.02, 0.06, 0.15, 0.50]


def _label(s) -> str:
    return "inf" if s is None else f"{s:2d}"


def _cell(r: dict) -> str:
    return f"{r['r_id_hat']:2d}{r['regime']}"


def sweep_msi_snr(rank: int = 8, srf_width: float = 0.02, **kw):
    header = "| MSI bands M | " + " | ".join(_label(s) for s in SNR_GRID) + " |"
    sep = "|---|" + "---|" * len(SNR_GRID)
    rows = []
    for m in M_GRID:
        cells = [_cell(simulate(rank, m, srf_width, s, **kw)) for s in SNR_GRID]
        rows.append(f"| {m:2d} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def sweep_srf_snr(rank: int = 8, msi_bands: int = 8, **kw):
    header = "| SRF width | " + " | ".join(_label(s) for s in SNR_GRID) + " | null_frac(clean) |"
    sep = "|---|" + "---|" * len(SNR_GRID) + "---|"
    rows = []
    for w in W_GRID:
        cells = [_cell(simulate(rank, msi_bands, w, s, **kw)) for s in SNR_GRID]
        nf = simulate(rank, msi_bands, w, None, **kw)["null_frac"]
        rows.append(f"| {w:4.2f} | " + " | ".join(cells) + f" | {nf:.3f} |")
    return "\n".join([header, sep, *rows])


def main():
    print("Legend: r_id_hat + regime (I/W/N).  SNR in dB, None = clean.\n")
    print("Table 1: MSI band count x SNR (srf_width = 0.02)\n")
    print(sweep_msi_snr())
    print("\nTable 2: SRF overlap x SNR (M = 8), with clean null_frac\n")
    print(sweep_srf_snr())


if __name__ == "__main__":
    main()