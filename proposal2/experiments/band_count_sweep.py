"""P2 experiment: MSI band-count sweep (identifiable rank vs spectral sampling).

The identifiable rank is bounded by the number of MSI bands; more bands with a
diverse SRF resolve more spectral degrees of freedom.
"""

from __future__ import annotations

from proposal2.metrics.identifiable_rank import estimate_ranks
from proposal2.rankest.generator import RankScene


def main() -> None:
    rows = []
    for M in [4, 8, 16, 31]:
        sc = RankScene(bands=31, msi_bands=M, rank=12, seed=2, decay=0.99,
                       snr_db=float("inf"))
        r_hat, rid_hat, _ = estimate_ranks(sc.Y_H, sc.Y_M)
        rows.append([str(M), str(sc.r_id_true), str(rid_hat),
                     str(abs(rid_hat - sc.r_id_true))])
    print("| MSI bands M | r_id_true | r_id_hat | |r_id_hat-r_id| |")
    print("|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()