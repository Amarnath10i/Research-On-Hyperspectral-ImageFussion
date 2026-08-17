"""P2 experiment: SRF-overlap sweep (identifiable rank vs spectral response).

Wide/overlapping SRF columns collapse spectral directions and reduce the rank
that the MSI can resolve; the estimator tracks this reduction.
"""

from __future__ import annotations

from proposal2.metrics.identifiable_rank import estimate_ranks
from proposal2.rankest.generator import RankScene


def main() -> None:
    rows = []
    for w in [0.02, 0.06, 0.15, 0.5]:
        sc = RankScene(bands=31, msi_bands=16, rank=12, seed=3, decay=0.99,
                       srf_width=w, snr_db=float("inf"))
        r_hat, rid_hat, _ = estimate_ranks(sc.Y_H, sc.Y_M)
        rows.append([f"{w:.2f}", str(sc.r_id_true), str(rid_hat),
                     str(abs(rid_hat - sc.r_id_true))])
    print("| SRF width | r_id_true | r_id_hat | |r_id_hat-r_id| |")
    print("|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()