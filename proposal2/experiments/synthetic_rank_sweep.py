"""P2 experiment: intrinsic-rank sweep on controlled synthetic scenes.

Headline result: |r_hat - r_id| within +/-1 for r in {3,5,8,12,20,30}.
"""

from __future__ import annotations

from proposal2.metrics.identifiable_rank import estimate_ranks
from proposal2.rankest.generator import RankScene


def main() -> None:
    rows = []
    for r in [3, 5, 8, 12, 20, 30]:
        sc = RankScene(bands=64, msi_bands=48, rank=r, seed=0, decay=0.99,
                       srf_width=0.02, snr_db=float("inf"))
        r_hat, rid_hat, _ = estimate_ranks(sc.Y_H, sc.Y_M)
        rows.append([str(r), str(sc.r_id_true), str(r_hat), str(rid_hat),
                     str(abs(r_hat - r)), str(abs(rid_hat - sc.r_id_true))])
    print("| intrinsic rank r | r_id_true | r_hat | r_id_hat | |r_hat-r| | |r_id_hat-r_id| |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()