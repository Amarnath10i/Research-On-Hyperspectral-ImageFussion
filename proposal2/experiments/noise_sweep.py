"""P2 experiment: SNR sweep (identifiable rank vs noise).

As SNR decreases, fewer spectral degrees of freedom survive the optimal
hard threshold -> r_id_hat decreases.  sigma_hat should track sigma.
"""

from __future__ import annotations

from proposal2.metrics.identifiable_rank import estimate_ranks
from proposal2.rankest.generator import RankScene


def main() -> None:
    rows = []
    for snr in [float("inf"), 30, 20, 10, 5]:
        sc = RankScene(bands=31, msi_bands=16, rank=12, seed=1, decay=0.7,
                       snr_db=snr)
        r_hat, rid_hat, shat = estimate_ranks(sc.Y_H, sc.Y_M)
        sig = f"{sc.sigma:.2e}" if sc.sigma > 0 else "-"
        rows.append([f"{snr if snr == float('inf') else snr:>4}", sig,
                     f"{shat:.2e}", str(r_hat), str(rid_hat)])
    print("| SNR (dB) | sigma | sigma_hat | r_hat | r_id_hat |")
    print("|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()