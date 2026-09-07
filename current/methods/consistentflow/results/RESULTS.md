# ConsistentFlow run report

## Status: mechanisms verified, ranking not yet — do not cite

The algebraic claims are verified on CPU:

| Claim | Check | Result |
|---|---|---|
| `D(ŷ) = X` after ONE consistency-map pass, untrained network | one-step sampler consistency | PASS (~1e-7) |
| The consistency map learns (CD loss and consistency gap shrink) | self-distillation on synthetic patches | PASS |
| One-step wrapper shape/gradient sanity | smoke test | PASS |

These prove the *identity at one step* — the structural claim the proposal
exists to make.  No full CAVE/Harvard training run has been completed, so there
are no PSNR/SSIM/SAM/ERGAS numbers to report.  Pending the Stage 1->4 ladder in
`docs/ARCHITECTURE.md`, nothing in this file should be compared against any
published method.
