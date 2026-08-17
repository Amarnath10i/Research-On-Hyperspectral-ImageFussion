# SpectralFlow run report

## Status: NO comparable results yet — do not cite

The algebraic claims are verified (see `daetf selfcheck` equivalent:
`python -c "import spectralflow; spectralflow.selfcheck()"`):

| Claim | Check | Result |
|---|---|---|
| `D(ŷ) = X` after sampling, untrained network | sampler consistency | 8.16e-07 |
| Score prior learns the manifold | denoise RMSE | 0.306 -> 0.208 (30 steps) |
| Range component = `D_pinv(X)` | range identity | PASS |
| Operator refinement reduces physics loss | refinement check | 0.0067 -> 0.0067 |
| Adjoint / consistency of the projector | nullspace checks | PASS |

These prove the *identity*, not the *ranking*.  No full CAVE/Harvard training
run has been completed for this proposal yet, so there are no PSNR/SSIM/SAM/ERGAS
numbers to report.  Pending the Stage 0->2 ladder in `docs/ARCHITECTURE.md`,
nothing in this file should be compared against any published method.
