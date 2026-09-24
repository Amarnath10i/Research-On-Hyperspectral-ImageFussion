# SLT run report

## Status: mechanisms verified, ranking not yet — do not cite

| Claim | Check | Result |
|---|---|---|
| exp_map keeps unit norm, differentiable | manifold algebra | PASS (5.96e-08) |
| SAM == geodesic distance == \|Log\| | manifold algebra | PASS (1.19e-07) |
| Log(Exp_p(v)) = v round-trip | manifold algebra | PASS (2.38e-07) |
| **1.7x illumination: SAM moves 0.03 deg (fp floor); Euclidean control moves ~10 deg** | root-cause claim | PASS (~290x contrast) |
| network SAM == geodesic(dir_out, dir_gt) | metric alignment | PASS (0.0) |
| ladder switches forward/backward | smoke | PASS |

These prove the *structural* claims — the network cannot mask spectral error.
No full CAVE/Harvard training run has produced PSNR/SSIM/SAM/ERGAS numbers yet,
so nothing here is comparable against published methods until the Stage 1-4
ladder runs on Kaggle.