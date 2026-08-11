# DAETF-Net run report

## Environment

- **python**: 3.12.13
- **platform**: Linux-6.12.90+-x86_64-with-glibc2.35
- **torch**: 2.10.0+cu128
- **numpy**: 2.0.2
- **cuda_available**: True
- **gpu**: Tesla T4
- **cuda**: 12.8
- **gpu_mem_GB**: 14.6

## Results

| Method | PSNR (dB) up | SSIM up | SAM (deg) down | ERGAS down |
|---|---|---|---|---|
| DAETF-Net (in-domain) | 28.032 | **0.7082** | **14.552** | **10.289** |
| DAETF-Net (zero-shot) | **33.859** | 0.6940 | 54.289 | 682.912 |
| DAETF-Net (+TTA) | 30.162 | 0.3881 | 48.337 | 341.580 |

## Cost

- **params_M**: 2.06047
- **gflops**: 398.6740032
- **latency_s**: 0.36421220302581786
- **peak_mem_MB**: 280.591796875
- **hr**: 512

## Table 3: Harvard in-domain

| Method | PSNR (dB) up | SSIM up | SAM (deg) down | ERGAS down |
|---|---|---|---|---|
| Bicubic | 59.497 | 0.9973 | **2.906** | 4.795 |
| GSA | **65.903** | **0.9992** | 3.128 | **2.648** |
| Subspace-LS | 22.853 | 0.3661 | 15.162 | 470.852 |
| DAETF-Net (Harvard-trained) | 24.976 | 0.3611 | 40.899 | 329.039 |
