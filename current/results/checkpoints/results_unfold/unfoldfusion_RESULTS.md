# unfoldfusion run report

## Environment

- **python**: 3.12.13
- **platform**: Linux-6.12.90+-x86_64-with-glibc2.35
- **torch**: 2.5.1+cu124
- **numpy**: 2.0.2
- **cuda_available**: True
- **gpu**: Tesla P100-PCIE-16GB
- **cuda**: 12.4
- **gpu_mem_GB**: 15.9

## Results

| Method | PSNR (dB) up | SSIM up | SAM (deg) down | ERGAS down |
|---|---|---|---|---|
| Bicubic | 29.962 | 0.8893 | 4.883 | 8.344 |
| GSA | 34.717 | 0.9324 | 6.913 | 4.898 |
| Subspace-LS | 33.842 | 0.9488 | **4.659** | 5.216 |
| unfoldfusion (in-domain) | **42.730** | **0.9835** | 4.816 | **1.867** |
