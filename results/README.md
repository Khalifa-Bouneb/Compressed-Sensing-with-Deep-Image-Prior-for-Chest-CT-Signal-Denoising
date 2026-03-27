# Results Directory

This folder stores output from experiments:

- `baseline_benchmark.csv` - PSNR/SSIM results for classical methods
- `baseline_comparison.png` - Bar chart comparing methods
- `comparison_montage.png` - Visual comparison grid

## Benchmark Results Template

| Image ID | Noisy PSNR | Wiener PSNR | BM3D PSNR |
|----------|------------|-------------|-----------|
| BSDS_01  | 18.5 dB    | 22.1 dB     | 28.4 dB   |
| BSDS_02  | 19.2 dB    | 23.0 dB     | 27.9 dB   |
| ...      | ...        | ...         | ...       |

Run `python run_baselines.py` to generate these results.
