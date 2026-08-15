# Eager FFT vs custom CUDA ADMM-DIP-TV

Two standalone scripts compare the previous eager FFT implementation in
`src/denoise_dip_tv_eager.py` with the custom CUDA/fallback implementation in
`src/denoise_dip_tv.py`. Both methods receive the same real BSDS300 image,
speckle-noise realization, network initialization seed, and DIP input seed.

## Unprofiled result and timing comparison

Run the short default comparison:

```bash
MPLCONFIGDIR=/tmp/matplotlib-compare \
python compare_admm_dip_tv.py \
  --image-size 64 \
  --iterations 1 \
  --output results/admm_tv_comparison
```

Run the complete configured experiment on the full processed image:

```bash
MPLCONFIGDIR=/tmp/matplotlib-compare \
python compare_admm_dip_tv.py \
  --image-size 0 \
  --iterations 150 \
  --output results/admm_tv_comparison_full
```

The full experiment is computationally expensive: each outer iteration uses
10 `LBFGS.step()` calls and LBFGS may evaluate its closure up to 20 times per
step.

The output directory contains:

- `output_comparison.png`: clean, noisy, eager, and custom images;
- `output_difference.png`: absolute difference heatmap;
- `psnr_convergence.png`: PSNR by ADMM iteration;
- `iteration_metrics.csv`: all recorded metrics;
- `comparison_arrays.npz`: raw clean/noisy/output arrays;
- `comparison_summary.json` and `.txt`: quality, timing, speedup, memory, and backend.

By default, CUDA operator startup and extension compilation are warmed before
the timed calls. Use `--no-warmup-operators` to include cold-start cost.

## Profiler comparison

Run a manageable trace:

```bash
MPLCONFIGDIR=/tmp/matplotlib-compare \
python profile_compare_admm_dip_tv.py \
  --image-size 64 \
  --iterations 1 \
  --inner-iterations 2 \
  --lbfgs-max-iter 2 \
  --output profiling/admm_tv_comparison
```

Add `--with-stack` when Python stack attribution is needed. It increases trace
size and profiling overhead.

Each method gets its own directory containing:

```text
eager_fft/trace.json
eager_fft/cpu_table.txt
eager_fft/cuda_table.txt
custom_cuda/trace.json
custom_cuda/cpu_table.txt
custom_cuda/cuda_table.txt
```

The root also contains `profile_summary.json`, `profile_summary.txt`, and an
`outputs/` directory with the visual and numerical comparison. Load either
`trace.json` directly in <https://ui.perfetto.dev>.

Profiler-instrumented wall times include profiler overhead. Use the unprofiled
script for the primary end-to-end speed comparison and the profiler script to
explain where the time is spent.

## CUDA status

Both scripts print `torch.cuda.is_available()`, the GPU name, PyTorch CUDA
version, and whether the native ADMM extension is loaded. If CUDA is not
available, the custom implementation correctly runs its PyTorch fallback and
the report does not claim CUDA-kernel performance.
