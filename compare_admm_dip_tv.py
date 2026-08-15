"""Compare eager FFT ADMM-DIP-TV with the custom CUDA/fallback implementation.

The script uses one real BSDS300 image and exactly the same noisy observation,
network seed, and DIP-input seed for both methods. It does not enable a profiler.
"""

import argparse
import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src import denoise_dip_tv as custom_tv
from src import denoise_dip_tv_eager as eager_tv
from src.admm_cuda import (
    cuda_extension_available,
    cuda_extension_status,
    periodic_gradient,
    shrink_and_update_dual,
)
from src.utils import D, add_speckle, process_image_tensor, psf2otf


METHODS = {
    "eager_fft": eager_tv,
    "custom_cuda": custom_tv,
}


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_real_input(dataset_root, image_index, image_size):
    image_root = Path(dataset_root) / "images"
    files = sorted(path for path in image_root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No images found below {image_root.resolve()}")
    if not 0 <= image_index < len(files):
        raise IndexError(f"image-index must be in [0, {len(files) - 1}]")

    image = skimage.io.imread(files[image_index])
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image[..., :3]
    if image.shape[0] > image.shape[1]:
        image = image.transpose(1, 0, 2)
    tensor = torch.from_numpy(
        image.transpose(2, 0, 1).astype(np.float32) / 255.0
    )
    if image_size:
        height, width = tensor.shape[-2:]
        if image_size > min(height, width):
            raise ValueError(
                f"image-size {image_size} exceeds input shape {height}x{width}"
            )
        top = (height - image_size) // 2
        left = (width - image_size) // 2
        tensor = tensor[:, top:top + image_size, left:left + image_size]

    image_pil, clean_tensor = process_image_tensor(tensor, d=32)
    return files[image_index], image_pil, clean_tensor.cpu()


@contextmanager
def method_settings(module, iterations, plotting=False):
    previous_iterations = module.num_iter
    previous_plot = module.PLOT
    module.num_iter = iterations
    module.PLOT = plotting
    try:
        yield
    finally:
        module.num_iter = previous_iterations
        module.PLOT = previous_plot


def evaluate_dump(output_dump):
    network, network_input = output_dump
    with torch.no_grad():
        output = network(network_input)
    return np.clip(output.detach().cpu().numpy()[0], 0.0, 1.0)


def run_variant(name, module, image_pil, clean, noisy, iterations, seed):
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    method = (
        module.admm_dip_single_eager
        if name == "eager_fft"
        else module.admm_dip_single
    )
    with method_settings(module, iterations):
        metrics, output_dump = method(
            image_pil, clean, noisy, ind=1, verbose=False
        )
    synchronize()
    elapsed = time.perf_counter() - started
    output = evaluate_dump(output_dump)
    memory = {
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
    }
    print(f"\n{name}: {elapsed:.6f} s, final PSNR={metrics[-1]['PSNR_gt']:.4f} dB")
    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "output": output,
        "memory": memory,
    }


def warm_up_operators(shape):
    """Exclude one-time CUDA-extension/cuFFT startup from method timings."""
    if not torch.cuda.is_available():
        return 0.0
    started = time.perf_counter()
    dummy = torch.zeros((1, 3, *shape), device="cuda")
    dh_psf = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    dv_psf = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    dh_dft = torch.from_numpy(psf2otf(dh_psf, list(shape))).to("cuda")
    dv_dft = torch.from_numpy(psf2otf(dv_psf, list(shape))).to("cuda")
    D(dummy, dh_dft, dv_dft)
    grad_h, grad_v = periodic_gradient(dummy)
    zeros = torch.zeros_like(dummy)
    shrink_and_update_dual(grad_h, grad_v, zeros, zeros, 0.01 / 12.5)
    synchronize()
    return time.perf_counter() - started


def quality_summary(clean, noisy, eager_output, custom_output):
    clean_hwc = clean.transpose(1, 2, 0)
    noisy_hwc = noisy.transpose(1, 2, 0)
    eager_hwc = eager_output.transpose(1, 2, 0)
    custom_hwc = custom_output.transpose(1, 2, 0)
    difference = eager_output - custom_output
    return {
        "noisy_psnr_db": float(peak_signal_noise_ratio(clean, noisy, data_range=1.0)),
        "noisy_ssim": float(structural_similarity(
            clean_hwc, noisy_hwc, data_range=1.0, channel_axis=2
        )),
        "eager_psnr_db": float(peak_signal_noise_ratio(clean, eager_output, data_range=1.0)),
        "eager_ssim": float(structural_similarity(
            clean_hwc, eager_hwc, data_range=1.0, channel_axis=2
        )),
        "custom_psnr_db": float(peak_signal_noise_ratio(clean, custom_output, data_range=1.0)),
        "custom_ssim": float(structural_similarity(
            clean_hwc, custom_hwc, data_range=1.0, channel_axis=2
        )),
        "output_mse": float(np.mean(difference ** 2)),
        "output_mae": float(np.mean(np.abs(difference))),
        "output_max_abs": float(np.max(np.abs(difference))),
        "output_psnr_db": float(peak_signal_noise_ratio(
            eager_output, custom_output, data_range=1.0
        )),
    }


def save_results(output_dir, source_path, clean, noisy, runs, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    eager = runs["eager_fft"]
    custom = runs["custom_cuda"]
    quality = quality_summary(clean, noisy, eager["output"], custom["output"])
    speedup = eager["elapsed_seconds"] / custom["elapsed_seconds"]

    summary = {
        **metadata,
        "source_image": str(source_path.resolve()),
        "input_shape_chw": list(clean.shape),
        "device": str(custom_tv.device),
        "cuda_available": torch.cuda.is_available(),
        "custom_backend": (
            "native_cuda" if cuda_extension_available() else cuda_extension_status()
        ),
        "timing": {
            "eager_fft_seconds": eager["elapsed_seconds"],
            "custom_cuda_seconds": custom["elapsed_seconds"],
            "eager_over_custom_speedup": speedup,
        },
        "quality": quality,
        "memory": {
            name: run["memory"] for name, run in runs.items()
        },
        "final_metrics": {
            name: run["metrics"][-1] for name, run in runs.items()
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )

    with (output_dir / "iteration_metrics.csv").open("w", newline="") as handle:
        fields = ["method", "iteration", "PSNR_noisy", "PSNR_gt", "SSIM_gt", "SSIM_noisy", "DSSIM", "loss"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, run in runs.items():
            for row in run["metrics"]:
                writer.writerow({"method": name, **row})

    np.savez_compressed(
        output_dir / "comparison_arrays.npz",
        clean=clean, noisy=noisy,
        eager=eager["output"], custom=custom["output"],
        difference=eager["output"] - custom["output"],
    )

    images = [clean, noisy, eager["output"], custom["output"]]
    titles = [
        "Clean input",
        f"Noisy\nPSNR {quality['noisy_psnr_db']:.2f} dB",
        f"Eager FFT\nPSNR {quality['eager_psnr_db']:.2f} dB",
        f"Custom CUDA/fallback\nPSNR {quality['custom_psnr_db']:.2f} dB",
    ]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(np.clip(image.transpose(1, 2, 0), 0, 1))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "output_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    difference = np.mean(np.abs(eager["output"] - custom["output"]), axis=0)
    figure, axis = plt.subplots(figsize=(6, 4))
    view = axis.imshow(difference, cmap="magma")
    axis.set_title(f"Mean absolute output difference\nMAE={quality['output_mae']:.3e}")
    axis.axis("off")
    figure.colorbar(view, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "output_difference.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for name, run in runs.items():
        axis.plot(
            [row["iteration"] for row in run["metrics"]],
            [row["PSNR_gt"] for row in run["metrics"]],
            label=name,
        )
    axis.set_xlabel("ADMM outer iteration")
    axis.set_ylabel("PSNR to ground truth (dB)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "psnr_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    text = [
        "ADMM-DIP-TV EAGER VS CUSTOM COMPARISON",
        f"Device: {summary['device']}",
        f"Custom backend: {summary['custom_backend']}",
        f"Input: {source_path} {tuple(clean.shape)}",
        f"Eager FFT: {eager['elapsed_seconds']:.6f} s",
        f"Custom: {custom['elapsed_seconds']:.6f} s",
        f"Speedup (eager/custom): {speedup:.4f}x",
        f"Eager PSNR/SSIM: {quality['eager_psnr_db']:.4f} dB / {quality['eager_ssim']:.6f}",
        f"Custom PSNR/SSIM: {quality['custom_psnr_db']:.4f} dB / {quality['custom_ssim']:.6f}",
        f"Output MSE/MAE/max: {quality['output_mse']:.6e} / {quality['output_mae']:.6e} / {quality['output_max_abs']:.6e}",
    ]
    (output_dir / "comparison_summary.txt").write_text("\n".join(text) + "\n")
    print("\n" + "\n".join(text))
    print(f"Results saved to: {output_dir.resolve()}")
    return summary


def prepare_experiment(args):
    seed_everything(args.seed)
    source_path, image_pil, clean_tensor = load_real_input(
        args.dataset_root, args.image_index, args.image_size
    )
    clean = clean_tensor.numpy()
    noisy = add_speckle(clean_tensor, args.sigma).numpy()
    return source_path, image_pil, clean, noisy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="Dataset/BSDS300/BSDS300")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64, help="Center-crop size; use 0 for the full processed image")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1, help="ADMM outer iterations (configured full run is 150)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="results/admm_tv_comparison")
    parser.add_argument("--warmup-operators", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source_path, image_pil, clean, noisy = prepare_experiment(args)
    warmup_seconds = warm_up_operators(clean.shape[-2:]) if args.warmup_operators else 0.0
    print(f"Input: {source_path} shape={clean.shape}")
    print(f"Device: {custom_tv.device}; CUDA extension before run: {cuda_extension_status()}")
    if args.warmup_operators:
        print(f"Operator warm-up/extension build: {warmup_seconds:.6f} s (excluded)")

    runs = {}
    for name, module in METHODS.items():
        runs[name] = run_variant(
            name, module, image_pil, clean, noisy, args.iterations, args.seed
        )
    save_results(
        Path(args.output), source_path, clean, noisy, runs,
        {
            "seed": args.seed,
            "sigma": args.sigma,
            "outer_iterations": args.iterations,
            "operator_warmup_seconds_excluded": warmup_seconds,
            "profiling_enabled": False,
        },
    )


if __name__ == "__main__":
    main()
