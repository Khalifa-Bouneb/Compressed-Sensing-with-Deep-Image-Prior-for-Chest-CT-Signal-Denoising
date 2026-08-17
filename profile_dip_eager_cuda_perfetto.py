#!/usr/bin/env python3
"""Profile eager and native-CUDA ADMM-DIP TV/WTV in one Perfetto trace.

The four implementations are run sequentially on the same clean image and the
same speckle-corrupted observation.  The output is a Chrome trace JSON that can
be opened directly at https://ui.perfetto.dev.  In addition to PyTorch CPU/CUDA
events, it contains one ``loss`` and one ``psnr_gt`` counter track per method.

Example:
    python profile_dip_eager_cuda_perfetto.py --iterations 20
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import Subset

from run_dip_denoise_deblur import BSDS300Dataset
from src import denoise_dip_tv_cuda as tv_cuda
from src import denoise_dip_tv_eager as tv_eager
from src import denoise_dip_tvw_cuda as wtv_cuda
from src import denoise_dip_tvw_eager as wtv_eager
from src.admm_cuda import cuda_extension_available, cuda_extension_status
from src.admm_profile import counter_snapshot, reset_counters
from src.utils import D, add_speckle, process_subset, psf2otf


METHODS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("TV / eager PyTorch", tv_eager.admm_dip_single_eager),
    ("TV / custom CUDA", tv_cuda.admm_dip_single_cuda),
    ("WTV / eager PyTorch", wtv_eager.admm_dip_wtv_single_eager),
    ("WTV / custom CUDA", wtv_cuda.admm_dip_wtv_single_cuda),
)
COMMON_ITERATION_REGIONS = {
    "ADMM_OUTER_ITERATION",
    "DIP_FORWARD",
    "ADMM_DERIVATIVES",
    "LOSS_COMPUTATION",
    "BACKWARD",
    "ADAM_STEP",
    "DIP_FORWARD_POST_STEP",
    "ADMM_DERIVATIVES_POST_STEP",
    "CPU_GPU_TRANSFER_METRICS",
    "METRICS",
}
EXPECTED_REGIONS = {
    "TV / eager PyTorch": COMMON_ITERATION_REGIONS | {
        "ADMM_DIP_TOTAL", "MODEL_INITIALIZATION", "CUDA_TRANSFER_MODEL",
        "INPUT_PREPARATION_DIP_NOISE", "OPTIMIZER_INITIALIZATION",
        "ADMM_INITIALIZATION", "TV_SHRINKAGE_AND_DUAL_UPDATE",
    },
    "TV / custom CUDA": COMMON_ITERATION_REGIONS | {
        "ADMM_DIP_TOTAL", "MODEL_INITIALIZATION", "CUDA_TRANSFER_MODEL",
        "INPUT_PREPARATION_DIP_NOISE", "OPTIMIZER_INITIALIZATION",
        "ADMM_INITIALIZATION", "TV_SHRINKAGE_AND_DUAL_UPDATE",
    },
    "WTV / eager PyTorch": COMMON_ITERATION_REGIONS | {
        "ADMM_DIP_WTV_TOTAL", "MODEL_INITIALIZATION", "CUDA_TRANSFER_MODEL",
        "INPUT_PREPARATION_DIP_NOISE", "OPTIMIZER_INITIALIZATION",
        "FFT_FILTER_INITIALIZATION", "ADMM_INITIALIZATION",
        "ADAPTIVE_WEIGHT_COMPUTATION",
        "WTV_SHRINKAGE_AND_DUAL_UPDATE",
    },
    "WTV / custom CUDA": COMMON_ITERATION_REGIONS | {
        "ADMM_DIP_WTV_TOTAL", "MODEL_INITIALIZATION", "CUDA_TRANSFER_MODEL",
        "INPUT_PREPARATION_DIP_NOISE", "OPTIMIZER_INITIALIZATION",
        "ADMM_INITIALIZATION", "ADAPTIVE_WEIGHT_COMPUTATION",
        "WTV_SHRINKAGE_AND_DUAL_UPDATE",
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=root / "Dataset/BSDS300/BSDS300",
        help="BSDS300 root passed to the project BSDS300Dataset",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="deterministic zero-based full-image dataset index",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--inner-iterations",
        type=int,
        default=None,
        help="optional profiling-only inner-iteration override",
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "profiling/dip_eager_cuda_perfetto.json",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="metric PNG path (default: OUTPUT with _metrics.png suffix)",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="record Python stacks (larger and slower trace)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow a CPU-only diagnostic run; this is not a CUDA comparison",
    )
    parser.add_argument(
        "--require-native-cuda",
        action="store_true",
        help="fail if the custom CUDA extension cannot be loaded",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.inner_iterations is not None and args.inner_iterations < 1:
        raise ValueError("--inner-iterations must be at least 1")
    if args.image_index < 0:
        raise ValueError("--image-index cannot be negative")
    if args.sigma < 0:
        raise ValueError("--sigma cannot be negative")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(
            f"BSDS300 dataset root does not exist: {args.dataset_root}"
        )
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA device is available. Run this on the CUDA machine, or use "
            "--allow-cpu only to validate the trace workflow."
        )


def configure_algorithms(
    iterations: int, inner_iterations: int | None
) -> None:
    os.environ["DIP_PROFILE"] = "1"
    os.environ["DIP_PROFILE_OUTER_ITERATIONS"] = str(iterations)
    if inner_iterations is None:
        os.environ.pop("DIP_PROFILE_INNER_ITERATIONS", None)
    else:
        os.environ["DIP_PROFILE_INNER_ITERATIONS"] = str(inner_iterations)
    tv_eager.PLOT = False
    tv_cuda.PLOT = False
    wtv_eager.PLOT = False
    wtv_cuda.PLOT = False


def load_observation(
    dataset_root: Path, image_index: int, sigma: float, seed: int
) -> tuple[Any, np.ndarray, np.ndarray, int, Path]:
    dataset = BSDS300Dataset(root=str(dataset_root), use_patches=False)
    if image_index >= len(dataset):
        raise IndexError(
            f"--image-index {image_index} is outside [0, {len(dataset) - 1}]"
        )

    # The runner calls process_subset on a dataset subset, then indexes the two
    # returned lists.  A one-item deterministic Subset preserves that path.
    subset = Subset(dataset, [image_index])
    processed_pils, processed_tensors = process_subset(subset)
    image_pil = processed_pils[0]
    clean_tensor = processed_tensors[0]
    clean = clean_tensor.numpy()

    reset_seed(seed)
    noisy = add_speckle(clean_tensor, sigma).numpy()
    if clean.shape != noisy.shape:
        raise AssertionError(
            f"Clean/noisy shape mismatch: {clean.shape} != {noisy.shape}"
        )
    selected_image = Path(
        dataset._resolve_image_files(str(dataset_root))[image_index]
    )
    return image_pil, clean, noisy, image_index, selected_image


def print_preprocessing_summary(
    dataset_root: Path,
    image_index: int,
    selected_image: Path,
    clean: np.ndarray,
    noisy: np.ndarray,
) -> None:
    print(
        f"Selected BSDS300 item: index={image_index}, "
        f"image={selected_image.resolve()}"
    )
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Clean: shape={clean.shape}, dtype={clean.dtype}, min={clean.min():.6f}, max={clean.max():.6f}")
    print(f"Noisy: shape={noisy.shape}, dtype={noisy.dtype}, min={noisy.min():.6f}, max={noisy.max():.6f}")
    print("\nPreprocessing mapping")
    print("run_dip_denoise_deblur.py       profiler")
    print("------------------------------------------------")
    print("BSDS300Dataset                  same imported class")
    print("orientation transpose           same dataset path")
    print("HWC -> CHW                      same dataset path")
    print("float32 / 255                   same dataset path")
    print("process_subset                  same function")
    print("add_speckle                     same function")
    print("denoiser arguments              same (PIL, clean ndarray, noisy ndarray)")


def warm_native_operators() -> str:
    """Compile/load the extension before measurement, avoiding cold-start bias."""
    if not torch.cuda.is_available():
        return cuda_extension_status()
    from src.admm_cuda import periodic_gradient, shrink_and_update_dual

    sample = torch.zeros((1, 1, 8, 8), device="cuda")
    grad_h, grad_v = periodic_gradient(sample)
    shrink_and_update_dual(
        grad_h, grad_v, torch.zeros_like(sample), torch.zeros_like(sample), 0.1
    )
    torch.cuda.synchronize()
    return cuda_extension_status()


def warm_fft_plans(clean: np.ndarray) -> None:
    """Create the eager-TV and eager-WTV FFT plans before measurement."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h, w = clean.shape[-2:]
    sample = torch.zeros((1, *clean.shape), device=device)
    # TV currently builds its filters from NumPy float64 arrays; WTV uses
    # float32 arrays.  cuFFT caches plans by dtype, so warm both exact paths.
    for dtype in (np.float64, np.float32):
        dh_psf = np.array(
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=dtype
        )
        dv_psf = np.array(
            [[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=dtype
        )
        dh_dft = torch.from_numpy(psf2otf(dh_psf, [h, w])).to(device)
        dv_dft = torch.from_numpy(psf2otf(dv_psf, [h, w])).to(device)
        D(sample, dh_dft, dv_dft)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_profile(
    args: argparse.Namespace,
    image_pil: Any,
    clean: np.ndarray,
    noisy: np.ndarray,
) -> tuple[Any, list[dict[str, Any]]]:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    results: list[dict[str, Any]] = []
    clean_reference = clean.copy()
    noisy_reference = noisy.copy()
    argument_identities = (id(image_pil), id(clean), id(noisy))
    profiler = profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
    )
    with profiler:
        for method_index, (name, function) in enumerate(METHODS):
            reset_seed(args.seed)
            reset_counters()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start_perf = time.perf_counter()
            with record_function(f"BENCHMARK::{name}"):
                metrics, model_dump = function(
                    image_pil, clean, noisy, 1, verbose=False
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            duration_s = time.perf_counter() - start_perf
            print()
            if (id(image_pil), id(clean), id(noisy)) != argument_identities:
                raise AssertionError(
                    "The shared PIL/clean/noisy argument identity changed"
                )
            if not np.array_equal(clean, clean_reference):
                raise AssertionError(f"{name} modified the shared clean array")
            if not np.array_equal(noisy, noisy_reference):
                raise AssertionError(f"{name} modified the shared noisy array")

            results.append(
                {
                    "name": name,
                    "method_index": method_index,
                    "duration_s": duration_s,
                    "iterations": len(metrics),
                    "metrics": metrics,
                    "execution_counters": counter_snapshot(),
                    "peak_cuda_memory_bytes": (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else 0
                    ),
                }
            )
            del model_dump
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return profiler, results


def metric_trace_events(
    trace: dict[str, Any], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build counters on the profiler's clock from recorded iteration events."""
    complete_events = [
        event
        for event in trace.get("traceEvents", [])
        if event.get("ph") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    events: list[dict[str, Any]] = []
    for result in results:
        benchmark_name = f"BENCHMARK::{result['name']}"
        benchmarks = [
            event for event in complete_events
            if event.get("name") == benchmark_name
        ]
        if len(benchmarks) != 1:
            raise RuntimeError(
                f"Trace contains {len(benchmarks)} occurrences of "
                f"{benchmark_name}"
            )
        benchmark = benchmarks[0]
        benchmark_start = benchmark["ts"]
        benchmark_end = benchmark_start + benchmark["dur"]
        iteration_events = sorted(
            (
                event for event in complete_events
                if event.get("name") == "ADMM_OUTER_ITERATION"
                and event.get("pid") == benchmark.get("pid")
                and event.get("tid") == benchmark.get("tid")
                and event["ts"] >= benchmark_start
                and event["ts"] + event["dur"] <= benchmark_end
            ),
            key=lambda event: event["ts"],
        )
        if len(iteration_events) == len(result["metrics"]):
            timestamps = [
                event["ts"] + event["dur"] for event in iteration_events
            ]
        else:
            # Retain counter tracks if a future profiler backend omits nested
            # iteration events, while keeping timestamps on the trace clock.
            step = benchmark["dur"] / max(len(result["metrics"]), 1)
            timestamps = [
                benchmark_start + (index + 1) * step
                for index in range(len(result["metrics"]))
            ]

        # Separate pseudo-processes make every method/metric a distinct
        # counter track in Perfetto instead of combining identically named data.
        base_pid = 91_000 + 10 * result["method_index"]
        for offset, metric_key, display_name in (
            (0, "loss", "loss"),
            (1, "PSNR_gt", "psnr_gt"),
        ):
            pid = base_pid + offset
            events.append(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": pid,
                    "tid": 0,
                    "args": {"name": f"{result['name']} — {display_name}"},
                }
            )
            for timestamp, metric in zip(
                timestamps, result["metrics"]
            ):
                events.append(
                    {
                        "ph": "C",
                        "cat": "DIP metrics",
                        "name": display_name,
                        "pid": pid,
                        "tid": 0,
                        "ts": timestamp,
                        "args": {"value": float(metric[metric_key])},
                    }
                )
    return events


def export_trace(
    profiler: Any,
    results: list[dict[str, Any]],
    output: Path,
    args: argparse.Namespace,
    extension_status: str,
) -> dict[str, list[str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = Path(temp.name)
        profiler.export_chrome_trace(str(temp_path))
        with temp_path.open("r", encoding="utf-8") as file:
            trace = json.load(file)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    trace.setdefault("traceEvents", []).extend(metric_trace_events(trace, results))
    confirmed_regions = validate_trace(trace)
    speedups = {
        "tv_custom_cuda_vs_eager": results[0]["duration_s"]
        / results[1]["duration_s"],
        "wtv_custom_cuda_vs_eager": results[2]["duration_s"]
        / results[3]["duration_s"],
    }
    trace.setdefault("otherData", {})["dip_benchmark"] = {
        "dataset_root": str(args.dataset_root.resolve()),
        "image_index": args.image_index,
        "sigma": args.sigma,
        "seed": args.seed,
        "requested_iterations": args.iterations,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "custom_cuda_extension": extension_status,
        "speedup": speedups,
        "results": [
            {
                "method": item["name"],
                "duration_s": item["duration_s"],
                "iterations": item["iterations"],
                "seconds_per_iteration": item["duration_s"]
                / max(item["iterations"], 1),
                "peak_cuda_memory_bytes": item["peak_cuda_memory_bytes"],
                "execution_counters": item["execution_counters"],
                "final_loss": float(item["metrics"][-1]["loss"]),
                "final_psnr_gt": float(item["metrics"][-1]["PSNR_gt"]),
            }
            for item in results
        ],
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(trace, file, separators=(",", ":"))
    return confirmed_regions


def validate_trace(trace: dict[str, Any]) -> dict[str, list[str]]:
    """Verify internal record_function events inside each benchmark event."""
    events = trace.get("traceEvents", [])
    complete_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    confirmed: dict[str, list[str]] = {}
    for method_name, expected in EXPECTED_REGIONS.items():
        benchmark_name = f"BENCHMARK::{method_name}"
        benchmarks = [
            event for event in complete_events
            if event.get("name") == benchmark_name
        ]
        if len(benchmarks) != 1:
            raise RuntimeError(
                f"Trace contains {len(benchmarks)} occurrences of {benchmark_name}"
            )
        benchmark = benchmarks[0]
        start = benchmark["ts"]
        end = start + benchmark["dur"]
        nested_names = {
            str(event.get("name"))
            for event in complete_events
            if event.get("pid") == benchmark.get("pid")
            and event.get("tid") == benchmark.get("tid")
            and event["ts"] >= start
            and event["ts"] + event["dur"] <= end
        }
        missing = expected - nested_names
        if missing:
            raise RuntimeError(
                f"Trace is missing regions inside {benchmark_name}: "
                + ", ".join(sorted(missing))
            )
        confirmed[method_name] = sorted(expected)

    counter_names = {
        event.get("name") for event in events if event.get("ph") == "C"
    }
    missing_counters = {"loss", "psnr_gt"} - counter_names
    if missing_counters:
        raise RuntimeError(
            "Trace is missing metric counter tracks: "
            + ", ".join(sorted(missing_counters))
        )
    return confirmed


def save_metric_plot(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for result in results:
        iterations = [entry["iteration"] for entry in result["metrics"]]
        axes[0].plot(
            iterations,
            [entry["loss"] for entry in result["metrics"]],
            label=result["name"],
        )
        axes[1].plot(
            iterations,
            [entry["PSNR_gt"] for entry in result["metrics"]],
            label=result["name"],
        )
    axes[0].set(title="Loss", xlabel="iteration", ylabel="loss")
    axes[0].set_yscale("log")
    axes[1].set(title="PSNR vs ground truth", xlabel="iteration", ylabel="dB")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nMethod                         seconds   sec/iter   final loss   PSNR_gt")
    print("-" * 78)
    for result in results:
        final = result["metrics"][-1]
        print(
            f"{result['name']:<30} {result['duration_s']:>8.3f} "
            f"{result['duration_s'] / result['iterations']:>10.4f} "
            f"{final['loss']:>12.5g} {final['PSNR_gt']:>9.3f}"
        )
    print(
        f"\nTV custom-CUDA speedup:  "
        f"{results[0]['duration_s'] / results[1]['duration_s']:.3f}x"
    )
    print(
        f"WTV custom-CUDA speedup: "
        f"{results[2]['duration_s'] / results[3]['duration_s']:.3f}x"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_algorithms(args.iterations, args.inner_iterations)
    image_pil, clean, noisy, image_index, selected_image = load_observation(
        args.dataset_root, args.image_index, args.sigma, args.seed
    )
    print_preprocessing_summary(
        args.dataset_root, image_index, selected_image, clean, noisy
    )

    extension_status = warm_native_operators()
    warm_fft_plans(clean)
    if args.require_native_cuda and not cuda_extension_available():
        raise RuntimeError(f"Native CUDA extension is not active: {extension_status}")
    if "fallback" in extension_status:
        print(f"WARNING: custom CUDA operators are using fallback: {extension_status}")

    profiler, results = run_profile(args, image_pil, clean, noisy)
    confirmed_regions = export_trace(
        profiler, results, args.output, args, extension_status
    )
    plot_path = args.plot or args.output.with_name(
        f"{args.output.stem}_metrics.png"
    )
    save_metric_plot(results, plot_path)
    print_summary(results)
    print(f"\nPerfetto trace: {args.output.resolve()}")
    print(f"Metric plot:    {plot_path.resolve()}")
    print("\nConfirmed internal regions in exported trace:")
    for method_name, region_names in confirmed_regions.items():
        print(f"  {method_name}: {', '.join(region_names)}")
    print("Confirmed metric counter tracks: loss, psnr_gt")
    print("Open the JSON at https://ui.perfetto.dev and drag in the counter tracks.")


if __name__ == "__main__":
    main()
