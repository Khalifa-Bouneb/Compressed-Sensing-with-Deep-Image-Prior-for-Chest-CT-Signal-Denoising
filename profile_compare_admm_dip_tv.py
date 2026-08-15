"""Profile eager FFT and custom CUDA ADMM-DIP-TV on the same real input."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, record_function

from compare_admm_dip_tv import (
    METHODS,
    custom_tv,
    prepare_experiment,
    run_variant,
    save_results,
    warm_up_operators,
)
from src.admm_cuda import cuda_extension_available, cuda_extension_status
from src.admm_profile import counter_snapshot, reset_counters


def event_record(event):
    return {
        "name": event.key,
        "calls": int(event.count),
        "self_cpu_time_us": float(event.self_cpu_time_total),
        "cpu_time_total_us": float(event.cpu_time_total),
        "self_cuda_time_us": float(getattr(event, "self_device_time_total", 0.0)),
        "cuda_time_total_us": float(getattr(event, "device_time_total", 0.0)),
        "cpu_memory_bytes": int(getattr(event, "cpu_memory_usage", 0)),
        "cuda_memory_bytes": int(getattr(event, "device_memory_usage", 0)),
    }


def summarize_events(averages):
    region_events = [
        event for event in averages
        if event.key.isupper()
        or event.key.endswith("_TOTAL")
        or event.key.endswith("_END_TO_END")
    ]
    operators = [
        event for event in averages
        if event.key.startswith("aten::")
        or event.key.startswith("Optimizer.step")
        or event.key.startswith("autograd::engine")
        or "dip_admm" in event.key
    ]
    cpu = sorted(operators, key=lambda event: event.self_cpu_time_total, reverse=True)
    cuda = sorted(
        operators,
        key=lambda event: getattr(event, "self_device_time_total", 0.0),
        reverse=True,
    )
    return {
        "regions": [event_record(event) for event in region_events],
        "top_cpu_operators": [event_record(event) for event in cpu[:25]],
        "top_cuda_operators": [event_record(event) for event in cuda[:25]],
    }


def profile_variant(name, module, image_pil, clean, noisy, args, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_counters()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
    ) as profiler:
        with record_function(f"{name.upper()}_END_TO_END"):
            run = run_variant(
                name, module, image_pil, clean, noisy, args.iterations, args.seed
            )

    trace_path = output_dir / "trace.json"
    profiler.export_chrome_trace(str(trace_path))
    averages = profiler.key_averages(group_by_input_shape=True)
    (output_dir / "cpu_table.txt").write_text(
        averages.table(sort_by="self_cpu_time_total", row_limit=100)
    )
    if torch.cuda.is_available():
        cuda_table = averages.table(sort_by="self_cuda_time_total", row_limit=100)
    else:
        cuda_table = "CUDA unavailable; no CUDA events were recorded.\n"
    (output_dir / "cuda_table.txt").write_text(cuda_table)

    details = summarize_events(averages)
    details.update({
        "elapsed_seconds": run["elapsed_seconds"],
        "counters": counter_snapshot(),
        "peak_cuda_allocated_bytes": run["memory"]["peak_cuda_allocated_bytes"],
        "peak_cuda_reserved_bytes": run["memory"]["peak_cuda_reserved_bytes"],
        "trace": str(trace_path.resolve()),
        "cpu_table": str((output_dir / "cpu_table.txt").resolve()),
        "cuda_table": str((output_dir / "cuda_table.txt").resolve()),
    })
    return run, details


def write_profile_summary(output_dir, profiles):
    eager = profiles["eager_fft"]
    custom = profiles["custom_cuda"]
    speedup = eager["elapsed_seconds"] / custom["elapsed_seconds"]

    def region_ms(details, name):
        return sum(
            event["cpu_time_total_us"] for event in details["regions"]
            if event["name"] == name
        ) / 1000.0

    eager_derivative_ms = region_ms(eager, "ADMM_DERIVATIVES_FFT")
    custom_derivative_ms = region_ms(custom, "ADMM_DERIVATIVES")
    eager_post_derivative_ms = region_ms(eager, "ADMM_DERIVATIVES_FFT_POST_STEP")
    custom_post_derivative_ms = region_ms(custom, "ADMM_DERIVATIVES_POST_STEP")
    eager_proximal_ms = (
        region_ms(eager, "TV_SHRINKAGE_EAGER")
        + region_ms(eager, "DUAL_UPDATE_EAGER")
    )
    custom_proximal_ms = region_ms(custom, "TV_SHRINKAGE_AND_DUAL_UPDATE")
    algorithm_regions = {
        "derivatives_inside_closure": {
            "eager_ms": eager_derivative_ms,
            "custom_ms": custom_derivative_ms,
            "speedup": eager_derivative_ms / custom_derivative_ms if custom_derivative_ms else None,
        },
        "post_step_derivatives": {
            "eager_ms": eager_post_derivative_ms,
            "custom_ms": custom_post_derivative_ms,
            "speedup": eager_post_derivative_ms / custom_post_derivative_ms if custom_post_derivative_ms else None,
        },
        "shrinkage_and_dual_update": {
            "eager_ms": eager_proximal_ms,
            "custom_ms": custom_proximal_ms,
            "speedup": eager_proximal_ms / custom_proximal_ms if custom_proximal_ms else None,
        },
    }
    payload = {
        "environment": {
            "pytorch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "custom_backend": (
                "native_cuda" if cuda_extension_available() else cuda_extension_status()
            ),
        },
        "eager_over_custom_profiled_speedup": speedup,
        "algorithm_region_comparison": algorithm_regions,
        "methods": profiles,
        "note": "Profiler-instrumented elapsed times include profiler overhead; use compare_admm_dip_tv.py for unprofiled performance.",
    }
    (output_dir / "profile_summary.json").write_text(
        json.dumps(payload, indent=2, default=float)
    )
    lines = [
        "PROFILE: EAGER FFT VS CUSTOM CUDA ADMM-DIP-TV",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"GPU: {payload['environment']['gpu']}",
        f"Custom backend: {payload['environment']['custom_backend']}",
        f"Eager profiled elapsed: {eager['elapsed_seconds']:.6f} s",
        f"Custom profiled elapsed: {custom['elapsed_seconds']:.6f} s",
        f"Profiled speedup (eager/custom): {speedup:.4f}x",
        f"Derivative speedup inside closures: {algorithm_regions['derivatives_inside_closure']['speedup']:.4f}x "
        f"({eager_derivative_ms:.4f} vs {custom_derivative_ms:.4f} ms)",
        f"Post-step derivative speedup: {algorithm_regions['post_step_derivatives']['speedup']:.4f}x "
        f"({eager_post_derivative_ms:.4f} vs {custom_post_derivative_ms:.4f} ms)",
        f"Shrinkage+dual speedup: {algorithm_regions['shrinkage_and_dual_update']['speedup']:.4f}x "
        f"({eager_proximal_ms:.4f} vs {custom_proximal_ms:.4f} ms)",
        "",
    ]
    for name, details in profiles.items():
        lines.extend([
            name,
            f"  Counters: {details['counters']}",
            f"  Peak CUDA allocated: {details['peak_cuda_allocated_bytes']} bytes",
            f"  Peak CUDA reserved: {details['peak_cuda_reserved_bytes']} bytes",
            f"  Trace: {details['trace']}",
            f"  CPU table: {details['cpu_table']}",
            f"  CUDA table: {details['cuda_table']}",
            "  Top CPU operators:",
        ])
        for event in details["top_cpu_operators"][:10]:
            lines.append(
                f"    {event['name']}: self={event['self_cpu_time_us'] / 1000:.3f} ms, "
                f"total={event['cpu_time_total_us'] / 1000:.3f} ms, calls={event['calls']}"
            )
        if torch.cuda.is_available():
            lines.append("  Top CUDA operators:")
            for event in details["top_cuda_operators"][:10]:
                lines.append(
                    f"    {event['name']}: self={event['self_cuda_time_us'] / 1000:.3f} ms, "
                    f"total={event['cuda_time_total_us'] / 1000:.3f} ms, calls={event['calls']}"
                )
        lines.append("")
    lines.append("Use the unprofiled comparison for the least-distorted end-to-end timing.")
    (output_dir / "profile_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="Dataset/BSDS300/BSDS300")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64, help="Center-crop size; use 0 for full image")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--inner-iterations", type=int, default=2)
    parser.add_argument("--lbfgs-max-iter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="profiling/admm_tv_comparison")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--warmup-operators", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["DIP_PROFILE"] = "1"
    os.environ["DIP_PROFILE_OUTER_ITERATIONS"] = str(args.iterations)
    os.environ["DIP_PROFILE_INNER_ITERATIONS"] = str(args.inner_iterations)
    os.environ["DIP_PROFILE_LBFGS_MAX_ITER"] = str(args.lbfgs_max_iter)

    source_path, image_pil, clean, noisy = prepare_experiment(args)
    warmup_seconds = warm_up_operators(clean.shape[-2:]) if args.warmup_operators else 0.0
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input: {source_path} shape={clean.shape}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"Custom backend after warm-up: {cuda_extension_status()}")
    print(f"Operator warm-up excluded: {warmup_seconds:.6f} s")

    runs = {}
    profiles = {}
    for name, module in METHODS.items():
        run, details = profile_variant(
            name, module, image_pil, clean, noisy, args, output_dir / name
        )
        runs[name] = run
        profiles[name] = details

    save_results(
        output_dir / "outputs", source_path, clean, noisy, runs,
        {
            "seed": args.seed,
            "sigma": args.sigma,
            "outer_iterations": args.iterations,
            "inner_iterations": args.inner_iterations,
            "lbfgs_max_iter": args.lbfgs_max_iter,
            "operator_warmup_seconds_excluded": warmup_seconds,
            "profiling_enabled": True,
        },
    )
    write_profile_summary(output_dir, profiles)
    print(f"Profile artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
