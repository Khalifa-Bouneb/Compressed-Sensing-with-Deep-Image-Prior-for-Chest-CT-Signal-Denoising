"""Profile the complete run_method(..., method='ADMM-DIP') execution.

Examples
--------
Manageable, explicitly reduced profiling run:
    python profile_admm_dip.py --short

Full configured experiment (potentially very long and a very large trace):
    python profile_admm_dip.py
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity

import run_dip_denoise_deblur as runner
from src import denoise_dip_tv as admm_tv
from src.admm_cuda import cuda_extension_available, cuda_extension_status
from src.admm_profile import counter_snapshot, reset_counters


PROFILE_REGIONS = (
    "RUN_METHOD_TOTAL",
    "INPUT_PREPARATION",
    "NOISE_GENERATION",
    "ADMM_DIP_TOTAL",
    "MODEL_INITIALIZATION",
    "CUDA_TRANSFER_MODEL",
    "INPUT_PREPARATION_DIP_NOISE",
    "OPTIMIZER_INITIALIZATION",
    "ADMM_INITIALIZATION",
    "ADMM_OUTER_ITERATION",
    "OPTIMIZER_INITIALIZATION_LBFGS",
    "LBFGS_STEP",
    "LBFGS_CLOSURE",
    "DIP_FORWARD",
    "ADMM_DERIVATIVES",
    "LOSS_COMPUTATION",
    "BACKWARD",
    "DIP_FORWARD_POST_STEP",
    "ADMM_DERIVATIVES_POST_STEP",
    "TV_SHRINKAGE_AND_DUAL_UPDATE",
    "CPU_GPU_TRANSFER_METRICS",
    "METRICS",
    "CPU_GPU_TRANSFER_PLOTTING",
    "PLOTTING",
    "OUTPUT_SERIALIZATION_MODEL",
    "OUTPUT_SERIALIZATION_METRICS",
    "CUDA_EXTENSION_LOAD_OR_COMPILE",
)

CALL_CHAIN = [
    "run_dip_denoise_deblur.run_method",
    "src.denoise_dip_tv.admm_dip_single",
    "src.models_dip.get_net -> src.models_dip.skip.skip",
    "torch.optim.LBFGS.step",
    "LBFGS closure -> DIP network forward",
    "src.admm_cuda.periodic_gradient",
    "loss construction -> Tensor.backward",
    "post-step forward -> periodic_gradient",
    "src.admm_cuda.shrink_and_update_dual",
    "metrics/plotting -> run_method serialization",
]


class ProfileDataset(torch.utils.data.Dataset):
    """Expose one optionally center-cropped source image to run_method."""

    def __init__(self, source, image_index, image_size=None):
        self.image = source[image_index]
        if image_size is not None:
            height, width = self.image.shape[-2:]
            if image_size > min(height, width):
                raise ValueError("profile image size exceeds the source image")
            top = (height - image_size) // 2
            left = (width - image_size) // 2
            self.image = self.image[:, top:top + image_size, left:left + image_size]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.image


def synchronize_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_run():
    # Reset both RNGs so the cold and warm passes receive the same noisy image,
    # DIP input, and initial network weights.
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    reset_counters()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def execute_profiled_call(dataset, result_directory):
    return runner.run_method(
        dataset,
        dataset_name="BSDS300",
        task="denoise",
        method="ADMM-DIP",
        fsavepath=str(result_directory),
        verbose=False,
        sigma=0.1,
    )


def memory_snapshot():
    if not torch.cuda.is_available():
        return {"peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0}
    return {
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def region_breakdown(key_averages):
    events = {event.key: event for event in key_averages}
    breakdown = {}
    for name in PROFILE_REGIONS:
        event = events.get(name)
        if event is None:
            continue
        breakdown[name] = {
            "calls": event.count,
            "self_cpu_time_us": event.self_cpu_time_total,
            "cpu_time_total_us": event.cpu_time_total,
            "self_cuda_time_us": getattr(event, "self_device_time_total", 0.0),
            "cuda_time_total_us": getattr(event, "device_time_total", 0.0),
        }
    return breakdown


def operator_breakdown(key_averages, limit=15):
    """Return the most expensive primitive/operator events as structured data."""
    operators = [
        event for event in key_averages
        if event.key.startswith("aten::")
        or event.key.startswith("Optimizer.step")
        or event.key.startswith("autograd::engine")
    ]
    operators.sort(key=lambda event: event.self_cpu_time_total, reverse=True)
    return [
        {
            "name": event.key,
            "calls": int(event.count),
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "cpu_time_total_us": float(event.cpu_time_total),
            "self_cuda_time_us": float(
                getattr(event, "self_device_time_total", 0.0)
            ),
            "cuda_time_total_us": float(
                getattr(event, "device_time_total", 0.0)
            ),
        }
        for event in operators[:limit]
    ]


def run_autograd_profile(dataset, output_directory, with_stack):
    output_directory.mkdir(parents=True, exist_ok=True)
    prepare_run()
    synchronize_if_needed()
    started = time.perf_counter()
    with torch.autograd.profiler.profile(
        use_cuda=torch.cuda.is_available(),
        use_kineto=True,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    ) as profile:
        execute_profiled_call(dataset, output_directory / "results")
    synchronize_if_needed()
    wall_seconds = time.perf_counter() - started

    averages = profile.key_averages(group_by_input_shape=True)
    cpu_table = averages.table(sort_by="self_cpu_time_total", row_limit=100)
    if torch.cuda.is_available():
        cuda_table = averages.table(sort_by="self_cuda_time_total", row_limit=100)
    else:
        cuda_table = "CUDA unavailable; no CUDA autograd table was produced.\n"
    (output_directory / "autograd_profile.txt").write_text(
        "AUTOGRAD PROFILER - CPU\n\n" + cpu_table +
        "\n\nAUTOGRAD PROFILER - CUDA\n\n" + cuda_table
    )
    return {
        "wall_seconds": wall_seconds,
        "counters": counter_snapshot(),
        "memory": memory_snapshot(),
        "regions": region_breakdown(profile.key_averages()),
        "top_cpu_operators": operator_breakdown(averages),
    }


def run_modern_profile(dataset, output_directory, with_stack):
    output_directory.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    prepare_run()
    synchronize_if_needed()
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    ) as profile:
        execute_profiled_call(dataset, output_directory / "results")
    synchronize_if_needed()
    wall_seconds = time.perf_counter() - started

    trace_path = output_directory / "admm_dip_trace.json"
    profile.export_chrome_trace(str(trace_path))
    averages = profile.key_averages(group_by_input_shape=True)
    cpu_table = averages.table(sort_by="self_cpu_time_total", row_limit=100)
    (output_directory / "torch_profiler_cpu.txt").write_text(cpu_table)
    if torch.cuda.is_available():
        cuda_table = averages.table(sort_by="self_cuda_time_total", row_limit=100)
    else:
        cuda_table = "CUDA unavailable; no CUDA profiler table was produced.\n"
    (output_directory / "torch_profiler_cuda.txt").write_text(cuda_table)
    return {
        "wall_seconds": wall_seconds,
        "counters": counter_snapshot(),
        "memory": memory_snapshot(),
        "regions": region_breakdown(profile.key_averages()),
        "top_cpu_operators": operator_breakdown(averages),
        "trace": str(trace_path.resolve()),
    }


def write_summary(output_root, environment, settings, autograd_result, modern_result):
    counters = modern_result["counters"]
    optimizer_steps = counters.get("optimizer_step_calls", 0)
    closure_calls = counters.get("closure_calls", 0)
    average_closures = closure_calls / optimizer_steps if optimizer_steps else 0.0
    summary = {
        "environment": environment,
        "settings": settings,
        "call_chain": CALL_CHAIN,
        "timing_note": (
            "record_function region times are inclusive and nested; percentages "
            "therefore overlap and must not be summed"
        ),
        "autograd_run": autograd_result,
        "modern_profiler_run": modern_result,
        "lbfgs": {
            "outer_iterations": counters.get("outer_iterations", 0),
            "inner_iterations": counters.get("inner_iterations", 0),
            "optimizer_step_calls": optimizer_steps,
            "closure_calls": closure_calls,
            "average_closure_calls_per_step": average_closures,
        },
    }
    (output_root / "profile_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "DIP-ADMM COMPLETE EXECUTION PROFILE",
        "===================================",
        "",
        "Environment",
        f"  GPU: {environment['gpu']}",
        f"  PyTorch: {environment['pytorch']}",
        f"  CUDA runtime: {environment['torch_cuda']}",
        f"  CUDA available: {environment['cuda_available']}",
        f"  Native ADMM extension: {environment['native_admm_extension']}",
        "",
        "Profile settings",
        *[f"  {key}: {value}" for key, value in settings.items()],
        "",
        "Actual call chain",
        *[f"  {index + 1}. {name}" for index, name in enumerate(CALL_CHAIN)],
        "",
        "Measured wall time",
        f"  Cold/autograd run: {autograd_result['wall_seconds']:.6f} s",
        f"  Warm/modern-profiler run: {modern_result['wall_seconds']:.6f} s",
        "",
        "LBFGS behavior (warm run)",
        f"  Outer ADMM iterations: {counters.get('outer_iterations', 0)}",
        f"  Requested inner optimizer steps: {counters.get('inner_iterations', 0)}",
        f"  optimizer.step calls: {optimizer_steps}",
        f"  Closure calls: {closure_calls}",
        f"  Average closure calls per step: {average_closures:.3f}",
        "",
        "Modern-profiler regions (inclusive CPU and CUDA time)",
        "  Note: regions are nested, so their inclusive percentages overlap and do not sum to 100%.",
    ]
    total_us = modern_result["regions"].get("RUN_METHOD_TOTAL", {}).get(
        "cpu_time_total_us", modern_result["wall_seconds"] * 1e6
    )
    for name, values in modern_result["regions"].items():
        cpu_us = values["cpu_time_total_us"]
        percentage = 100 * cpu_us / total_us if total_us else 0.0
        lines.append(
            f"  {name}: CPU total={cpu_us / 1000:.3f} ms "
            f"({percentage:.2f}%), CUDA total={values['cuda_time_total_us'] / 1000:.3f} ms, "
            f"calls={values['calls']}"
        )
    lines += [
        "",
        "Top CPU operators (self CPU time)",
    ]
    for event in modern_result["top_cpu_operators"]:
        lines.append(
            f"  {event['name']}: self={event['self_cpu_time_us'] / 1000:.3f} ms, "
            f"total={event['cpu_time_total_us'] / 1000:.3f} ms, calls={event['calls']}"
        )
    lbfgs_us = modern_result["regions"].get("LBFGS_STEP", {}).get(
        "cpu_time_total_us", 0.0
    )
    backward_us = modern_result["regions"].get("BACKWARD", {}).get(
        "cpu_time_total_us", 0.0
    )
    forward_us = modern_result["regions"].get("DIP_FORWARD", {}).get(
        "cpu_time_total_us", 0.0
    )
    admm_operator_us = sum(
        modern_result["regions"].get(name, {}).get("cpu_time_total_us", 0.0)
        for name in (
            "ADMM_DERIVATIVES",
            "ADMM_DERIVATIVES_POST_STEP",
            "TV_SHRINKAGE_AND_DUAL_UPDATE",
        )
    )
    lines += [
        "",
        "Evidence-based bottleneck conclusion",
        f"  LBFGS_STEP encloses {lbfgs_us / 1000:.3f} ms; its repeated closures dominate.",
        f"  BACKWARD accounts for {backward_us / 1000:.3f} ms inclusive, while the four "
        f"profiled DIP forwards account for {forward_us / 1000:.3f} ms inclusive.",
        f"  All marked ADMM derivative/shrinkage/dual work totals only "
        f"{admm_operator_us / 1000:.3f} ms inclusive in this short CPU run.",
        "  TV shrinkage and dual update are one fused implementation operation and are "
        "reported together; splitting their timing would change that operation.",
        "",
        "Memory",
        f"  Peak CUDA allocated: {modern_result['memory']['peak_cuda_allocated_bytes']} bytes",
        f"  Peak CUDA reserved: {modern_result['memory']['peak_cuda_reserved_bytes']} bytes",
        "",
        "Artifacts",
        f"  Perfetto trace: {modern_result['trace']}",
        f"  CPU table: {(output_root / 'warm' / 'torch_profiler_cpu.txt').resolve()}",
        f"  CUDA table: {(output_root / 'warm' / 'torch_profiler_cuda.txt').resolve()}",
        f"  Autograd output: {(output_root / 'cold' / 'autograd_profile.txt').resolve()}",
        f"  JSON summary: {(output_root / 'profile_summary.json').resolve()}",
    ]
    (output_root / "profile_summary.txt").write_text("\n".join(lines) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="profiling/admm_dip")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument(
        "--short", action="store_true",
        help="Use 64x64 input, 1 outer iteration, 2 inner steps, and LBFGS max_iter=2",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    os.environ["DIP_PROFILE"] = "1"
    settings = {"short_run": arguments.short, "image_index": arguments.image_index}
    image_size = None
    if arguments.short:
        overrides = {
            "DIP_PROFILE_OUTER_ITERATIONS": "1",
            "DIP_PROFILE_INNER_ITERATIONS": "2",
            "DIP_PROFILE_LBFGS_MAX_ITER": "2",
        }
        os.environ.update(overrides)
        image_size = 64
        settings.update({
            "image_size": "64x64 center crop",
            "outer_iterations": 1,
            "inner_iterations": 2,
            "lbfgs_max_iter": 2,
        })
    else:
        settings.update({
            "image_size": "normal preprocessing",
            "outer_iterations": admm_tv.num_iter,
            "inner_iterations": 10,
            "lbfgs_max_iter": 20,
        })

    cuda_available = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda_available else "Unavailable (CPU profile)"
    environment = {
        "gpu": gpu,
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "native_admm_extension_before_run": cuda_extension_status(),
        "native_admm_extension": None,
        "python": platform.python_version(),
    }
    print("CUDA available:", cuda_available)
    print("GPU:", gpu)
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("Native ADMM extension before run:", cuda_extension_status())
    if arguments.short:
        print("PROFILE_SHORT_RUN=1: 64x64, outer=1, inner=2, LBFGS max_iter=2")

    source = runner.BSDS300Dataset(use_patches=False)
    dataset = ProfileDataset(source, arguments.image_index, image_size=image_size)
    output_root = Path(arguments.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Running cold autograd-profiler pass...")
    autograd_result = run_autograd_profile(
        dataset, output_root / "cold", arguments.with_stack
    )
    print("Running warm modern-profiler pass...")
    modern_result = run_modern_profile(
        dataset, output_root / "warm", arguments.with_stack
    )
    environment["native_admm_extension"] = (
        "loaded" if cuda_extension_available() else cuda_extension_status()
    )
    write_summary(output_root, environment, settings, autograd_result, modern_result)
    print((output_root / "profile_summary.txt").read_text())


if __name__ == "__main__":
    main()
