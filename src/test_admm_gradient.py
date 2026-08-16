import os
from glob import glob

import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset

from config import *
from models_dip import *
from utils import *

if __package__:
    from .admm_cuda import cuda_extension_status, periodic_gradient, shrink_and_update_dual
else:
    from admm_cuda import cuda_extension_status, periodic_gradient, shrink_and_update_dual


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WARMUP_ITERATIONS = 20
PROFILE_ITERATIONS = 100
BENCHMARK_ITERATIONS = 100


# ============================================================
# Dataset
# ============================================================

class BSDS300Dataset(Dataset):

    def __init__(self, root="../Dataset/BSDS300/BSDS300", patch_size=32, use_patches=True):
        files = self._resolve_image_files(root)
        self.use_patches = use_patches
        self.images = self.load_images(files)
        self.patches = self.patchify(self.images, patch_size)

        self.mean = torch.mean(self.patches)
        self.std = torch.std(self.patches)

    def _resolve_image_files(self, root, split=None):
        image_root = os.path.join(root, "images")
        candidates = []

        if split is not None:
            candidates.append(os.path.join(image_root, split, "*"))
        candidates.append(os.path.join(image_root, "*"))

        for pattern in candidates:
            files = sorted(fname for fname in glob(pattern) if os.path.isfile(fname))
            if files:
                return files

        searched = ", ".join(candidates)
        raise FileNotFoundError(
            f"No image files were found for BSDS300Dataset. Searched: {searched}. Check the root path: {root}"
        )

    def load_images(self, files):
        out = []
        for fname in files:
            img = skimage.io.imread(fname)
            if img.shape[0] > img.shape[1]:
                img = img.transpose(1, 0, 2)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            out.append(torch.from_numpy(img))
        return torch.stack(out)

    def patchify(self, img_array, patch_size):
        patches = img_array.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(patches.shape[0], 3, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, 3, patch_size, patch_size)
        return patches

    def __len__(self):
        return self.patches.shape[0] if self.use_patches else self.images.shape[0]

    def __getitem__(self, idx):
        return self.patches[idx] if self.use_patches else self.images[idx]


# ============================================================
# Prepare ONE real DIP output
# ============================================================

def prepare_real_dip_output(dataset, subset_size=1):
    """
    Generate one real DIP output.

    The exact same tensor is used for:
        - FFT gradient
        - custom CUDA gradient
    """
    subset = get_random_subset(dataset, subset_size=subset_size)
    processed_pils, processed_tensors = process_subset(subset)

    img_pil, img_tensor = processed_pils[0], processed_tensors[0]
    img_np = img_tensor.numpy()

    # --------------------------------------------------------
    # DIP noise input
    # --------------------------------------------------------
    net_input = get_noise(32, "noise", (img_pil.size[1], img_pil.size[0])).to(device).detach()

    # --------------------------------------------------------
    # DIP network
    # --------------------------------------------------------
    net = get_net(
        32, "skip", "pad", skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, upsample_mode="bilinear"
    ).to(device)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    out = net(net_input)
    h, w = img_np.shape[-2], img_np.shape[-1]

    return out, h, w, net, net_input, img_tensor


# ============================================================
# Prepare FFT derivative filters
# ============================================================

def prepare_fft_filters(h, w, device):
    """
    Build FFT derivative filters once.

    Their initialization is deliberately excluded from
    the timing because the original ADMM algorithm also
    creates them once before the optimization loop.
    """
    Dh_psf = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    Dv_psf = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])

    Dh_DFT = torch.from_numpy(psf2otf(Dh_psf, [h, w])).to(device)
    Dv_DFT = torch.from_numpy(psf2otf(Dv_psf, [h, w])).to(device)

    return Dh_DFT, Dv_DFT


# ============================================================
# FFT implementation
# ============================================================

def run_fft_gradient(out, Dh_DFT, Dv_DFT):
    """Original FFT gradient implementation."""
    Dh_out, Dv_out = D(out, Dh_DFT, Dv_DFT)
    return Dh_out, Dv_out


# ============================================================
# CUDA implementation
# ============================================================

def run_cuda_gradient(out):
    """Custom CUDA gradient implementation."""
    out = out.contiguous()
    Dh_out, Dv_out = periodic_gradient(out)
    return Dh_out, Dv_out


# ============================================================
# Numerical correctness comparison
# ============================================================

def compare_gradients(Dh_fft, Dv_fft, Dh_cuda, Dv_cuda):
    dh_error = (Dh_fft - Dh_cuda).abs()
    dv_error = (Dv_fft - Dv_cuda).abs()

    print("\n============================================================")
    print("FORWARD CORRECTNESS COMPARISON")
    print("============================================================")

    print("\nFFT horizontal first 10:")
    print(Dh_fft.flatten()[:10])

    print("\nCUDA horizontal first 10:")
    print(Dh_cuda.flatten()[:10])

    print("\nHorizontal max absolute error :", dh_error.max().item())
    print("Horizontal mean absolute error:", dh_error.mean().item())

    print("\nFFT vertical first 10:")
    print(Dv_fft.flatten()[:10])

    print("\nCUDA vertical first 10:")
    print(Dv_cuda.flatten()[:10])

    print("\nVertical max absolute error :", dv_error.max().item())
    print("Vertical mean absolute error:", dv_error.mean().item())

    torch.testing.assert_close(Dh_cuda, Dh_fft, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(Dv_cuda, Dv_fft, rtol=1e-4, atol=1e-5)

    print("\nPASS: CUDA gradient matches FFT gradient.")


# ============================================================
# Profile FFT with torch.autograd.profiler
# ============================================================

def profile_fft_gradient(out, Dh_DFT, Dv_DFT, warmup=WARMUP_ITERATIONS, iterations=PROFILE_ITERATIONS):
    print("\nWarming up FFT implementation...")
    for _ in range(warmup):
        run_fft_gradient(out, Dh_DFT, Dv_DFT)

    torch.cuda.synchronize()
    print("Profiling FFT implementation...")

    with torch.autograd.profiler.profile(use_device="cuda", record_shapes=True) as prof:
        for _ in range(iterations):
            with torch.autograd.profiler.record_function("FFT_GRADIENT"):
                Dh_out, Dv_out = run_fft_gradient(out, Dh_DFT, Dv_DFT)

    torch.cuda.synchronize()
    return Dh_out, Dv_out, prof


# ============================================================
# Profile CUDA with torch.autograd.profiler
# ============================================================

def profile_cuda_gradient(out, warmup=WARMUP_ITERATIONS, iterations=PROFILE_ITERATIONS):
    out = out.contiguous()

    # --------------------------------------------------------
    # Trigger extension compilation before timing
    # --------------------------------------------------------
    run_cuda_gradient(out)
    torch.cuda.synchronize()

    print("\nWarming up CUDA implementation...")
    for _ in range(warmup):
        run_cuda_gradient(out)

    torch.cuda.synchronize()
    print("Profiling CUDA implementation...")

    with torch.autograd.profiler.profile(use_device="cuda", record_shapes=True) as prof:
        for _ in range(iterations):
            with torch.autograd.profiler.record_function("CUDA_GRADIENT"):
                Dh_out, Dv_out = run_cuda_gradient(out)

    torch.cuda.synchronize()
    return Dh_out, Dv_out, prof


# ============================================================
# CUDA event benchmark
# ============================================================

def benchmark_cuda_time(fn, warmup=WARMUP_ITERATIONS, iterations=BENCHMARK_ITERATIONS):
    """
    Measure GPU execution time with CUDA events.

    This provides a cleaner average execution time than
    using Python wall-clock timing.
    """
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(iterations):
        fn()

    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    average_ms = total_ms / iterations

    return total_ms, average_ms


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    print("============================================================")
    print("CUDA EXTENSION STATUS")
    print("============================================================")
    print(cuda_extension_status())

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset = BSDS300Dataset(use_patches=False)

    # --------------------------------------------------------
    # Generate ONE real DIP tensor
    # --------------------------------------------------------
    out, h, w, net, net_input, img_tensor = prepare_real_dip_output(dataset, subset_size=1)

    print("\n============================================================")
    print("COMMON DIP OUTPUT")
    print("============================================================")
    print("shape         :", out.shape)
    print("dtype         :", out.dtype)
    print("device        :", out.device)
    print("contiguous    :", out.is_contiguous())
    print("requires_grad :", out.requires_grad)
    print("grad_fn       :", out.grad_fn)

    # --------------------------------------------------------
    # Build FFT filters once
    # --------------------------------------------------------
    Dh_DFT, Dv_DFT = prepare_fft_filters(h, w, out.device)

    print("\nFFT filter dtype:")
    print("Dh_DFT:", Dh_DFT.dtype)
    print("Dv_DFT:", Dv_DFT.dtype)

    # --------------------------------------------------------
    # Trigger CUDA extension compilation BEFORE profiling
    # --------------------------------------------------------
    print("\nTriggering custom CUDA extension...")
    _ = run_cuda_gradient(out)
    torch.cuda.synchronize()
    print("CUDA extension ready.")

    # --------------------------------------------------------
    # Simple forward correctness test
    # --------------------------------------------------------
    Dh_fft, Dv_fft = run_fft_gradient(out, Dh_DFT, Dv_DFT)
    Dh_cuda, Dv_cuda = run_cuda_gradient(out)
    compare_gradients(Dh_fft, Dv_fft, Dh_cuda, Dv_cuda)

    # ========================================================
    # AUTOGRAD PROFILER - FFT
    # ========================================================
    Dh_fft_profiled, Dv_fft_profiled, fft_prof = profile_fft_gradient(
        out, Dh_DFT, Dv_DFT, warmup=WARMUP_ITERATIONS, iterations=PROFILE_ITERATIONS
    )

    # ========================================================
    # AUTOGRAD PROFILER - CUDA
    # ========================================================
    Dh_cuda_profiled, Dv_cuda_profiled, cuda_prof = profile_cuda_gradient(
        out, warmup=WARMUP_ITERATIONS, iterations=PROFILE_ITERATIONS
    )

    # --------------------------------------------------------
    # Verify profiling did not change results
    # --------------------------------------------------------
    torch.testing.assert_close(Dh_cuda_profiled, Dh_fft_profiled, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(Dv_cuda_profiled, Dv_fft_profiled, rtol=1e-4, atol=1e-5)

    # ========================================================
    # PRINT FFT PROFILER
    # ========================================================
    print("\n============================================================")
    print("FFT PROFILER")
    print("SORTED BY SELF CUDA TIME")
    print("============================================================")
    print(fft_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    # ========================================================
    # PRINT CUSTOM CUDA PROFILER
    # ========================================================
    print("\n============================================================")
    print("CUSTOM CUDA PROFILER")
    print("SORTED BY SELF CUDA TIME")
    print("============================================================")
    print(cuda_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    # ========================================================
    # CUDA EVENT TIMING
    # ========================================================
    print("\n============================================================")
    print("CUDA EVENT BENCHMARK")
    print("============================================================")

    fft_total_ms, fft_average_ms = benchmark_cuda_time(
        lambda: run_fft_gradient(out, Dh_DFT, Dv_DFT),
        warmup=WARMUP_ITERATIONS,
        iterations=BENCHMARK_ITERATIONS,
    )

    cuda_total_ms, cuda_average_ms = benchmark_cuda_time(
        lambda: run_cuda_gradient(out),
        warmup=WARMUP_ITERATIONS,
        iterations=BENCHMARK_ITERATIONS,
    )

    speedup = fft_average_ms / cuda_average_ms

    print(f"\nIterations: {BENCHMARK_ITERATIONS}")

    print("\nFFT:")
    print(f"  Total time   : {fft_total_ms:.6f} ms")
    print(f"  Average/call : {fft_average_ms:.6f} ms")
    print(f"  Average/call : {fft_average_ms * 1000:.3f} us")

    print("\nCustom CUDA:")
    print(f"  Total time   : {cuda_total_ms:.6f} ms")
    print(f"  Average/call : {cuda_average_ms:.6f} ms")
    print(f"  Average/call : {cuda_average_ms * 1000:.3f} us")

    print("\n============================================================")
    print(f"SPEEDUP: {speedup:.2f}x")
    print("============================================================")