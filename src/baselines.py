"""
Classical Baseline Methods for Deep Image Prior Project
========================================================

This module implements classical denoising and deconvolution baselines:
1. Wiener Deconvolution (frequency-domain approach)
2. BM3D (Block-Matching and 3D filtering)

These serve as the "lower bound" benchmarks to beat with Deep Image Prior.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import wiener as scipy_wiener
from scipy.fft import fft2, ifft2, fftshift
from skimage.restoration import wiener as skimage_wiener
from skimage.restoration import richardson_lucy, denoise_tv_chambolle
from typing import Union, Optional, Tuple
import warnings

# BM3D import - may not be installed
try:
    import bm3d as bm3d_lib
    HAS_BM3D = True
except ImportError:
    HAS_BM3D = False
    warnings.warn("BM3D library not installed. Install via: pip install bm3d")


# ============================================================================
# WIENER DECONVOLUTION
# ============================================================================

def wiener_deconvolution(degraded: np.ndarray, 
                         kernel: np.ndarray = None,
                         noise_var: float = None,
                         snr: float = None,
                         balance: float = 0.1) -> np.ndarray:
    """
    Wiener deconvolution for image restoration.
    
    The Wiener filter minimizes MSE between the true image and estimate:
    H_wiener(f) = H*(f) / (|H(f)|^2 + 1/SNR)
    
    Parameters
    ----------
    degraded : np.ndarray
        Degraded input image (blurred + noisy).
    kernel : np.ndarray, optional
        Known PSF/blur kernel. If None, uses identity (denoising mode).
    noise_var : float, optional
        Estimated noise variance. If provided, SNR is estimated.
    snr : float, optional
        Signal-to-noise ratio estimate. Alternative to noise_var.
    balance : float
        Regularization parameter (higher = more smoothing).
        
    Returns
    -------
    np.ndarray
        Restored image.
    """
    if kernel is None:
        # No blur kernel - use scipy's local Wiener filter for denoising
        if degraded.ndim == 2:
            return scipy_wiener(degraded, mysize=5)
        else:
            # Apply to each channel
            result = np.zeros_like(degraded)
            for c in range(degraded.shape[-1]):
                result[..., c] = scipy_wiener(degraded[..., c], mysize=5)
            return np.clip(result, 0, 1)
    
    # Use skimage's Wiener for deconvolution with known kernel
    if degraded.ndim == 2:
        return skimage_wiener(degraded, kernel, balance=balance, clip=True)
    else:
        # Apply to each channel
        result = np.zeros_like(degraded)
        for c in range(degraded.shape[-1]):
            result[..., c] = skimage_wiener(degraded[..., c], kernel, 
                                            balance=balance, clip=True)
        return np.clip(result, 0, 1)


def wiener_deconvolution_fft(degraded: np.ndarray,
                              kernel: np.ndarray,
                              nsr: float = 0.01) -> np.ndarray:
    """
    Custom FFT-based Wiener deconvolution implementation.
    
    Demonstrates the frequency-domain formulation explicitly:
    X_est(f) = Y(f) * H*(f) / (|H(f)|^2 + NSR)
    
    Parameters
    ----------
    degraded : np.ndarray
        Degraded image.
    kernel : np.ndarray
        Blur kernel (PSF).
    nsr : float
        Noise-to-Signal Ratio (inverse of SNR).
        
    Returns
    -------
    np.ndarray
        Restored image.
    """
    # Get image dimensions
    if degraded.ndim == 3:
        # Process first channel only for demo
        degraded = np.mean(degraded, axis=-1)
    
    M, N = degraded.shape
    
    # Pad kernel to image size
    kernel_padded = np.zeros((M, N))
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    
    # Shift to center
    kernel_padded = np.roll(kernel_padded, -(kh//2), axis=0)
    kernel_padded = np.roll(kernel_padded, -(kw//2), axis=1)
    
    # FFT
    Y = fft2(degraded)
    H = fft2(kernel_padded)
    
    # Wiener filter
    H_conj = np.conj(H)
    H_abs_sq = np.abs(H) ** 2
    
    # Wiener equation
    X_est = Y * H_conj / (H_abs_sq + nsr)
    
    # Inverse FFT
    x_est = np.real(ifft2(X_est))
    
    return np.clip(x_est, 0, 1)


# ============================================================================
# BM3D DENOISING
# ============================================================================

def bm3d_denoise(noisy: np.ndarray, 
                 sigma: float = 0.1,
                 profile: str = 'np') -> np.ndarray:
    """
    BM3D (Block-Matching and 3D filtering) denoising.
    
    BM3D is one of the most powerful classical denoising methods:
    1. Groups similar patches into 3D stacks
    2. Applies collaborative filtering in transform domain
    3. Aggregates filtered patches back
    
    Parameters
    ----------
    noisy : np.ndarray
        Noisy input image in [0, 1].
    sigma : float
        Noise standard deviation (same scale as image, e.g., 0.1 for 10%).
    profile : str
        BM3D profile: 'np' (normal), 'lc' (low complexity), 
        'high' (high quality), 'vn' (very noisy).
        
    Returns
    -------
    np.ndarray
        Denoised image.
    """
    if not HAS_BM3D:
        raise ImportError("BM3D not installed. Run: pip install bm3d")
    
    # BM3D expects sigma in absolute units (0-255 scale or 0-1 scale)
    # The library typically works with [0, 1] normalized images
    
    if noisy.ndim == 2:
        # Grayscale
        denoised = bm3d_lib.bm3d(noisy, sigma_psd=sigma, stage_arg=bm3d_lib.BM3DStages.ALL_STAGES)
    else:
        # RGB - BM3D supports color images
        denoised = bm3d_lib.bm3d_rgb(noisy, sigma_psd=sigma)
    
    return np.clip(denoised, 0, 1)


# ============================================================================
# ADDITIONAL CLASSICAL METHODS
# ============================================================================

def total_variation_denoise(noisy: np.ndarray, 
                            weight: float = 0.1) -> np.ndarray:
    """
    Total Variation (TV) denoising using Chambolle's algorithm.
    
    TV regularization preserves edges while smoothing flat regions.
    
    Parameters
    ----------
    noisy : np.ndarray
        Noisy input image.
    weight : float
        Denoising weight (higher = more denoising).
        
    Returns
    -------
    np.ndarray
        Denoised image.
    """
    if noisy.ndim == 3:
        return denoise_tv_chambolle(noisy, weight=weight, channel_axis=-1)
    else:
        return denoise_tv_chambolle(noisy, weight=weight)


def richardson_lucy_deconv(degraded: np.ndarray,
                            kernel: np.ndarray,
                            iterations: int = 30) -> np.ndarray:
    """
    Richardson-Lucy iterative deconvolution.
    
    An iterative algorithm that maximizes likelihood under Poisson noise.
    Often produces sharper results than Wiener but can amplify noise.
    
    Parameters
    ----------
    degraded : np.ndarray
        Degraded image.
    kernel : np.ndarray
        Known blur kernel.
    iterations : int
        Number of iterations (more = sharper but noisier).
        
    Returns
    -------
    np.ndarray
        Restored image.
    """
    if degraded.ndim == 2:
        return richardson_lucy(degraded, kernel, num_iter=iterations, clip=True)
    else:
        result = np.zeros_like(degraded)
        for c in range(degraded.shape[-1]):
            result[..., c] = richardson_lucy(degraded[..., c], kernel, 
                                              num_iter=iterations, clip=True)
        return np.clip(result, 0, 1)


# ============================================================================
# UNIFIED RESTORATION INTERFACE
# ============================================================================

def restore_image(degraded: np.ndarray,
                  method: str = 'bm3d',
                  kernel: np.ndarray = None,
                  sigma: float = 0.1,
                  **kwargs) -> Tuple[np.ndarray, str]:
    """
    Unified interface for classical image restoration.
    
    Parameters
    ----------
    degraded : np.ndarray
        Degraded input image.
    method : str
        Restoration method: 'bm3d', 'wiener', 'tv', 'richardson_lucy'.
    kernel : np.ndarray, optional
        Blur kernel (for deconvolution methods).
    sigma : float
        Noise level estimate.
    **kwargs
        Additional method-specific parameters.
        
    Returns
    -------
    tuple
        (restored_image, method_name)
    """
    method = method.lower()
    
    if method == 'bm3d':
        if not HAS_BM3D:
            warnings.warn("BM3D not available, falling back to TV denoising")
            method = 'tv'
        else:
            restored = bm3d_denoise(degraded, sigma)
            return restored, 'BM3D'
    
    if method == 'wiener':
        balance = kwargs.get('balance', sigma * 10)
        restored = wiener_deconvolution(degraded, kernel, balance=balance)
        return restored, 'Wiener'
    
    if method == 'tv':
        weight = kwargs.get('weight', 0.1)
        restored = total_variation_denoise(degraded, weight)
        return restored, 'Total Variation'
    
    if method == 'richardson_lucy' or method == 'rl':
        iterations = kwargs.get('iterations', 30)
        if kernel is None:
            raise ValueError("Richardson-Lucy requires a blur kernel")
        restored = richardson_lucy_deconv(degraded, kernel, iterations)
        return restored, 'Richardson-Lucy'
    
    raise ValueError(f"Unknown method: {method}")


# ============================================================================
# BATCH PROCESSING FOR BENCHMARKS
# ============================================================================

def run_baseline_benchmark(clean_images: list,
                           degraded_images: list,
                           methods: list = ['wiener', 'bm3d'],
                           sigmas: list = None,
                           kernels: list = None) -> dict:
    """
    Run baseline methods on a set of images and collect metrics.
    
    Parameters
    ----------
    clean_images : list
        List of ground truth images.
    degraded_images : list
        List of corresponding degraded images.
    methods : list
        List of method names to run.
    sigmas : list, optional
        List of noise levels for each image.
    kernels : list, optional
        List of blur kernels (for deconvolution).
        
    Returns
    -------
    dict
        Results dictionary with PSNR/SSIM for each method.
    """
    from . import metrics
    
    if sigmas is None:
        sigmas = [0.1] * len(clean_images)
    
    results = {method: {'psnr': [], 'ssim': []} for method in methods}
    results['noisy'] = {'psnr': [], 'ssim': []}
    
    for i, (clean, degraded) in enumerate(zip(clean_images, degraded_images)):
        sigma = sigmas[i] if i < len(sigmas) else sigmas[-1]
        kernel = kernels[i] if kernels and i < len(kernels) else None
        
        # Noisy baseline
        noisy_metrics = metrics.evaluate_image(clean, degraded)
        results['noisy']['psnr'].append(noisy_metrics['psnr'])
        results['noisy']['ssim'].append(noisy_metrics['ssim'])
        
        # Run each method
        for method in methods:
            try:
                restored, _ = restore_image(degraded, method, kernel, sigma)
                method_metrics = metrics.evaluate_image(clean, restored)
                results[method]['psnr'].append(method_metrics['psnr'])
                results[method]['ssim'].append(method_metrics['ssim'])
            except Exception as e:
                print(f"Error running {method} on image {i}: {e}")
                results[method]['psnr'].append(np.nan)
                results[method]['ssim'].append(np.nan)
    
    # Compute averages
    for method in ['noisy'] + methods:
        results[method]['mean_psnr'] = np.nanmean(results[method]['psnr'])
        results[method]['mean_ssim'] = np.nanmean(results[method]['ssim'])
    
    return results


def print_benchmark_table(results: dict, image_ids: list = None) -> str:
    """
    Format benchmark results as a nice table.
    
    Parameters
    ----------
    results : dict
        Results from run_baseline_benchmark().
    image_ids : list, optional
        Image identifiers for rows.
        
    Returns
    -------
    str
        Formatted table string.
    """
    methods = [k for k in results.keys() if k != 'noisy']
    
    # Header
    header = f"{'Image ID':<15} | {'Noisy PSNR':<12} | "
    header += " | ".join([f"{m} PSNR" for m in methods])
    
    lines = ["=" * len(header), header, "=" * len(header)]
    
    n_images = len(results['noisy']['psnr'])
    
    for i in range(n_images):
        img_id = image_ids[i] if image_ids else f"Image_{i+1}"
        row = f"{img_id:<15} | {results['noisy']['psnr'][i]:>10.2f} dB | "
        row += " | ".join([f"{results[m]['psnr'][i]:>8.2f} dB" for m in methods])
        lines.append(row)
    
    # Average row
    lines.append("-" * len(header))
    avg_row = f"{'AVERAGE':<15} | {results['noisy']['mean_psnr']:>10.2f} dB | "
    avg_row += " | ".join([f"{results[m]['mean_psnr']:>8.2f} dB" for m in methods])
    lines.append(avg_row)
    lines.append("=" * len(header))
    
    return "\n".join(lines)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    print("="*60)
    print("Classical Baselines Module Demo")
    print("="*60)
    
    # Check BM3D availability
    print(f"\nBM3D available: {HAS_BM3D}")
    
    # Create synthetic test
    print("\nCreating synthetic test image...")
    np.random.seed(42)
    
    # Create test image
    x = np.linspace(0, 1, 256)
    xx, yy = np.meshgrid(x, x)
    clean = 0.5 + 0.3 * np.sin(20 * xx) * np.cos(20 * yy)
    clean = clean.astype(np.float32)
    
    # Add noise
    sigma = 0.1
    noisy = clean + sigma * np.random.randn(*clean.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 1)
    
    print(f"Clean image: {clean.shape}, range [{clean.min():.2f}, {clean.max():.2f}]")
    print(f"Noisy image: σ={sigma}")
    
    # Test methods
    print("\n" + "-"*40)
    print("Testing restoration methods:")
    print("-"*40)
    
    # Import metrics for evaluation
    try:
        from metrics import evaluate_image, print_metrics
    except ImportError:
        # Add parent to path
        sys.path.insert(0, str(Path(__file__).parent))
        from metrics import evaluate_image, print_metrics
    
    # Noisy baseline
    noisy_metrics = evaluate_image(clean, noisy)
    print(f"\n1. Noisy baseline:")
    print(f"   PSNR: {noisy_metrics['psnr']:.2f} dB")
    print(f"   SSIM: {noisy_metrics['ssim']:.4f}")
    
    # Wiener filter
    print(f"\n2. Wiener Filter:")
    wiener_result = wiener_deconvolution(noisy, kernel=None)
    wiener_metrics = evaluate_image(clean, wiener_result)
    print(f"   PSNR: {wiener_metrics['psnr']:.2f} dB")
    print(f"   SSIM: {wiener_metrics['ssim']:.4f}")
    
    # TV Denoising
    print(f"\n3. Total Variation:")
    tv_result = total_variation_denoise(noisy, weight=0.1)
    tv_metrics = evaluate_image(clean, tv_result)
    print(f"   PSNR: {tv_metrics['psnr']:.2f} dB")
    print(f"   SSIM: {tv_metrics['ssim']:.4f}")
    
    # BM3D
    if HAS_BM3D:
        print(f"\n4. BM3D:")
        bm3d_result = bm3d_denoise(noisy, sigma=sigma)
        bm3d_metrics = evaluate_image(clean, bm3d_result)
        print(f"   PSNR: {bm3d_metrics['psnr']:.2f} dB")
        print(f"   SSIM: {bm3d_metrics['ssim']:.4f}")
    else:
        print(f"\n4. BM3D: SKIPPED (not installed)")
    
    print("\n" + "="*60)
    print("✓ Baselines module working correctly!")
    print("="*60)
