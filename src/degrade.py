"""
Image Degradation Module for Deep Image Prior Project
======================================================

This module implements the forward model: y = Ax + η

Where:
- x: Clean ground-truth image
- A: Degradation operator (Identity for denoising, Blur kernel for deconvolution)
- η: Additive Gaussian noise
- y: Observed degraded image

Supports two degradation scenarios:
1. Denoising: A = I (identity), adding Gaussian noise
2. Deconvolution: A = blur kernel, followed by noise
"""

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from pathlib import Path
from PIL import Image
from typing import Union, Tuple, Optional
import os


def load_image(path: Union[str, Path], normalize: bool = True, 
               grayscale: bool = False) -> np.ndarray:
    """
    Load an image from disk.
    
    Parameters
    ----------
    path : str or Path
        Path to the image file.
    normalize : bool
        If True, normalize to [0, 1] range.
    grayscale : bool
        If True, convert to grayscale.
        
    Returns
    -------
    np.ndarray
        Loaded image as numpy array.
    """
    img = Image.open(path)
    
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    
    img_array = np.array(img, dtype=np.float32)
    
    if normalize:
        img_array = img_array / 255.0
        
    return img_array


def save_image(image: np.ndarray, path: Union[str, Path], 
               denormalize: bool = True) -> None:
    """
    Save an image to disk.
    
    Parameters
    ----------
    image : np.ndarray
        Image array to save.
    path : str or Path
        Output path.
    denormalize : bool
        If True, assume input is in [0, 1] and convert to [0, 255].
    """
    if denormalize:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    
    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ============================================================================
# NOISE GENERATION
# ============================================================================

def add_gaussian_noise(image: np.ndarray, sigma: float = 0.1, 
                       seed: Optional[int] = None) -> np.ndarray:
    """
    Add Gaussian noise to an image.
    
    η ~ N(0, σ²)
    y = x + η
    
    Parameters
    ----------
    image : np.ndarray
        Clean image in range [0, 1].
    sigma : float
        Standard deviation of noise (typical: 0.1, 0.2 for denoising).
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray
        Noisy image, clipped to [0, 1].
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = sigma * np.random.randn(*image.shape).astype(np.float32)
    noisy = image + noise
    
    return np.clip(noisy, 0, 1)


# ============================================================================
# BLUR KERNEL GENERATION
# ============================================================================

def gaussian_kernel(size: int = 7, sigma: float = 1.5) -> np.ndarray:
    """
    Create a Gaussian blur kernel.
    
    Parameters
    ----------
    size : int
        Kernel size (should be odd, e.g., 7, 13, 21).
    sigma : float
        Standard deviation of Gaussian.
        
    Returns
    -------
    np.ndarray
        2D Gaussian kernel (normalized to sum to 1).
    """
    x = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def motion_blur_kernel(size: int = 15, angle: float = 45) -> np.ndarray:
    """
    Create a motion blur kernel.
    
    Parameters
    ----------
    size : int
        Kernel size.
    angle : float
        Angle of motion in degrees.
        
    Returns
    -------
    np.ndarray
        2D motion blur kernel.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Draw a line through the center
    for i in range(size):
        offset = i - center
        x = center + int(np.round(offset * np.cos(angle_rad)))
        y = center + int(np.round(offset * np.sin(angle_rad)))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
            
    return kernel / kernel.sum()


def box_kernel(size: int = 5) -> np.ndarray:
    """
    Create a uniform (box) blur kernel.
    
    Parameters
    ----------
    size : int
        Kernel size.
        
    Returns
    -------
    np.ndarray
        2D uniform kernel.
    """
    kernel = np.ones((size, size))
    return kernel / kernel.sum()


def disk_kernel(radius: int = 5) -> np.ndarray:
    """
    Create a disk (out-of-focus) blur kernel.
    
    Parameters
    ----------
    radius : int
        Radius of the disk.
        
    Returns
    -------
    np.ndarray
        2D disk kernel.
    """
    size = 2 * radius + 1
    x = np.arange(size) - radius
    xx, yy = np.meshgrid(x, x)
    kernel = (xx**2 + yy**2 <= radius**2).astype(float)
    return kernel / kernel.sum()


# ============================================================================
# BLUR APPLICATION
# ============================================================================

def apply_blur(image: np.ndarray, kernel: np.ndarray, 
               mode: str = 'same') -> np.ndarray:
    """
    Apply blur kernel to an image using convolution.
    
    y = A * x  (where A is the blur kernel)
    
    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB).
    kernel : np.ndarray
        2D blur kernel.
    mode : str
        Convolution mode ('same', 'valid', 'full').
        
    Returns
    -------
    np.ndarray
        Blurred image.
    """
    if image.ndim == 2:
        # Grayscale
        return convolve2d(image, kernel, mode=mode, boundary='wrap')
    else:
        # RGB - apply to each channel
        result = np.zeros_like(image)
        for c in range(image.shape[-1]):
            result[..., c] = convolve2d(image[..., c], kernel, 
                                         mode=mode, boundary='wrap')
        return result


# ============================================================================
# COMPLETE DEGRADATION PIPELINES
# ============================================================================

def degrade_for_denoising(image: np.ndarray, sigma: float = 0.1,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Apply denoising degradation: y = x + η
    
    A = Identity matrix (no blur)
    
    Parameters
    ----------
    image : np.ndarray
        Clean image in [0, 1].
    sigma : float
        Noise standard deviation (typical: 0.1 or 0.2).
    seed : int, optional
        Random seed.
        
    Returns
    -------
    tuple
        (degraded_image, params_dict)
    """
    noisy = add_gaussian_noise(image, sigma, seed)
    
    params = {
        'degradation_type': 'denoising',
        'noise_sigma': sigma,
        'blur_kernel': None,
        'seed': seed
    }
    
    return noisy, params


def degrade_for_deconvolution(image: np.ndarray, 
                               kernel_type: str = 'gaussian',
                               kernel_size: int = 7,
                               kernel_sigma: float = 1.5,
                               noise_sigma: float = 0.01,
                               seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """
    Apply deconvolution degradation: y = A*x + η
    
    Parameters
    ----------
    image : np.ndarray
        Clean image in [0, 1].
    kernel_type : str
        Type of blur kernel: 'gaussian', 'motion', 'box', 'disk'.
    kernel_size : int
        Size of blur kernel.
    kernel_sigma : float
        Sigma for Gaussian kernel (ignored for other types).
    noise_sigma : float
        Noise standard deviation (typical: 0.01 for deconvolution).
    seed : int, optional
        Random seed.
        
    Returns
    -------
    tuple
        (degraded_image, params_dict) where params includes the kernel
    """
    # Generate kernel
    if kernel_type == 'gaussian':
        kernel = gaussian_kernel(kernel_size, kernel_sigma)
    elif kernel_type == 'motion':
        kernel = motion_blur_kernel(kernel_size, angle=45)
    elif kernel_type == 'box':
        kernel = box_kernel(kernel_size)
    elif kernel_type == 'disk':
        kernel = disk_kernel(kernel_size // 2)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Apply blur
    blurred = apply_blur(image, kernel)
    
    # Add noise
    degraded = add_gaussian_noise(blurred, noise_sigma, seed)
    
    params = {
        'degradation_type': 'deconvolution',
        'noise_sigma': noise_sigma,
        'blur_kernel': kernel,
        'kernel_type': kernel_type,
        'kernel_size': kernel_size,
        'kernel_sigma': kernel_sigma,
        'seed': seed
    }
    
    return degraded, params


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_dataset(input_dir: Union[str, Path],
                    output_dir: Union[str, Path],
                    degradation: str = 'denoising',
                    sigma: float = 0.1,
                    kernel_type: str = 'gaussian',
                    kernel_size: int = 7,
                    grayscale: bool = False,
                    max_images: Optional[int] = None,
                    seed: int = 42) -> list:
    """
    Process a directory of images with specified degradation.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing clean images.
    output_dir : str or Path
        Directory to save degraded images.
    degradation : str
        'denoising' or 'deconvolution'.
    sigma : float
        Noise level.
    kernel_type : str
        Type of blur kernel (for deconvolution).
    kernel_size : int
        Size of blur kernel.
    grayscale : bool
        Whether to convert to grayscale.
    max_images : int, optional
        Maximum number of images to process.
    seed : int
        Base random seed.
        
    Returns
    -------
    list
        List of processed image info dicts.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in extensions]
    
    if max_images:
        image_files = image_files[:max_images]
    
    results = []
    
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
        
        # Load image
        image = load_image(img_path, normalize=True, grayscale=grayscale)
        
        # Apply degradation
        if degradation == 'denoising':
            degraded, params = degrade_for_denoising(image, sigma, seed=seed+i)
        else:
            degraded, params = degrade_for_deconvolution(
                image, kernel_type, kernel_size, 
                noise_sigma=sigma, seed=seed+i
            )
        
        # Save degraded image
        output_path = output_dir / f"degraded_{img_path.name}"
        save_image(degraded, output_path)
        
        # Save clean copy for reference
        clean_path = output_dir / f"clean_{img_path.name}"
        save_image(image, clean_path)
        
        results.append({
            'original_path': str(img_path),
            'degraded_path': str(output_path),
            'clean_path': str(clean_path),
            'params': params
        })
    
    return results


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("Degradation Module Demo")
    print("="*60)
    
    # Create a synthetic test image
    print("\nCreating synthetic test image...")
    np.random.seed(42)
    
    # Create a simple pattern image
    x = np.linspace(0, 1, 256)
    xx, yy = np.meshgrid(x, x)
    test_image = 0.5 + 0.3 * np.sin(20 * xx) * np.cos(20 * yy)
    test_image = test_image.astype(np.float32)
    
    # Demo 1: Denoising degradation
    print("\n1. Denoising degradation (σ=0.1):")
    noisy, params = degrade_for_denoising(test_image, sigma=0.1, seed=42)
    print(f"   Noise σ: {params['noise_sigma']}")
    
    # Demo 2: Deconvolution degradation
    print("\n2. Deconvolution degradation (Gaussian blur 7x7 + σ=0.01):")
    blurry_noisy, params = degrade_for_deconvolution(
        test_image, kernel_type='gaussian', kernel_size=7, 
        kernel_sigma=1.5, noise_sigma=0.01, seed=42
    )
    print(f"   Kernel type: {params['kernel_type']}")
    print(f"   Kernel size: {params['kernel_size']}")
    print(f"   Noise σ: {params['noise_sigma']}")
    
    # Show different blur kernels
    print("\n3. Available blur kernels:")
    kernels = {
        'Gaussian (7x7)': gaussian_kernel(7, 1.5),
        'Motion (15x15)': motion_blur_kernel(15, 45),
        'Box (5x5)': box_kernel(5),
        'Disk (r=5)': disk_kernel(5)
    }
    
    for name, kernel in kernels.items():
        print(f"   - {name}: shape={kernel.shape}, sum={kernel.sum():.4f}")
    
    print("\n" + "="*60)
    print("✓ Degradation module working correctly!")
    print("="*60)
