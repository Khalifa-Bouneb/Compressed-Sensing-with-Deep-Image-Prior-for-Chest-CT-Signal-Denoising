"""
Image Quality Metrics for Deep Image Prior Project
===================================================

This module implements evaluation metrics for comparing restored images
against ground truth:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- DSSIM (Gradient-based Structural Similarity using Sobel)

All metrics assume images are in range [0, 1] or [0, 255].
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import ndimage


def psnr(img_true: np.ndarray, img_test: np.ndarray, data_range: float = None) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Higher PSNR indicates better quality (typical values: 20-40 dB).
    
    Parameters
    ----------
    img_true : np.ndarray
        Ground truth image.
    img_test : np.ndarray
        Restored/noisy image to evaluate.
    data_range : float, optional
        The data range of the input image. If None, it's determined from
        the image dtype (1.0 for float, 255 for uint8).
        
    Returns
    -------
    float
        PSNR value in dB.
    """
    if data_range is None:
        if img_true.dtype == np.uint8:
            data_range = 255.0
        else:
            data_range = 1.0
            
    return peak_signal_noise_ratio(img_true, img_test, data_range=data_range)


def ssim(img_true: np.ndarray, img_test: np.ndarray, data_range: float = None,
         multichannel: bool = None) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    
    SSIM measures structural information preservation, considering luminance,
    contrast, and structure. Range: [-1, 1], where 1 means perfect similarity.
    
    Parameters
    ----------
    img_true : np.ndarray
        Ground truth image.
    img_test : np.ndarray
        Restored/noisy image to evaluate.
    data_range : float, optional
        The data range of the input image.
    multichannel : bool, optional
        If True, treat the last dimension as channels.
        Auto-detected if None.
        
    Returns
    -------
    float
        SSIM value.
    """
    if data_range is None:
        if img_true.dtype == np.uint8:
            data_range = 255.0
        else:
            data_range = 1.0
    
    # Auto-detect if image is multichannel (RGB)
    if multichannel is None:
        multichannel = (img_true.ndim == 3 and img_true.shape[-1] in [3, 4])
    
    if multichannel:
        return structural_similarity(img_true, img_test, data_range=data_range,
                                     channel_axis=-1)
    else:
        return structural_similarity(img_true, img_test, data_range=data_range)


def sobel_gradient(image: np.ndarray) -> tuple:
    """
    Compute Sobel gradients for an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB).
        
    Returns
    -------
    tuple
        (gradient_x, gradient_y) - horizontal and vertical gradients.
    """
    if image.ndim == 3:
        # Convert to grayscale for gradient computation
        image_gray = np.mean(image, axis=-1)
    else:
        image_gray = image
        
    # Sobel kernels
    sobel_x = ndimage.sobel(image_gray, axis=1)
    sobel_y = ndimage.sobel(image_gray, axis=0)
    
    return sobel_x, sobel_y


def dssim(img_true: np.ndarray, img_test: np.ndarray) -> float:
    """
    Calculate Gradient-based Structural Dissimilarity (DSSIM).
    
    Uses Sobel gradients to compare edge structure between images.
    This is useful for detecting edge preservation quality.
    
    DSSIM = (1 - SSIM) / 2, applied to gradient images.
    
    Parameters
    ----------
    img_true : np.ndarray
        Ground truth image.
    img_test : np.ndarray
        Restored/noisy image to evaluate.
        
    Returns
    -------
    float
        DSSIM value. Lower is better (0 = perfect).
    """
    # Compute gradients
    gx_true, gy_true = sobel_gradient(img_true)
    gx_test, gy_test = sobel_gradient(img_test)
    
    # Gradient magnitude
    grad_true = np.sqrt(gx_true**2 + gy_true**2)
    grad_test = np.sqrt(gx_test**2 + gy_test**2)
    
    # Normalize for SSIM computation
    grad_true_norm = (grad_true - grad_true.min()) / (grad_true.max() - grad_true.min() + 1e-8)
    grad_test_norm = (grad_test - grad_test.min()) / (grad_test.max() - grad_test.min() + 1e-8)
    
    # Compute SSIM on gradients
    ssim_val = structural_similarity(grad_true_norm, grad_test_norm, data_range=1.0)
    
    # Convert to dissimilarity
    return (1 - ssim_val) / 2


def mse(img_true: np.ndarray, img_test: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Parameters
    ----------
    img_true : np.ndarray
        Ground truth image.
    img_test : np.ndarray
        Restored/noisy image to evaluate.
        
    Returns
    -------
    float
        MSE value. Lower is better (0 = perfect).
    """
    return np.mean((img_true.astype(float) - img_test.astype(float)) ** 2)


def evaluate_image(img_true: np.ndarray, img_test: np.ndarray, 
                   data_range: float = None) -> dict:
    """
    Compute all metrics for an image pair.
    
    Parameters
    ----------
    img_true : np.ndarray
        Ground truth image.
    img_test : np.ndarray
        Restored/noisy image to evaluate.
    data_range : float, optional
        The data range of the input image.
        
    Returns
    -------
    dict
        Dictionary containing all metrics: 
        {'psnr': float, 'ssim': float, 'dssim': float, 'mse': float}
    """
    return {
        'psnr': psnr(img_true, img_test, data_range),
        'ssim': ssim(img_true, img_test, data_range),
        'dssim': dssim(img_true, img_test),
        'mse': mse(img_true, img_test)
    }


def print_metrics(metrics: dict, name: str = "Image") -> None:
    """
    Pretty print evaluation metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics from evaluate_image().
    name : str
        Name/ID of the image for display.
    """
    print(f"\n{'='*50}")
    print(f"Metrics for: {name}")
    print(f"{'='*50}")
    print(f"  PSNR:  {metrics['psnr']:.2f} dB")
    print(f"  SSIM:  {metrics['ssim']:.4f}")
    print(f"  DSSIM: {metrics['dssim']:.4f}")
    print(f"  MSE:   {metrics['mse']:.6f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Quick test with synthetic images
    print("Testing metrics module...")
    
    # Create test images
    np.random.seed(42)
    img_clean = np.random.rand(256, 256).astype(np.float32)
    img_noisy = img_clean + 0.1 * np.random.randn(256, 256).astype(np.float32)
    img_noisy = np.clip(img_noisy, 0, 1)
    
    # Evaluate
    results = evaluate_image(img_clean, img_noisy, data_range=1.0)
    print_metrics(results, "Synthetic Test (σ=0.1)")
    
    print("✓ Metrics module working correctly!")
