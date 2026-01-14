"""Evaluation metrics for image quality assessment."""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: First image (reference)
        img2: Second image (reconstructed)
        data_range: Dynamic range of the images
        
    Returns:
        PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Remove batch and channel dimensions if present
    if img1.ndim == 4:
        img1 = img1[0, 0]
    elif img1.ndim == 3:
        img1 = img1[0]
    
    if img2.ndim == 4:
        img2 = img2[0, 0]
    elif img2.ndim == 3:
        img2 = img2[0]
    
    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        img1: First image (reference)
        img2: Second image (reconstructed)
        data_range: Dynamic range of the images
        
    Returns:
        SSIM value (between -1 and 1, higher is better)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Remove batch and channel dimensions if present
    if img1.ndim == 4:
        img1 = img1[0, 0]
    elif img1.ndim == 3:
        img1 = img1[0]
    
    if img2.ndim == 4:
        img2 = img2[0, 0]
    elif img2.ndim == 3:
        img2 = img2[0]
    
    return structural_similarity(img1, img2, data_range=data_range)


def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        MSE value
    """
    if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
        return torch.mean((img1 - img2) ** 2).item()
    else:
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        return np.mean((img1 - img2) ** 2)


def calculate_nmse(img1, img2):
    """
    Calculate Normalized Mean Squared Error (NMSE).
    
    Args:
        img1: First image (reference)
        img2: Second image (reconstructed)
        
    Returns:
        NMSE value
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    norm = np.mean(img1 ** 2)
    return mse / (norm + 1e-8)
