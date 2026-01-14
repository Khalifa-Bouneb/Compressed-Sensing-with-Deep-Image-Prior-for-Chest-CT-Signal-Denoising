"""Visualization utilities for displaying results."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_results(original, noisy, reconstructed, save_path=None):
    """
    Plot original, noisy, and reconstructed images side by side.
    
    Args:
        original: Original clean image
        noisy: Noisy/undersampled image
        reconstructed: Reconstructed image
        save_path: Path to save the figure (optional)
    """
    # Convert tensors to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().squeeze().numpy()
    if isinstance(noisy, torch.Tensor):
        noisy = noisy.detach().cpu().squeeze().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title('Noisy/Undersampled Image')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed, cmap='gray')
    axes[2].set_title('Reconstructed Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_convergence(losses, psnrs=None, ssims=None, save_path=None):
    """
    Plot convergence curves during optimization.
    
    Args:
        losses: List of loss values
        psnrs: List of PSNR values (optional)
        ssims: List of SSIM values (optional)
        save_path: Path to save the figure (optional)
    """
    num_plots = 1 + (psnrs is not None) + (ssims is not None)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot loss
    axes[0].plot(losses)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Convergence')
    axes[0].grid(True)
    
    plot_idx = 1
    
    # Plot PSNR if available
    if psnrs is not None:
        axes[plot_idx].plot(psnrs)
        axes[plot_idx].set_xlabel('Iteration')
        axes[plot_idx].set_ylabel('PSNR (dB)')
        axes[plot_idx].set_title('PSNR over Iterations')
        axes[plot_idx].grid(True)
        plot_idx += 1
    
    # Plot SSIM if available
    if ssims is not None:
        axes[plot_idx].plot(ssims)
        axes[plot_idx].set_xlabel('Iteration')
        axes[plot_idx].set_ylabel('SSIM')
        axes[plot_idx].set_title('SSIM over Iterations')
        axes[plot_idx].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison_grid(images_dict, save_path=None, cmap='gray'):
    """
    Plot a grid of images for comparison.
    
    Args:
        images_dict: Dictionary mapping titles to images
        save_path: Path to save the figure (optional)
        cmap: Colormap to use
    """
    num_images = len(images_dict)
    cols = min(4, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, (title, img) in enumerate(images_dict.items()):
        row = idx // cols
        col = idx % cols
        
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().squeeze().numpy()
        
        axes[row][col].imshow(img, cmap=cmap)
        axes[row][col].set_title(title)
        axes[row][col].axis('off')
    
    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_image(image, save_path, cmap='gray'):
    """
    Save a single image to file.
    
    Args:
        image: Image to save
        save_path: Path to save the image
        cmap: Colormap to use
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().squeeze().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
