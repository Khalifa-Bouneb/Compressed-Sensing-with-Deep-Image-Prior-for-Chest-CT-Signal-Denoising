"""
Example usage of DIP for compressed sensing reconstruction.
This script demonstrates how to use the library on a sample image.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    UnderSamplingMask,
    CompressedSensingOperator,
    add_noise,
    normalize_image,
    plot_results,
    plot_convergence,
    calculate_psnr,
    calculate_ssim
)
from src.reconstruction import reconstruct_with_dip


def load_sample_image(image_path=None, size=(256, 256)):
    """
    Load a sample image for testing.
    If no path is provided, creates a synthetic test image.
    """
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('L')
        img = img.resize(size)
        img = np.array(img).astype(np.float32) / 255.0
    else:
        # Create a synthetic test image (shepp-logan phantom-like)
        print("No image path provided. Creating synthetic test image...")
        try:
            from skimage import data
            img = data.shepp_logan_phantom()
            img = np.array(Image.fromarray(img).resize(size))
            img = img.astype(np.float32)
        except ImportError:
            print("Warning: scikit-image not available. Creating random test image...")
            img = np.random.rand(*size).astype(np.float32)
    
    return normalize_image(img)


def main():
    """Main example demonstrating DIP reconstruction."""
    
    print("=" * 60)
    print("Compressed Sensing with Deep Image Prior")
    print("Example: CT Image Reconstruction")
    print("=" * 60)
    
    # Parameters
    image_size = (256, 256)
    sampling_rate = 0.3  # Keep 30% of measurements
    mask_type = 'random'  # or 'cartesian', 'radial'
    noise_level = 0.01
    num_iter = 2000  # Reduced for quick demo
    lr = 0.01
    
    print(f"\nParameters:")
    print(f"  Image size: {image_size}")
    print(f"  Sampling rate: {sampling_rate * 100}%")
    print(f"  Mask type: {mask_type}")
    print(f"  Noise level: {noise_level}")
    print(f"  Iterations: {num_iter}")
    print(f"  Learning rate: {lr}")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Load or create test image
    print("\n1. Loading image...")
    original_image = load_sample_image(size=image_size)
    print(f"   Image shape: {original_image.shape}")
    
    # Create undersampling mask
    print(f"\n2. Creating {mask_type} undersampling mask...")
    mask = UnderSamplingMask(
        shape=image_size,
        sampling_rate=sampling_rate,
        mask_type=mask_type
    )
    print(f"   Mask keeps {np.sum(mask.mask) / mask.mask.size * 100:.1f}% of pixels")
    
    # Apply mask to get measurements
    print("\n3. Generating undersampled measurements...")
    cs_operator = CompressedSensingOperator(mask)
    measurements = cs_operator.forward(original_image)
    
    # Add noise
    measurements_noisy = add_noise(measurements, noise_level)
    print(f"   Added Gaussian noise (std={noise_level})")
    
    # Reconstruct using DIP
    print(f"\n4. Starting DIP reconstruction...")
    print(f"   This may take a few minutes...")
    
    reconstructed, reconstructor = reconstruct_with_dip(
        measurements=measurements_noisy,
        mask=mask,
        image_shape=image_size,
        ground_truth=original_image,
        input_depth=32,
        lr=lr,
        num_iter=num_iter,
        device=device,
        log_interval=100
    )
    
    # Calculate final metrics
    print("\n5. Evaluation:")
    reconstructed_np = reconstructed.detach().cpu().squeeze().numpy()
    psnr = calculate_psnr(original_image, reconstructed_np)
    ssim = calculate_ssim(original_image, reconstructed_np)
    print(f"   Final PSNR: {psnr:.2f} dB")
    print(f"   Final SSIM: {ssim:.4f}")
    
    # Plot results
    print("\n6. Generating visualizations...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot reconstruction results
    fig1 = plot_results(
        original_image,
        measurements_noisy,
        reconstructed_np,
        save_path=os.path.join(output_dir, 'reconstruction_results.png')
    )
    print(f"   Saved: {output_dir}/reconstruction_results.png")
    
    # Plot convergence
    history = reconstructor.get_metrics_history()
    fig2 = plot_convergence(
        history['losses'],
        history['psnrs'],
        history['ssims'],
        save_path=os.path.join(output_dir, 'convergence.png')
    )
    print(f"   Saved: {output_dir}/convergence.png")
    
    # Plot mask
    plt.figure(figsize=(6, 6))
    plt.imshow(mask.mask, cmap='gray')
    plt.title(f'{mask_type.capitalize()} Sampling Mask ({sampling_rate*100:.0f}%)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'sampling_mask.png'), dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir}/sampling_mask.png")
    
    print("\n" + "=" * 60)
    print("Reconstruction complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
