"""
Example for working with actual CT image files.
Demonstrates loading DICOM or PNG CT images and processing them.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

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


def load_ct_image(image_path, target_size=(256, 256)):
    """
    Load CT image from various formats.
    Supports: PNG, JPEG, TIFF, NPY, NPZ
    For DICOM, install pydicom: pip install pydicom
    """
    from PIL import Image
    
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    
    if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
        # Load image file
        img = Image.open(image_path).convert('L')
        img = img.resize(target_size)
        img = np.array(img).astype(np.float32)
        # Normalize to [0, 1]
        img = img / 255.0
    
    elif ext in ['.npy']:
        # Load numpy array
        img = np.load(image_path)
        img = np.array(Image.fromarray(img).resize(target_size))
        img = normalize_image(img)
    
    elif ext in ['.npz']:
        # Load compressed numpy array
        data = np.load(image_path)
        img = data[data.files[0]]  # Get first array
        img = np.array(Image.fromarray(img).resize(target_size))
        img = normalize_image(img)
    
    elif ext in ['.dcm', '.dicom']:
        try:
            import pydicom
            # Load DICOM file
            dcm = pydicom.dcmread(image_path)
            img = dcm.pixel_array.astype(np.float32)
            img = np.array(Image.fromarray(img).resize(target_size))
            img = normalize_image(img)
        except ImportError:
            raise ImportError("pydicom not installed. Install with: pip install pydicom")
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return img


def process_ct_image(
    image_path,
    output_dir,
    sampling_rate=0.3,
    mask_type='random',
    noise_level=0.01,
    num_iter=5000,
    lr=0.01,
    device='cuda'
):
    """Process a single CT image."""
    
    print(f"\nProcessing: {os.path.basename(image_path)}")
    print("-" * 60)
    
    # Load image
    print("1. Loading CT image...")
    ct_image = load_ct_image(image_path)
    print(f"   Image shape: {ct_image.shape}")
    print(f"   Value range: [{ct_image.min():.3f}, {ct_image.max():.3f}]")
    
    # Create mask
    print(f"2. Creating {mask_type} mask (sampling rate: {sampling_rate})...")
    mask = UnderSamplingMask(
        shape=ct_image.shape,
        sampling_rate=sampling_rate,
        mask_type=mask_type
    )
    
    # Generate measurements
    print("3. Generating undersampled measurements...")
    cs_operator = CompressedSensingOperator(mask)
    measurements = cs_operator.forward(ct_image)
    measurements_noisy = add_noise(measurements, noise_level)
    
    # Reconstruct
    print(f"4. Reconstructing with DIP ({num_iter} iterations)...")
    reconstructed, reconstructor = reconstruct_with_dip(
        measurements=measurements_noisy,
        mask=mask,
        image_shape=ct_image.shape,
        ground_truth=ct_image,
        input_depth=32,
        lr=lr,
        num_iter=num_iter,
        device=device,
        log_interval=100
    )
    
    # Evaluate
    print("5. Evaluating results...")
    reconstructed_np = reconstructed.detach().cpu().squeeze().numpy()
    psnr = calculate_psnr(ct_image, reconstructed_np)
    ssim = calculate_ssim(ct_image, reconstructed_np)
    print(f"   PSNR: {psnr:.2f} dB")
    print(f"   SSIM: {ssim:.4f}")
    
    # Save results
    print("6. Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save comparison
    plot_results(
        ct_image,
        measurements_noisy,
        reconstructed_np,
        save_path=os.path.join(output_dir, f'{base_name}_comparison.png')
    )
    
    # Save convergence
    history = reconstructor.get_metrics_history()
    plot_convergence(
        history['losses'],
        history['psnrs'],
        history['ssims'],
        save_path=os.path.join(output_dir, f'{base_name}_convergence.png')
    )
    
    # Save reconstructed as numpy
    np.save(os.path.join(output_dir, f'{base_name}_reconstructed.npy'), reconstructed_np)
    
    print(f"   Results saved to: {output_dir}")
    print("-" * 60)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'reconstructed': reconstructed_np
    }


def main():
    parser = argparse.ArgumentParser(
        description='CT Image Reconstruction with Deep Image Prior'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to CT image file (PNG, JPEG, DICOM, NPY, NPZ)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=0.3,
        help='Sampling rate (0-1, default: 0.3)'
    )
    parser.add_argument(
        '--mask-type',
        type=str,
        default='random',
        choices=['random', 'cartesian', 'radial'],
        help='Type of undersampling mask (default: random)'
    )
    parser.add_argument(
        '--noise-level',
        type=float,
        default=0.01,
        help='Noise standard deviation (default: 0.01)'
    )
    parser.add_argument(
        '--num-iter',
        type=int,
        default=5000,
        help='Number of optimization iterations (default: 5000)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda if available, else cpu)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.image_path):
        print(f"Error: File not found: {args.image_path}")
        return
    
    print("=" * 60)
    print("CT Image Reconstruction with Deep Image Prior")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Input: {args.image_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Sampling rate: {args.sampling_rate * 100}%")
    print(f"  Mask type: {args.mask_type}")
    print(f"  Noise level: {args.noise_level}")
    print(f"  Iterations: {args.num_iter}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    # Process image
    results = process_ct_image(
        image_path=args.image_path,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        mask_type=args.mask_type,
        noise_level=args.noise_level,
        num_iter=args.num_iter,
        lr=args.lr,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Final PSNR: {results['psnr']:.2f} dB")
    print(f"Final SSIM: {results['ssim']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
