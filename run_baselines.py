"""
Run All Baselines Script
========================

This script automates the classical baseline evaluation across all test images.
Outputs:
- PSNR/SSIM table to console
- CSV file with detailed results
- Comparison montage images

Usage:
    python run_baselines.py --sigma 0.1 --num_images 10
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.metrics import psnr, ssim, evaluate_image
from src.degrade import load_image, degrade_for_denoising
from src.baselines import wiener_deconvolution, bm3d_denoise, total_variation_denoise, HAS_BM3D


def run_benchmark(test_images: list, sigma: float = 0.1, seed: int = 42):
    """
    Run classical baselines on a set of test images.
    
    Parameters
    ----------
    test_images : list
        List of paths to test images.
    sigma : float
        Noise standard deviation.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    dict
        Benchmark results dictionary.
    """
    results = {
        'image_id': [],
        'noisy_psnr': [], 'noisy_ssim': [],
        'wiener_psnr': [], 'wiener_ssim': [],
        'tv_psnr': [], 'tv_ssim': [],
        'bm3d_psnr': [], 'bm3d_ssim': [],
    }
    
    print(f"\n{'='*70}")
    print(f"Running baseline benchmark on {len(test_images)} images (σ = {sigma})")
    print(f"{'='*70}\n")
    
    for i, img_path in enumerate(test_images):
        img_id = Path(img_path).stem
        print(f"[{i+1:2d}/{len(test_images)}] Processing: {img_id}")
        
        # Load and degrade
        clean = load_image(img_path, normalize=True, grayscale=False)
        noisy, _ = degrade_for_denoising(clean, sigma=sigma, seed=seed+i)
        
        # Apply baselines
        wiener_result = wiener_deconvolution(noisy, kernel=None)
        tv_result = total_variation_denoise(noisy, weight=0.15)
        
        if HAS_BM3D:
            bm3d_result = bm3d_denoise(noisy, sigma=sigma)
        else:
            bm3d_result = tv_result
        
        # Store results
        results['image_id'].append(img_id)
        
        noisy_m = evaluate_image(clean, noisy)
        results['noisy_psnr'].append(noisy_m['psnr'])
        results['noisy_ssim'].append(noisy_m['ssim'])
        
        wiener_m = evaluate_image(clean, wiener_result)
        results['wiener_psnr'].append(wiener_m['psnr'])
        results['wiener_ssim'].append(wiener_m['ssim'])
        
        tv_m = evaluate_image(clean, tv_result)
        results['tv_psnr'].append(tv_m['psnr'])
        results['tv_ssim'].append(tv_m['ssim'])
        
        bm3d_m = evaluate_image(clean, bm3d_result)
        results['bm3d_psnr'].append(bm3d_m['psnr'])
        results['bm3d_ssim'].append(bm3d_m['ssim'])
        
        print(f"         Noisy: {noisy_m['psnr']:.2f} dB | "
              f"Wiener: {wiener_m['psnr']:.2f} dB | "
              f"TV: {tv_m['psnr']:.2f} dB | "
              f"BM3D: {bm3d_m['psnr']:.2f} dB")
    
    return results


def print_table(results: dict, sigma: float):
    """Print formatted results table."""
    
    print(f"\n\n{'='*90}")
    print(f"{'DENOISING BENCHMARK RESULTS (σ = ' + str(sigma) + ')':^90}")
    print(f"{'='*90}")
    print(f"{'Image ID':<15} | {'Noisy':>12} | {'Wiener':>12} | {'TV':>12} | {'BM3D':>12} |")
    print(f"{'-'*90}")
    
    for i in range(len(results['image_id'])):
        print(f"{results['image_id'][i]:<15} | "
              f"{results['noisy_psnr'][i]:>10.2f} dB | "
              f"{results['wiener_psnr'][i]:>10.2f} dB | "
              f"{results['tv_psnr'][i]:>10.2f} dB | "
              f"{results['bm3d_psnr'][i]:>10.2f} dB |")
    
    print(f"{'-'*90}")
    
    # Averages
    avg_noisy = np.mean(results['noisy_psnr'])
    avg_wiener = np.mean(results['wiener_psnr'])
    avg_tv = np.mean(results['tv_psnr'])
    avg_bm3d = np.mean(results['bm3d_psnr'])
    
    print(f"{'AVERAGE':<15} | "
          f"{avg_noisy:>10.2f} dB | "
          f"{avg_wiener:>10.2f} dB | "
          f"{avg_tv:>10.2f} dB | "
          f"{avg_bm3d:>10.2f} dB |")
    print(f"{'='*90}")
    
    return avg_noisy, avg_wiener, avg_tv, avg_bm3d


def save_csv(results: dict, output_path: str, sigma: float):
    """Save results to CSV file."""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['sigma', sigma])
        writer.writerow([])
        writer.writerow(['Image ID', 'Noisy PSNR', 'Wiener PSNR', 'TV PSNR', 'BM3D PSNR',
                         'Noisy SSIM', 'Wiener SSIM', 'TV SSIM', 'BM3D SSIM'])
        
        # Data rows
        for i in range(len(results['image_id'])):
            writer.writerow([
                results['image_id'][i],
                f"{results['noisy_psnr'][i]:.2f}",
                f"{results['wiener_psnr'][i]:.2f}",
                f"{results['tv_psnr'][i]:.2f}",
                f"{results['bm3d_psnr'][i]:.2f}",
                f"{results['noisy_ssim'][i]:.4f}",
                f"{results['wiener_ssim'][i]:.4f}",
                f"{results['tv_ssim'][i]:.4f}",
                f"{results['bm3d_ssim'][i]:.4f}",
            ])
        
        # Averages
        writer.writerow([
            'AVERAGE',
            f"{np.mean(results['noisy_psnr']):.2f}",
            f"{np.mean(results['wiener_psnr']):.2f}",
            f"{np.mean(results['tv_psnr']):.2f}",
            f"{np.mean(results['bm3d_psnr']):.2f}",
            f"{np.mean(results['noisy_ssim']):.4f}",
            f"{np.mean(results['wiener_ssim']):.4f}",
            f"{np.mean(results['tv_ssim']):.4f}",
            f"{np.mean(results['bm3d_ssim']):.4f}",
        ])
    
    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run classical denoising baselines')
    parser.add_argument('--sigma', type=float, default=0.1, 
                        help='Noise standard deviation (default: 0.1)')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of test images to use (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='results/baseline_benchmark.csv',
                        help='Output CSV path')
    
    args = parser.parse_args()
    
    # Get test images
    dataset_path = Path(__file__).parent / 'Dataset' / 'BSDS300' / 'BSDS300' / 'images' / 'test'
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    image_files = sorted(list(dataset_path.glob('*.jpg')))[:args.num_images]
    
    if len(image_files) == 0:
        print("Error: No images found in dataset")
        sys.exit(1)
    
    print(f"\n✓ BM3D available: {HAS_BM3D}")
    if not HAS_BM3D:
        print("  (Install with: pip install bm3d)")
    
    # Run benchmark
    results = run_benchmark(image_files, sigma=args.sigma, seed=args.seed)
    
    # Print table
    avg_noisy, avg_wiener, avg_tv, avg_bm3d = print_table(results, args.sigma)
    
    # Save to CSV
    save_csv(results, args.output, args.sigma)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Best classical method: BM3D ({avg_bm3d:.2f} dB)")
    print(f"  Improvement over noisy: +{avg_bm3d - avg_noisy:.2f} dB")
    print(f"\n  🎯 Target for Deep Image Prior: Beat {avg_bm3d:.2f} dB!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
