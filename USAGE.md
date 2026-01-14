# Usage Guide

This guide provides detailed instructions for using the DIP-CS library.

## Basic Workflow

### 1. Import Required Modules

```python
import numpy as np
from src.utils import (
    UnderSamplingMask,
    CompressedSensingOperator,
    add_noise,
    normalize_image,
    calculate_psnr,
    calculate_ssim,
    plot_results
)
from src.reconstruction import reconstruct_with_dip
```

### 2. Load Your Image

```python
from PIL import Image

# Load CT image
img = Image.open('path/to/ct_image.png').convert('L')
img = np.array(img).astype(np.float32) / 255.0

# Resize if needed
from PIL import Image
img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((256, 256)))
img = img.astype(np.float32) / 255.0
```

### 3. Create Undersampling Mask

```python
# Random mask (30% sampling)
mask = UnderSamplingMask(
    shape=img.shape,
    sampling_rate=0.3,
    mask_type='random'
)

# Or Cartesian mask (full lines)
mask = UnderSamplingMask(
    shape=img.shape,
    sampling_rate=0.3,
    mask_type='cartesian'
)

# Or Radial mask (center-focused)
mask = UnderSamplingMask(
    shape=img.shape,
    sampling_rate=0.3,
    mask_type='radial'
)
```

### 4. Generate Measurements

```python
# Create CS operator
cs_operator = CompressedSensingOperator(mask)

# Apply mask to get measurements
measurements = cs_operator.forward(img)

# Optionally add noise
measurements_noisy = add_noise(measurements, noise_level=0.01)
```

### 5. Reconstruct Using DIP

```python
# Reconstruct
reconstructed, reconstructor = reconstruct_with_dip(
    measurements=measurements_noisy,
    mask=mask,
    image_shape=img.shape,
    ground_truth=img,  # Optional, for evaluation
    num_iter=5000,
    lr=0.01,
    device='cuda'  # or 'cpu'
)
```

### 6. Evaluate Results

```python
# Calculate metrics
reconstructed_np = reconstructed.detach().cpu().squeeze().numpy()
psnr = calculate_psnr(img, reconstructed_np)
ssim = calculate_ssim(img, reconstructed_np)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
```

### 7. Visualize

```python
# Plot results
plot_results(
    original=img,
    noisy=measurements_noisy,
    reconstructed=reconstructed_np,
    save_path='results.png'
)

# Plot convergence
from src.utils import plot_convergence
history = reconstructor.get_metrics_history()
plot_convergence(
    losses=history['losses'],
    psnrs=history['psnrs'],
    ssims=history['ssims'],
    save_path='convergence.png'
)
```

## Advanced Usage

### Custom Network Architecture

```python
from src.models import UNet

# Create custom UNet
net = UNet(
    input_channels=32,
    output_channels=1,
    num_channels_down=[64, 128, 256],  # Custom layers
    num_channels_up=[256, 128, 64],
    num_channels_skip=[4, 4, 4]
)
```

### Custom Reconstruction Loop

```python
from src.reconstruction import DIPReconstructor

# Create reconstructor
reconstructor = DIPReconstructor(
    image_shape=img.shape,
    mask=mask,
    input_depth=32,
    lr=0.01,
    num_iter=5000,
    device='cuda'
)

# Reconstruct with custom settings
reconstructed = reconstructor.reconstruct(
    measurements=measurements_noisy,
    ground_truth=img,
    log_interval=50  # Log every 50 iterations
)
```

### Working with Configuration Files

```python
import yaml

# Load configuration
with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use config parameters
mask = UnderSamplingMask(
    shape=(config['image']['height'], config['image']['width']),
    sampling_rate=config['compressed_sensing']['sampling_rate'],
    mask_type=config['compressed_sensing']['mask_type']
)
```

## Parameter Tuning

### Sampling Rate
- Lower rates (10-20%): More challenging, may need more iterations
- Medium rates (30-40%): Good balance
- Higher rates (50%+): Easier reconstruction, less benefit from DIP

### Number of Iterations
- Quick test: 1000-2000 iterations
- Standard: 3000-5000 iterations
- High quality: 5000-10000 iterations

### Learning Rate
- Start with: 0.01
- If unstable: reduce to 0.001
- If too slow: increase to 0.05

### Input Depth
- Standard: 32 channels
- More complex images: 64 channels
- Simple images: 16 channels

## Examples

### Example 1: Quick Reconstruction

```python
# Minimal code for quick reconstruction
from src import reconstruct_with_dip, UnderSamplingMask

reconstructed, _ = reconstruct_with_dip(
    measurements=your_measurements,
    mask=UnderSamplingMask(shape=(256, 256), sampling_rate=0.3),
    image_shape=(256, 256),
    num_iter=2000
)
```

### Example 2: High-Quality Reconstruction

```python
# Detailed reconstruction with evaluation
reconstructed, reconstructor = reconstruct_with_dip(
    measurements=measurements,
    mask=mask,
    image_shape=img.shape,
    ground_truth=img,
    input_depth=64,
    lr=0.01,
    num_iter=8000,
    device='cuda'
)

# Get detailed metrics
history = reconstructor.get_metrics_history()
```

### Example 3: Batch Processing

```python
# Process multiple images
import os
from glob import glob

image_files = glob('data/*.png')

for img_path in image_files:
    img = load_image(img_path)
    measurements = get_measurements(img)
    reconstructed, _ = reconstruct_with_dip(
        measurements=measurements,
        mask=mask,
        image_shape=img.shape,
        num_iter=3000
    )
    # Save result
    save_path = f"output/{os.path.basename(img_path)}"
    save_image(reconstructed, save_path)
```

## Tips and Best Practices

1. **Start Small**: Test with small images and few iterations first
2. **Use GPU**: GPU acceleration provides 10-50x speedup
3. **Monitor Convergence**: Check if loss is decreasing; if not, adjust learning rate
4. **Early Stopping**: You can stop early if metrics plateau
5. **Experiment**: Try different mask types and sampling rates
6. **Normalize**: Always normalize images to [0, 1] range

## Common Issues

### Slow Reconstruction
- Use GPU if available
- Reduce image size
- Decrease number of iterations

### Poor Quality Results
- Increase number of iterations
- Try different mask types
- Increase sampling rate
- Adjust learning rate

### Out of Memory
- Reduce image size
- Use CPU instead of GPU
- Reduce network complexity

## Getting Help

- Check the [README.md](README.md) for overview
- See [examples/](examples/) directory for code samples
- Open an issue on GitHub for bugs or questions
