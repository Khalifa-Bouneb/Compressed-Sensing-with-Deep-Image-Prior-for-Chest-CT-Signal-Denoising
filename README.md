# Compressed Sensing with Deep Image Prior for Chest CT Signal Denoising

A PyTorch implementation of image reconstruction using Deep Image Prior (DIP) for compressed sensing applications, specifically designed for chest CT signal denoising.

## Overview

This repository implements a deep learning-based approach to reconstruct high-quality CT images from undersampled measurements. The method combines:

- **Compressed Sensing (CS)**: Efficient signal acquisition using fewer measurements than traditional methods
- **Deep Image Prior (DIP)**: Using the structure of a convolutional neural network as an implicit prior for image reconstruction
- **Medical Imaging**: Application to chest CT denoising and reconstruction

### Key Features

- 🧠 UNet-based Deep Image Prior architecture
- 📊 Multiple undersampling patterns (random, Cartesian, radial)
- 📈 Real-time metrics tracking (PSNR, SSIM)
- 🎨 Comprehensive visualization tools
- ⚡ GPU acceleration support
- 📝 Easy-to-use API and examples

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Khalifa-Bouneb/Compressed-Sensing-with-Deep-Image-Prior-for-Chest-CT-Signal-Denoising.git
cd Compressed-Sensing-with-Deep-Image-Prior-for-Chest-CT-Signal-Denoising
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import numpy as np
from src.utils import UnderSamplingMask, CompressedSensingOperator
from src.reconstruction import reconstruct_with_dip

# Load your CT image (as numpy array)
ct_image = ...  # Shape: (H, W)

# Create undersampling mask
mask = UnderSamplingMask(
    shape=ct_image.shape,
    sampling_rate=0.3,  # Keep 30% of measurements
    mask_type='random'
)

# Generate measurements
cs_operator = CompressedSensingOperator(mask)
measurements = cs_operator.forward(ct_image)

# Reconstruct using DIP
reconstructed, reconstructor = reconstruct_with_dip(
    measurements=measurements,
    mask=mask,
    image_shape=ct_image.shape,
    ground_truth=ct_image,
    num_iter=5000
)
```

### Run Example

```bash
python examples/basic_reconstruction.py
```

This will:
1. Create or load a test image
2. Apply undersampling
3. Reconstruct using DIP
4. Save results to `examples/output/`

## Architecture

### Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py              # UNet architecture for DIP
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── cs_utils.py          # Compressed sensing utilities
│   │   ├── metrics.py           # Evaluation metrics (PSNR, SSIM)
│   │   └── visualization.py     # Plotting and visualization
│   ├── __init__.py
│   └── reconstruction.py        # Main reconstruction pipeline
├── examples/
│   └── basic_reconstruction.py  # Example usage
├── configs/
│   └── default_config.yaml      # Configuration file
├── requirements.txt
└── README.md
```

### Deep Image Prior Network

The implementation uses a UNet architecture with:
- 5 downsampling layers
- 5 upsampling layers
- Skip connections for multi-scale features
- Batch normalization and LeakyReLU activations

## Usage Details

### Undersampling Masks

Three types of undersampling patterns are supported:

```python
# Random undersampling
mask = UnderSamplingMask(shape=(256, 256), sampling_rate=0.3, mask_type='random')

# Cartesian undersampling (full horizontal lines)
mask = UnderSamplingMask(shape=(256, 256), sampling_rate=0.3, mask_type='cartesian')

# Radial undersampling (more samples at center)
mask = UnderSamplingMask(shape=(256, 256), sampling_rate=0.3, mask_type='radial')
```

### Reconstruction Parameters

Key parameters for reconstruction:

- `num_iter`: Number of optimization iterations (default: 5000)
- `lr`: Learning rate (default: 0.01)
- `input_depth`: Number of channels in input noise (default: 32)
- `sampling_rate`: Fraction of measurements to keep (0-1)
- `noise_level`: Standard deviation of Gaussian noise

### Evaluation Metrics

The library provides common image quality metrics:

```python
from src.utils import calculate_psnr, calculate_ssim

psnr = calculate_psnr(original, reconstructed)  # Peak Signal-to-Noise Ratio
ssim = calculate_ssim(original, reconstructed)  # Structural Similarity Index
```

## Configuration

Edit `configs/default_config.yaml` to customize:
- Image dimensions
- Sampling parameters
- Network architecture
- Optimization settings
- Output options

## Results Visualization

The library automatically generates:
- Side-by-side comparison (original, noisy, reconstructed)
- Convergence curves (loss, PSNR, SSIM)
- Sampling mask visualization

Example output:

```
output/
├── reconstruction_results.png   # Visual comparison
├── convergence.png              # Metrics over iterations
└── sampling_mask.png            # Undersampling pattern
```

## Theory

### Deep Image Prior

Deep Image Prior leverages the implicit bias of convolutional neural networks towards natural images. Instead of learning from a dataset, DIP fits a randomly initialized network to a single image, using the network architecture itself as a prior.

### Compressed Sensing

Compressed Sensing enables reconstruction from fewer measurements by exploiting signal sparsity. The reconstruction problem is formulated as:

```
minimize ||x||_TV  subject to  y = Ax
```

where:
- `y`: measurements
- `A`: measurement operator (undersampling mask)
- `x`: image to reconstruct
- TV: total variation (regularization)

DIP replaces explicit regularization with the implicit prior from the network structure.

## Applications

This implementation is suitable for:
- CT image reconstruction from undersampled data
- Medical image denoising
- Accelerated MRI reconstruction
- Any compressed sensing reconstruction task

## Performance

Typical results (256×256 images, 30% sampling):
- PSNR: 25-35 dB (depends on image complexity)
- SSIM: 0.85-0.95
- Reconstruction time: 2-5 minutes on GPU

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). "Deep Image Prior." CVPR.
2. Donoho, D. L. (2006). "Compressed sensing." IEEE Transactions on Information Theory.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dip_cs_ct_2024,
  title={Compressed Sensing with Deep Image Prior for Chest CT Signal Denoising},
  author={Khalifa Bouneb},
  year={2024},
  url={https://github.com/Khalifa-Bouneb/Compressed-Sensing-with-Deep-Image-Prior-for-Chest-CT-Signal-Denoising}
}
```

## Acknowledgments

- Original Deep Image Prior paper and implementation
- PyTorch team for the deep learning framework
- scikit-image for image processing utilities
