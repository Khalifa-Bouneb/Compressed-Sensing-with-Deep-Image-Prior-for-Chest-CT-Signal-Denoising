# Installation Guide

This guide will help you set up the Compressed Sensing with Deep Image Prior project.

## Prerequisites

- Python 3.7 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster computation

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Khalifa-Bouneb/Compressed-Sensing-with-Deep-Image-Prior-for-Chest-CT-Signal-Denoising.git
cd Compressed-Sensing-with-Deep-Image-Prior-for-Chest-CT-Signal-Denoising
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n dip-cs python=3.9
conda activate dip-cs
```

### 3. Install Dependencies

#### CPU-only Installation

```bash
pip install -r requirements.txt
```

#### GPU Installation (CUDA)

If you have a CUDA-capable GPU, install PyTorch with CUDA support first:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt
```

For other CUDA versions, visit: https://pytorch.org/get-started/locally/

### 4. Install the Package

For development:
```bash
pip install -e .
```

For regular use:
```bash
pip install .
```

### 5. Verify Installation

Run the structure test to verify everything is set up correctly:

```bash
python examples/test_structure.py
```

You should see all checkmarks (✓) indicating successful setup.

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError` for any package:
```bash
pip install <missing-package>
```

### CUDA/GPU Issues

Check if CUDA is available:
```python
import torch
print(torch.cuda.is_available())
```

If False, PyTorch will use CPU (which is slower but still functional).

### Memory Issues

For large images or many iterations:
- Reduce `num_iter` in the configuration
- Use smaller image sizes
- Close other applications to free up memory

## Next Steps

After installation, try the example:
```bash
python examples/basic_reconstruction.py
```

See [USAGE.md](USAGE.md) for more detailed usage instructions.
