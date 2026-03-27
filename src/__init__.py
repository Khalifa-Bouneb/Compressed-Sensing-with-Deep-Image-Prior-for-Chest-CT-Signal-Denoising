# Empty file to make src a Python package
"""
Deep Image Prior for CT Signal Denoising - Source Package
=========================================================

Modules:
- metrics.py: PSNR, SSIM, DSSIM evaluation metrics
- degrade.py: Image degradation (noise, blur) for y = Ax + η
- baselines.py: Classical restoration methods (Wiener, BM3D)
"""

from . import metrics
from . import degrade
from . import baselines

__version__ = "0.1.0"
__author__ = "DIP Project Team"
