"""
Rohan Sanda, Akhilesh Balasingam 2024

This file contains the wrapper for BM3D denoising algorithm.
Set configuration parameters in config.py file.
Computes and saves metrics automatically. 
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import bm3d
from tqdm import tqdm

# set random seeds
torch.manual_seed(1)
np.random.seed(1)

torch.use_deterministic_algorithms(True)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from .config import *
from .models_dip import *
from .utils import *

dtype = torch.FloatTensor  # This specifies that we are using CPU

def bm3d_single(img_pil, img_np, img_noisy_np, ind, verbose=False):
    sigma = BM3D_PARAMS['sigma']
    
    img_np = img_np.astype(np.float32)
    img_noisy_np = img_noisy_np.astype(np.float32)
    
    bm3d_image_noisy_input = np.transpose(img_noisy_np, (1, 2, 0))
    bm3d_image = bm3d.bm3d(bm3d_image_noisy_input, sigma)   
    bm3d_image_out = np.transpose(bm3d_image, (2, 0, 1)).astype(np.float32)
    
    #print(bm3d_image_out.shape, np.max(bm3d_image_out), np.min(bm3d_image_out))
        
    # Compute metrics
    psrn_noisy = compare_psnr(img_noisy_np, bm3d_image_out) 
    psrn_gt    = compare_psnr(img_np, bm3d_image_out) 
    ssim_gt, _ = ssim(img_np.transpose(1,2,0), bm3d_image_out.transpose(1,2,0), win_size=7, full=True, data_range = 1.0, channel_axis=2)
    ssim_noisy, _ = ssim(img_noisy_np.transpose(1,2,0), bm3d_image_out.transpose(1,2,0), win_size=7, full=True, data_range = 1.0, channel_axis=2)
    sobel_gt = cv2.Sobel(img_np.transpose(1,2,0), cv2.CV_64F, 1,1, ksize=5)
    sobel_out = cv2.Sobel(bm3d_image_out.transpose(1,2,0), cv2.CV_64F, 1,1, ksize=5)
    dssim, _ = ssim(sobel_gt, sobel_out, win_size=7, full=True, data_range = 1.0, channel_axis=2)
    
    print(f"PSNR_noisy: {psrn_noisy},  PSRN_gt: {psrn_gt}")
    print(f"SSIM_gt: {ssim_gt},  SSIM_noisy: {ssim_noisy},  DSSIM: {dssim}")
    
    metrics = {
        "PSNR_noisy": psrn_noisy, 
        "PSNR_gt": psrn_gt, 
        # "PSNR_gt_sm": psrn_gt,
        "SSIM_gt": ssim_gt,
        "SSIM_noisy": ssim_noisy,
        "DSSIM": dssim,
    }
    
    # Save plot of bm3d output vs. noisy image
    if  BM3D_PARAMS['PLOT']:
        plot_image_grid([np.clip(bm3d_image_out, 0, 1), 
                        img_noisy_np, img_np], factor=13, nrow=1, view = verbose, index=f"image={ind}", prefix="BM3D", tag=f"noise={sigma}")
    
    return metrics, bm3d_image_out