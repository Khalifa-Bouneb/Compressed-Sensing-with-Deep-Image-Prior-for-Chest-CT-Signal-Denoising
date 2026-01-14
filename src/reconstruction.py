"""
Deep Image Prior reconstruction for compressed sensing.
This module implements the optimization procedure for CS reconstruction using DIP.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..models.unet import UNet, get_input_noise
from ..utils.cs_utils import CompressedSensingOperator, normalize_image
from ..utils.metrics import calculate_psnr, calculate_ssim


class DIPReconstructor:
    """
    Deep Image Prior reconstructor for compressed sensing.
    Optimizes a randomly initialized neural network to fit undersampled measurements.
    """
    
    def __init__(
        self,
        image_shape,
        mask,
        input_depth=32,
        lr=0.01,
        num_iter=5000,
        num_channels_down=None,
        num_channels_up=None,
        num_channels_skip=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the DIP reconstructor.
        
        Args:
            image_shape: Shape of the target image (H, W)
            mask: Undersampling mask
            input_depth: Number of channels for input noise
            lr: Learning rate
            num_iter: Number of optimization iterations
            num_channels_down: List of channel sizes for downsampling (optional)
            num_channels_up: List of channel sizes for upsampling (optional)
            num_channels_skip: List of channel sizes for skip connections (optional)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.image_shape = image_shape
        self.mask = mask
        self.input_depth = input_depth
        self.lr = lr
        self.num_iter = num_iter
        self.device = device
        
        # Default network architecture if not specified
        if num_channels_down is None:
            num_channels_down = [128, 128, 128, 128, 128]
        if num_channels_up is None:
            num_channels_up = [128, 128, 128, 128, 128]
        if num_channels_skip is None:
            num_channels_skip = [4, 4, 4, 4, 4]
        
        # Initialize the network
        self.net = UNet(
            input_channels=input_depth,
            output_channels=1,
            num_channels_down=num_channels_down,
            num_channels_up=num_channels_up,
            num_channels_skip=num_channels_skip
        ).to(device)
        
        # Initialize the CS operator
        self.cs_operator = CompressedSensingOperator(mask)
        
        # Initialize input noise (fixed during optimization)
        self.input_noise = get_input_noise(
            input_depth,
            image_shape,
            noise_type='uniform'
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Storage for tracking
        self.losses = []
        self.psnrs = []
        self.ssims = []
    
    def reconstruct(self, measurements, ground_truth=None, log_interval=100):
        """
        Reconstruct image from undersampled measurements.
        
        Args:
            measurements: Undersampled measurements (with mask applied)
            ground_truth: Ground truth image for evaluation (optional)
            log_interval: Interval for logging metrics
            
        Returns:
            Reconstructed image
        """
        # Convert to tensor if needed
        if not isinstance(measurements, torch.Tensor):
            measurements = torch.from_numpy(measurements).float()
        
        measurements = measurements.to(self.device)
        
        # Ensure measurements have correct shape
        if measurements.dim() == 2:
            measurements = measurements.unsqueeze(0).unsqueeze(0)
        elif measurements.dim() == 3:
            measurements = measurements.unsqueeze(0)
        
        # Normalize measurements
        measurements = normalize_image(measurements)
        
        # Ground truth for evaluation
        if ground_truth is not None:
            if not isinstance(ground_truth, torch.Tensor):
                ground_truth = torch.from_numpy(ground_truth).float()
            ground_truth = ground_truth.to(self.device)
            if ground_truth.dim() == 2:
                ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)
            elif ground_truth.dim() == 3:
                ground_truth = ground_truth.unsqueeze(0)
            ground_truth = normalize_image(ground_truth)
        
        # Get mask tensor
        mask_tensor = torch.from_numpy(self.cs_operator.mask).float().to(self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        
        # Optimization loop
        self.net.train()
        pbar = tqdm(range(self.num_iter), desc='DIP Reconstruction')
        
        for iteration in pbar:
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.net(self.input_noise)
            
            # Apply measurement operator
            measured_output = output * mask_tensor
            
            # Compute loss (data fidelity term)
            loss = self.criterion(measured_output, measurements)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Store loss
            self.losses.append(loss.item())
            
            # Evaluate if ground truth is available
            if ground_truth is not None and (iteration % log_interval == 0 or iteration == self.num_iter - 1):
                with torch.no_grad():
                    psnr = calculate_psnr(ground_truth, output)
                    ssim = calculate_ssim(ground_truth, output)
                    self.psnrs.append(psnr)
                    self.ssims.append(ssim)
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'PSNR': f'{psnr:.2f}',
                        'SSIM': f'{ssim:.4f}'
                    })
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Final reconstruction
        self.net.eval()
        with torch.no_grad():
            reconstructed = self.net(self.input_noise)
        
        return reconstructed
    
    def get_metrics_history(self):
        """
        Get the history of losses and metrics.
        
        Returns:
            Dictionary containing losses, PSNRs, and SSIMs
        """
        return {
            'losses': self.losses,
            'psnrs': self.psnrs,
            'ssims': self.ssims
        }


def reconstruct_with_dip(
    measurements,
    mask,
    image_shape,
    ground_truth=None,
    input_depth=32,
    lr=0.01,
    num_iter=5000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_interval=100
):
    """
    Convenience function for DIP reconstruction.
    
    Args:
        measurements: Undersampled measurements
        mask: Undersampling mask
        image_shape: Shape of target image (H, W)
        ground_truth: Ground truth for evaluation (optional)
        input_depth: Number of input noise channels
        lr: Learning rate
        num_iter: Number of iterations
        device: Device to run on
        log_interval: Logging interval
        
    Returns:
        Reconstructed image tensor
    """
    reconstructor = DIPReconstructor(
        image_shape=image_shape,
        mask=mask,
        input_depth=input_depth,
        lr=lr,
        num_iter=num_iter,
        device=device
    )
    
    reconstructed = reconstructor.reconstruct(
        measurements,
        ground_truth=ground_truth,
        log_interval=log_interval
    )
    
    return reconstructed, reconstructor
