"""Utilities for compressed sensing operations."""

import torch
import numpy as np


class UnderSamplingMask:
    """
    Generate undersampling masks for compressed sensing.
    Supports random, radial, and Cartesian sampling patterns.
    """
    
    def __init__(self, shape, sampling_rate=0.3, mask_type='random'):
        """
        Initialize the undersampling mask.
        
        Args:
            shape: Shape of the mask (height, width)
            sampling_rate: Fraction of samples to keep (0 to 1)
            mask_type: Type of mask ('random', 'radial', 'cartesian')
        """
        self.shape = shape
        self.sampling_rate = sampling_rate
        self.mask_type = mask_type
        self.mask = self._generate_mask()
    
    def _generate_mask(self):
        """Generate the undersampling mask."""
        h, w = self.shape
        
        if self.mask_type == 'random':
            mask = np.random.random((h, w)) < self.sampling_rate
        
        elif self.mask_type == 'cartesian':
            # Sample full lines in one direction
            mask = np.zeros((h, w))
            num_lines = int(h * self.sampling_rate)
            indices = np.random.choice(h, num_lines, replace=False)
            mask[indices, :] = 1
        
        elif self.mask_type == 'radial':
            # Radial sampling pattern (more samples at center)
            center_h, center_w = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            distances = np.sqrt((x - center_w)**2 + (y - center_h)**2)
            max_distance = np.sqrt(center_h**2 + center_w**2)
            
            # Higher probability for center frequencies
            probabilities = 1 - (distances / max_distance) * (1 - self.sampling_rate)
            mask = np.random.random((h, w)) < probabilities
        
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")
        
        return mask.astype(np.float32)
    
    def apply(self, image):
        """
        Apply the undersampling mask to an image.
        
        Args:
            image: Input image tensor or array
            
        Returns:
            Masked image
        """
        if isinstance(image, torch.Tensor):
            mask_tensor = torch.from_numpy(self.mask).to(image.device)
            return image * mask_tensor
        else:
            return image * self.mask
    
    def get_mask_tensor(self, device='cpu'):
        """Get the mask as a PyTorch tensor."""
        return torch.from_numpy(self.mask).to(device)


class CompressedSensingOperator:
    """
    Compressed Sensing measurement operator.
    Implements forward and adjoint operations for CS reconstruction.
    """
    
    def __init__(self, mask):
        """
        Initialize the CS operator.
        
        Args:
            mask: Undersampling mask (UnderSamplingMask object or array)
        """
        if isinstance(mask, UnderSamplingMask):
            self.mask = mask.mask
        else:
            self.mask = mask
    
    def forward(self, x):
        """
        Forward operator: undersample the image.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Undersampled measurements
        """
        if isinstance(x, torch.Tensor):
            mask_tensor = torch.from_numpy(self.mask).to(x.device)
            if x.dim() == 4:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(0)
            return x * mask_tensor
        else:
            return x * self.mask
    
    def adjoint(self, y):
        """
        Adjoint operator: apply mask transpose (same as forward for binary mask).
        
        Args:
            y: Undersampled measurements
            
        Returns:
            Adjoint result
        """
        return self.forward(y)
    
    def forward_adjoint(self, x):
        """
        Apply forward then adjoint operation.
        
        Args:
            x: Input image
            
        Returns:
            Result of A^T(A(x))
        """
        return self.adjoint(self.forward(x))


def add_noise(image, noise_level=0.01):
    """
    Add Gaussian noise to an image.
    
    Args:
        image: Input image tensor
        noise_level: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    if isinstance(image, torch.Tensor):
        noise = torch.randn_like(image) * noise_level
        return image + noise
    else:
        noise = np.random.randn(*image.shape) * noise_level
        return image + noise


def normalize_image(image, min_val=None, max_val=None):
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image
        min_val: Minimum value for normalization (if None, use image min)
        max_val: Maximum value for normalization (if None, use image max)
        
    Returns:
        Normalized image
    """
    if isinstance(image, torch.Tensor):
        if min_val is None:
            min_val = image.min()
        if max_val is None:
            max_val = image.max()
        return (image - min_val) / (max_val - min_val + 1e-8)
    else:
        if min_val is None:
            min_val = image.min()
        if max_val is None:
            max_val = image.max()
        return (image - min_val) / (max_val - min_val + 1e-8)
