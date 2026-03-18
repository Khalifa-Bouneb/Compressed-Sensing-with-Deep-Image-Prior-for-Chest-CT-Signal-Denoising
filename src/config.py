import torch
from torch import nn
import torch.optim as optim

# Random seed, will be used in all files for reproducibility
SEED = 1

MODEL_TYPE = 'SDDP'   # DIP, DCNN, ADMM-DIP, BM3D

BM3D_PARAMS = {
    'sigma': 0.01,
    'PLOT': True
}

# Training params
DIP_PARAMS = {
    'imsize': -1,           # -1 for no resizing
    'PLOT': True,           # Save image of model output and noisy image sidebyside
    'sigma': 0.01,           # Noise level std
    'INPUT': 'noise',       # Set to 'noise' for denoising and deblurring
    'pad': 'zero',          # Padding type for convolutional layers
    'OPT_OVER': 'net',
    'reg_noise_std': 1/40,  # Noise level std for regularization
    'LR': 0.001,            # Learning rate
    'OPTIMIZER': 'adam',  
    'show_every': 400,      # Save metrics and plots every n iterations
    'num_iter': 2000,       # Number of iterations
    'input_depth': 32,      # Number of input channels for pure noise input to DIP model
    'figsize': 4,           # Size of the plot (output vs. noisy image)
    'exp_weight': 0.99      # No longer used 

}

SMART_DEBLUR_DIP_PARAMS = {
    'imsize': -1,
    'PLOT': True,
    'sigma': 0.2,
    'INPUT': 'noise',
    'pad': 'reflection',
    'OPT_OVER': 'net',
    'reg_noise_std': 1/40, 
    'LR': 0.001,
    'OPTIMIZER': 'adam', 
    'show_every': 50,
    'exp_weight': 0.99,
    'num_iter': 2500,
    'input_depth': 32,
    'figsize': 4
}

# DIP + TV denoising
ADMM_DIP_PARAMS = {
    'imsize': -1,
    'PLOT': True,
    'sigma': 0.1,
    'INPUT': 'noise',
    'pad': 'reflection',
    'OPT_OVER': 'net',
    'reg_noise_std': 1. / 40., 
    'LR': 0.001,
    'OPTIMIZER': 'adam', 
    'show_every': 200,
    'exp_weight': 0,
    'num_iter': 500,
    'input_depth': 3,
    'figsize': 4
}
ALL_PARAMS = {
    'DIP': DIP_PARAMS,
    'ADMM-DIP': ADMM_DIP_PARAMS,
    'BM3D': BM3D_PARAMS
}




