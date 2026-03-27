"""
Rohan Sanda 2024
This code is adapted from Cascarano et al., 2020, "Combining Weighted Total Variation and Deep 
Image Prior for natural and medical image restoration via ADMM"

Repo: https://github.com/sedaboni/ADMM-DIPTV/tree/master
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import cv2

# set random seeds
torch.manual_seed(1)
np.random.seed(1)

torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


from .config import *
from .models_dip import *
from .utils import *
from .hw5_task2 import BSDS300Dataset

# Set config params
imsize = ADMM_DIP_PARAMS['imsize']
PLOT = ADMM_DIP_PARAMS['PLOT']
sigma = ADMM_DIP_PARAMS['sigma']
INPUT = ADMM_DIP_PARAMS['INPUT']
pad = ADMM_DIP_PARAMS['pad']
OPT_OVER = ADMM_DIP_PARAMS['OPT_OVER']
reg_noise_std = ADMM_DIP_PARAMS['reg_noise_std']
LR = ADMM_DIP_PARAMS['LR']
OPTIMIZER= ADMM_DIP_PARAMS['OPTIMIZER']
show_every = ADMM_DIP_PARAMS['show_every']
exp_weight= ADMM_DIP_PARAMS['exp_weight']

num_iter = ADMM_DIP_PARAMS['num_iter']
input_depth = ADMM_DIP_PARAMS['input_depth']
figsize = ADMM_DIP_PARAMS['figsize']


def admm_dip_single(img_pil, img_clean_np, y, ind, verbose=False):
    
    net = get_net(input_depth, 'skip', pad,
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    upsample_mode='bilinear')


    # Convert net to the specified dtype
    # net = net.type(dtype)
    net = net.to(device)

    # net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).to(device).detach()
    # Compute number of parameters
    if verbose:
        s  = sum([np.prod(list(p.size())) for p in net.parameters()])
        print('Number of params: %d' % s)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Set up ADMM stuff
    size = img_clean_np.shape
    h = size[-2]
    w = size[-1]
    Dh_psf = np.array([ [0, 0, 0], [1, -1, 0], [0, 0, 0]])
    Dv_psf = np.array([ [0, 1, 0], [0, -1, 0], [0, 0, 0]])

    # Dh_DFT = torch.from_numpy(psf2otf(Dh_psf, [h,w]))
    # Dv_DFT = torch.from_numpy(psf2otf(Dv_psf, [h,w]))
    Dh_DFT = torch.from_numpy(psf2otf(Dh_psf, [h,w])).to(device)
    Dv_DFT = torch.from_numpy(psf2otf(Dv_psf, [h,w])).to(device)


    # img_noisy_torch = np_to_torch(y).type(dtype)
    img_noisy_torch = np_to_torch(y).to(device) 
    u = 0*img_noisy_torch.detach().clone()
    t_h = 0*img_noisy_torch.detach().clone()
    t_v = 0*img_noisy_torch.detach().clone()

    mu_t_h = torch.zeros_like(img_noisy_torch, device=device)
    mu_t_v = torch.zeros_like(img_noisy_torch, device=device)

    # Hyperparameters to tune
    beta_t = 25
    weight = 0.01
    inner_iterations = 20 
    
    metrics = []
    loss_values = []
    psnr_values = []
    running_loss=0

    for i in range(num_iter):
        
        # Regularization
        if inner_iterations>1:
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)

        # Update theta
        for j in range(inner_iterations):
            optimizer.zero_grad()
        
            #First problem
            out = net(net_input)
            [Dh_out, Dv_out] = D(out, Dh_DFT, Dv_DFT)

            total_loss = norm2_loss(out-img_noisy_torch)
            total_loss += (beta_t/2)*norm2_loss(Dh_out-(t_h-mu_t_h).detach()) + (beta_t/2)*norm2_loss(Dv_out-(t_v-mu_t_v).detach())
            
            total_loss.backward()
            optimizer.step()
        
        running_loss = total_loss.item()
        loss_values.append(running_loss)

        out = net(net_input)
    
        [Dh_out, Dv_out] = D(out, Dh_DFT, Dv_DFT)

        #TV problem: second problem 
        q_h                 = Dh_out + mu_t_h
        q_v                 = Dv_out + mu_t_v
        q_norm              = torch.sqrt(torch.pow(q_h,2) + torch.pow(q_v,2))
        q_norm[q_norm == 0] = weight/beta_t
        q_norm              = torch.clamp( q_norm - weight/beta_t , min=0 )/q_norm
        t_h                 = (q_norm * q_h).detach().clone()
        t_v                 = (q_norm * q_v).detach().clone()

        # Ascent step: update the dual variables on the same device as the image tensors.
        mu_t_h = (mu_t_h + (Dh_out - t_h)).detach().clone()
        mu_t_v = (mu_t_v + (Dv_out - t_v)).detach().clone()

        psrn_noisy = compare_psnr(y, out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_clean_np, out.detach().cpu().numpy()[0]) 
        psnr_values.append(psrn_gt)
        ssim_gt, _ = ssim(img_clean_np.transpose(1,2,0), out.detach().cpu().numpy()[0].transpose(1,2,0), win_size=7, full=True, data_range = 1.0, channel_axis=2)
        ssim_noisy, _ = ssim(y.transpose(1,2,0), out.detach().cpu().numpy()[0].transpose(1,2,0), win_size=7, full=True, data_range = 1.0, channel_axis=2)
        # ssim_gt_sm, _ = ssim(img_np.transpose(1,2,0), out_avg.detach().cpu().numpy()[0].transpose(1,2,0), win_size=7, full=True, channel_axis=2)
        
        sobel_gt = cv2.Sobel(img_clean_np.transpose(1,2,0), cv2.CV_64F, 1,1, ksize=5)
        sobel_out = cv2.Sobel(out.detach().cpu().numpy()[0].transpose(1,2,0), cv2.CV_64F, 1,1, ksize=5)
        dssim, _ = ssim(sobel_gt, sobel_out, win_size=7, full=True, data_range = 1.0, channel_axis=2)
        
        metrics.append({
            "iteration": i, 
            "PSNR_noisy": psrn_noisy, 
            "PSNR_gt": psrn_gt, 
            # "PSNR_gt_sm": psrn_gt,
            "SSIM_gt": ssim_gt,
            "SSIM_noisy": ssim_noisy,
            "DSSIM": dssim,
            "loss": total_loss.item()  # Include the loss here
        })
        
        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt), '\r', end='')
        
        if  PLOT and ((i % show_every == 0) or (i == num_iter-1)):
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1), y, img_clean_np], factor=figsize, index=i, view=verbose, prefix= "ADMM-DIP", tag=f"sigma{sigma}", tag1=f"image={ind}")
    
        out_dump = [net, net_input]
    return metrics, out_dump      
    
            
        
        
