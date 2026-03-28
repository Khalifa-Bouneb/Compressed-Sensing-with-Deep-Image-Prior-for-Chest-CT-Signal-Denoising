"""
    this code implements a DIP(vanilla DIP) methode to denoise and deblur an image 
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio 
from skimage.metrics import structural_similarity 
import cv2
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# set random seeds
torch.manual_seed(1)
np.random.seed(1)

torch.use_deterministic_algorithms(True)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from src.config import *
from src.models_dip import *
from src.utils import *

# Load in config settings
imsize = DIP_PARAMS['imsize']
PLOT = DIP_PARAMS['PLOT']
sigma = DIP_PARAMS['sigma']
INPUT = DIP_PARAMS['INPUT']
pad = DIP_PARAMS['pad']
OPT_OVER = DIP_PARAMS['OPT_OVER']
reg_noise_std = DIP_PARAMS['reg_noise_std']
LR = DIP_PARAMS['LR']

OPTIMIZER= DIP_PARAMS['OPTIMIZER']
show_every = DIP_PARAMS['show_every']
exp_weight= DIP_PARAMS['exp_weight']
early_stopping = DIP_PARAMS.get('early_stopping', False)
early_stopping_patience = DIP_PARAMS.get('early_stopping_patience', 0)
early_stopping_min_delta = DIP_PARAMS.get('early_stopping_min_delta', 0.0)
early_stopping_metric = DIP_PARAMS.get('early_stopping_metric', 'PSNR_gt')

num_iter = DIP_PARAMS['num_iter']
input_depth = DIP_PARAMS['input_depth']
figsize = DIP_PARAMS['figsize']


def printnn():
    net = get_net(input_depth, 'skip', pad,
                skip_n33d=64, 
                skip_n33u=64, 
                skip_n11=3, 
                num_scales=3,
                n_channels=3,
                upsample_mode='bilinear').to(device)
    return net


def dip_single(img_clean_pil, img_clean_np, y, ind, verbose=False, deblur=False):   #y is the observation (np noisy image) of size 3 x 321 x 481 (np)
    
    if (img_clean_np.shape[0]) == 1:
        n_channels = 1
    else:
        n_channels = 3
    
    if deblur == True:
        skip_n11 = 16
    else:
        skip_n11 = 4

    img_h, img_w = img_clean_pil.size[1], img_clean_pil.size[0]

    # net = dcgan(
    #         inp=input_depth,
    #         ndf=128,
    #         num_ups=8,                        # enough for your image size
    #         n_channels=n_channels,
    #         need_sigmoid=True,
    #         upsample_mode='bilinear',
    #         need_convT=False,                 # no checkerboard
    #         output_size=(img_h, img_w)        # fix Bug 1
    #     ).to(device)


    net = get_net(input_depth, 'skip', pad,
                skip_n33d=128, 
                skip_n33u=128, 
                skip_n11=skip_n11, 
                num_scales=5,
                n_channels=n_channels,
                upsample_mode='bilinear')
    


    # input_depth = 32
    # Convert net to the specified device
    net = net.to(device)
    # print(net)





    # Random input to the DIP
    # net_input = torch.randn(1, input_depth, 1, 1).to(device).detach()     

    net_input = get_noise(input_depth, 'noise', (img_clean_pil.size[1], img_clean_pil.size[0])).to(device).detach()  # 1 x input_depth x H x W
   
    # Compute number of parameters
    if verbose:
        s  = sum([np.prod(list(p.size())) for p in net.parameters()])
        print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().to(device)

    y_torch = np_to_torch(y).to(device)  #this is the observation

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone() #.clone() is just will use another variable in another space in memory
    Gz = None
    last_net = None
    psrn_noisy_last = 0
    metrics = []
    best_score = -float('inf')
    best_iteration = -1
    best_state_dict = None
    should_stop = False
    i = 0
    def closure():
        
        nonlocal i, Gz, psrn_noisy_last, last_net, net_input, metrics
        nonlocal best_score, best_iteration, best_state_dict, should_stop
        
        
        # Add regularization noise for generalization
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
        G = net(net_input)    #this is ftheta(z)
        Gz = G.detach()  #.detach() will make require_grad = False
                
        # calculate measurement loss || y - ftheta(z) ||^2  y is the noisy image and ftheta(z) is the output of the network
        total_loss = mse(G, y_torch)        
        total_loss.backward()
        
        # breakpoint()  
        psrn_noisy = peak_signal_noise_ratio(y, G.detach().cpu().numpy()[0], data_range=np.max(y))    #peak_signal_noise_ratio(image_true, image_test)
        psrn_gt    = peak_signal_noise_ratio(img_clean_np, G.detach().cpu().numpy()[0], data_range=np.max(y)) 
        if y.shape[0] == 1:
            ssim_gt, _ = structural_similarity(img_clean_np.squeeze(0), G.detach().cpu().numpy()[0].squeeze(0), data_range= 1.0, full=True, channel_axis=0)
        else:
            ssim_gt, _ = structural_similarity(img_clean_np.transpose(1,2,0), G.detach().cpu().numpy()[0].transpose(1,2,0), data_range= 1.0, full=True, channel_axis=2)
        #ssim_gt_noisy, _ = ssim(y.transpose(1,2,0), G.detach().cpu().numpy()[0].transpose(1,2,0), win_size=7, full=True, channel_axis=2)
        
        sobel_gt = cv2.Sobel(img_clean_np, cv2.CV_64F, 1,1, ksize=5)
        sobel_out = cv2.Sobel(G.detach().cpu().numpy()[0], cv2.CV_64F, 1,1, ksize=5)
        dssim, _ = structural_similarity(sobel_gt, sobel_out, win_size=7, full=True, data_range=1.0, channel_axis=0)
        
        metrics.append({
            "iteration": i, 
            "PSNR_noisy": psrn_noisy, 
            "PSNR_gt": psrn_gt, 
            "SSIM_gt": ssim_gt,
            #"SSIM_gt_noisy": ssim_gt_noisy,
            "DSSIM": dssim,
            "loss": total_loss.item()  # Include the loss here
        })
        
        current_score = metrics[-1].get(early_stopping_metric)
        if current_score is None:
            raise ValueError(f"Unsupported early stopping metric: {early_stopping_metric}")

        if current_score > best_score + early_stopping_min_delta:
            best_score = current_score
            best_iteration = i
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in net.state_dict().items()
            }
        elif early_stopping and (i - best_iteration) >= early_stopping_patience:
            should_stop = True
        
        sigma = DIP_PARAMS['sigma']
        print(
            f"Iteration {i:05d}  Loss {total_loss.item():.6f}  PSNR_noisy: {psrn_noisy:.2f}  PSNR_gt: {psrn_gt:.2f}  SSIM_gt: {ssim_gt:.4f}  DSSIM: {dssim:.4f}",
            end='\r')
        if  PLOT and ((i % show_every == 0) or (i == num_iter-1)):
            out_np = torch_to_np(G)
            if deblur:
                task = "deblur"
                sigma = SMART_DEBLUR_DIP_PARAMS['sigma']
            else:
                task = "denoise"
            plot_image_grid([np.clip(out_np, 0, 1), 
                            y, img_clean_np], factor=figsize, nrow=1, prefix="DIP", tag=f"sigma={sigma}", task=task, index=i, view = verbose, tag1=f"image={ind}")
            
        # Backtracking
        if i % show_every:
            if (psrn_noisy - psrn_noisy_last < -5) and i > 2: 
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy
                
        i += 1

        return total_loss

    
    p = get_params(OPT_OVER, net, net_input)
    if OPTIMIZER.lower() != 'adam':
        raise NotImplementedError("Early stopping is implemented for the Adam optimizer in dip_single.")

    optimizer = torch.optim.Adam(p, lr=LR)

    print('Starting optimization with ADAM')
    for _ in range(num_iter):
        optimizer.zero_grad()
        closure()
        optimizer.step()

        if should_stop:
            print(f"\nEarly stopping at iteration {i - 1}; restoring best checkpoint from iteration {best_iteration}.")
            break

    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
    
    out_dump = [net, net_input]
    return metrics, out_dump
