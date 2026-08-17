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
from .admm_cuda import cuda_extension_status, periodic_gradient, shrink_and_update_dual
from .admm_profile import (
    increment_counter,
    profile_iteration_setting,
    profile_region,
    profiled,
)

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
early_stopping = ADMM_DIP_PARAMS.get('early_stopping', False)
early_stopping_patience = ADMM_DIP_PARAMS.get('early_stopping_patience', 0)
early_stopping_min_delta = ADMM_DIP_PARAMS.get(
    'early_stopping_min_delta', 0.0
)
early_stopping_metric = ADMM_DIP_PARAMS.get(
    'early_stopping_metric', 'PSNR_gt'
)

num_iter = ADMM_DIP_PARAMS['num_iter']
input_depth = ADMM_DIP_PARAMS['input_depth']
figsize = ADMM_DIP_PARAMS['figsize']


@profiled("ADMM_DIP_TOTAL")
def admm_dip_single_cuda(img_pil, img_clean_np, y, ind, verbose=False):
    if verbose:
        print(f"ADMM image operators: {cuda_extension_status()}")

    with profile_region("MODEL_INITIALIZATION"):
        net = get_net(input_depth, 'skip', pad,
                        skip_n33d=128,
                        skip_n33u=128,
                        skip_n11=4,
                        num_scales=5,
                        upsample_mode='bilinear')


    # Convert net to the specified dtype
    # net = net.type(dtype)
    with profile_region("CUDA_TRANSFER_MODEL"):
        net = net.to(device)

    # net_input = get_noise(input_depth, INPUT, (img_pil.si
    # ze[1], img_pil.size[0])).type(dtype).detach()
    with profile_region("INPUT_PREPARATION_DIP_NOISE"):
        net_input = get_noise(
            input_depth, INPUT, (img_pil.size[1], img_pil.size[0])
        ).to(device).detach()
    # Compute number of parameters
    if verbose:
        s  = sum([np.prod(list(p.size())) for p in net.parameters()])
        print('Number of params: %d' % s)

    # optimizer
    with profile_region("OPTIMIZER_INITIALIZATION"):
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Set up ADMM variables. periodic_gradient replaces the previous
    # FFT/IFFT derivative with an equivalent direct CUDA stencil when the
    # extension is available (and torch.roll otherwise).
    size = img_clean_np.shape
    h = size[-2]
    w = size[-1]


    # img_noisy_torch = np_to_torch(y).type(dtype)
    with profile_region("ADMM_INITIALIZATION"):
        img_noisy_torch = np_to_torch(y).to(device)
        u = 0*img_noisy_torch.detach().clone()
        t_h = 0*img_noisy_torch.detach().clone()
        t_v = 0*img_noisy_torch.detach().clone()

        mu_t_h = torch.zeros_like(img_noisy_torch, device=device)
        mu_t_v = torch.zeros_like(img_noisy_torch, device=device)

    # Hyperparameters to tune
    beta_t = 25 / 2
    weight = 0.01
    outer_iterations = profile_iteration_setting(
        "DIP_PROFILE_OUTER_ITERATIONS", num_iter
    )
    inner_iterations = profile_iteration_setting(
        "DIP_PROFILE_INNER_ITERATIONS", 1
    )
    lbfgs_max_iter = profile_iteration_setting(
        "DIP_PROFILE_LBFGS_MAX_ITER", 20
    )
    
    metrics = []
    loss_values = []
    psnr_values = []
    running_loss=0
    best_score = -float('inf')
    best_iteration = -1
    best_state_dict = None
    should_stop = False

    for i in range(outer_iterations):
        increment_counter("outer_iterations")
        with profile_region("ADMM_OUTER_ITERATION"):
            if inner_iterations > 1:
                with profile_region("OPTIMIZER_INITIALIZATION_LBFGS"):
                    optimizer = torch.optim.LBFGS(
                        net.parameters(), lr=LR, tolerance_grad=-1,
                        tolerance_change=-1, max_iter=lbfgs_max_iter
                    )

            for j in range(inner_iterations):
                increment_counter("inner_iterations")

                def closure():
                    increment_counter("closure_calls")
                    with profile_region("LBFGS_CLOSURE"):
                        optimizer.zero_grad()
                        with profile_region("DIP_FORWARD"):
                            out = net(net_input)
                        with profile_region("ADMM_DERIVATIVES"):
                            Dh_out, Dv_out = periodic_gradient(out)
                        with profile_region("LOSS_COMPUTATION"):
                            total_loss = norm2_loss(out - img_noisy_torch)
                            total_loss += (beta_t / 2) * norm2_loss(
                                Dh_out - (t_h - mu_t_h).detach()
                            )
                            total_loss += (beta_t / 2) * norm2_loss(
                                Dv_out - (t_v - mu_t_v).detach()
                            )
                        with profile_region("BACKWARD"):
                            total_loss.backward()
                        return total_loss

                increment_counter("optimizer_step_calls")
                with profile_region(
                    "LBFGS_STEP" if inner_iterations > 1 else "ADAM_STEP"
                ):
                    total_loss = optimizer.step(closure)

            running_loss = total_loss.item()
            loss_values.append(running_loss)

            with profile_region("DIP_FORWARD_POST_STEP"):
                out = net(net_input)
            with profile_region("ADMM_DERIVATIVES_POST_STEP"):
                Dh_out, Dv_out = periodic_gradient(out)

            # The current native operator fuses both phases into one kernel.
            with profile_region("TV_SHRINKAGE_AND_DUAL_UPDATE"):
                t_h, t_v, mu_t_h, mu_t_v = shrink_and_update_dual(
                    Dh_out, Dv_out, mu_t_h, mu_t_v, weight / beta_t
                )

            with profile_region("CPU_GPU_TRANSFER_METRICS"):
                psrn_noisy = compare_psnr(y, out.detach().cpu().numpy()[0])
                psrn_gt = compare_psnr(
                    img_clean_np, out.detach().cpu().numpy()[0]
                )
            with profile_region("METRICS"):
                psnr_values.append(psrn_gt)
                ssim_gt, _ = ssim(
                    img_clean_np.transpose(1,2,0),
                    out.detach().cpu().numpy()[0].transpose(1,2,0),
                    win_size=7, full=True, data_range=1.0, channel_axis=2
                )
                ssim_noisy, _ = ssim(
                    y.transpose(1,2,0),
                    out.detach().cpu().numpy()[0].transpose(1,2,0),
                    win_size=7, full=True, data_range=1.0, channel_axis=2
                )
                sobel_gt = cv2.Sobel(
                    img_clean_np.transpose(1,2,0), cv2.CV_64F, 1, 1, ksize=5
                )
                sobel_out = cv2.Sobel(
                    out.detach().cpu().numpy()[0].transpose(1,2,0),
                    cv2.CV_64F, 1, 1, ksize=5
                )
                dssim, _ = ssim(
                    sobel_gt, sobel_out, win_size=7, full=True,
                    data_range=1.0, channel_axis=2
                )

                metrics.append({
                    "iteration": i,
                    "PSNR_noisy": psrn_noisy,
                    "PSNR_gt": psrn_gt,
                    "SSIM_gt": ssim_gt,
                    "SSIM_noisy": ssim_noisy,
                    "DSSIM": dssim,
                    "loss": total_loss.item()
                })

                current_score = metrics[-1].get(early_stopping_metric)
                if current_score is None:
                    raise ValueError(
                        "Unsupported early stopping metric: "
                        f"{early_stopping_metric}"
                    )

                if current_score > best_score + early_stopping_min_delta:
                    best_score = current_score
                    best_iteration = i
                    best_state_dict = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in net.state_dict().items()
                    }
                elif (
                    early_stopping
                    and (i - best_iteration) >= early_stopping_patience
                ):
                    should_stop = True

            print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt), '\r', end='')

            if PLOT and ((i % show_every == 0) or (i == outer_iterations - 1)):
                with profile_region("CPU_GPU_TRANSFER_PLOTTING"):
                    out_np = torch_to_np(out)
                with profile_region("PLOTTING"):
                    plot_image_grid(
                        [np.clip(out_np, 0, 1), y, img_clean_np],
                        factor=figsize, index=i, view=verbose,
                        prefix="ADMM-DIP", tag=f"sigma{sigma}",
                        tag1=f"image={ind}"
                    )

            out_dump = [net, net_input]

            if should_stop:
                print(
                    f"\nEarly stopping at iteration {i}; restoring best "
                    f"checkpoint from iteration {best_iteration} with "
                    f"{early_stopping_metric}={best_score:.6f}."
                )
                break

    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)

    out_dump = [net, net_input]
    return metrics, out_dump      
    
            
        
        
