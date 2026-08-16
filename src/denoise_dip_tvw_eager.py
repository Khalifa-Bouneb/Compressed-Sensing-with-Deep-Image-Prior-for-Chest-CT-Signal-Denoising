"""
Adaptive weighted-TV ADMM-DIP denoising integrated with the project runner.
"""

import cv2
import matplotlib
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim

from .config import ADMM_DIPWTV_PARAMS
from .models_dip import get_net
from .utils import D, get_noise, norm2_loss, np_to_torch, plot_image_grid, psf2otf, torch_to_np


torch.manual_seed(1)
np.random.seed(1)

torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


imsize = ADMM_DIPWTV_PARAMS['imsize']
PLOT = ADMM_DIPWTV_PARAMS['PLOT']
sigma = ADMM_DIPWTV_PARAMS['sigma']
INPUT = ADMM_DIPWTV_PARAMS['INPUT']
pad = ADMM_DIPWTV_PARAMS['pad']
OPT_OVER = ADMM_DIPWTV_PARAMS['OPT_OVER']
reg_noise_std = ADMM_DIPWTV_PARAMS['reg_noise_std']
LR = ADMM_DIPWTV_PARAMS['LR']
OPTIMIZER = ADMM_DIPWTV_PARAMS['OPTIMIZER']
show_every = ADMM_DIPWTV_PARAMS['show_every']
num_iter = ADMM_DIPWTV_PARAMS['num_iter']
input_depth = ADMM_DIPWTV_PARAMS['input_depth']
figsize = ADMM_DIPWTV_PARAMS['figsize']
beta_t = ADMM_DIPWTV_PARAMS['beta_t']
inner_iterations = ADMM_DIPWTV_PARAMS['inner_iterations']


def admm_dip_wtv_single_eager(img_pil, img_clean_np, y, ind, verbose=False):
    n_channels = 1 if img_clean_np.shape[0] == 1 else 3

    net = get_net(
        input_depth,
        'skip',
        pad,
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        n_channels=n_channels,
        upsample_mode='bilinear'
    ).to(device)

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).to(device).detach()

    if verbose:
        n_params = sum(np.prod(list(p.size())) for p in net.parameters())
        print('Number of params: %d' % n_params)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    h, w = img_clean_np.shape[-2], img_clean_np.shape[-1]
    dh_psf = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32)
    dv_psf = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)

    dh_dft = torch.from_numpy(psf2otf(dh_psf, [h, w])).to(device)
    dv_dft = torch.from_numpy(psf2otf(dv_psf, [h, w])).to(device)

    img_noisy_torch = np_to_torch(y).to(device=device, dtype=torch.float32)
    t_h = torch.zeros_like(img_noisy_torch)
    t_v = torch.zeros_like(img_noisy_torch)
    mu_t_h = torch.zeros_like(img_noisy_torch)
    mu_t_v = torch.zeros_like(img_noisy_torch)

    metrics = []

    for i in range(num_iter):
        if inner_iterations > 1:
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)

        for _ in range(inner_iterations):
            optimizer.zero_grad()

            out = net(net_input)
            dh_out, dv_out = D(out, dh_dft, dv_dft)

            total_loss = norm2_loss(out - img_noisy_torch)
            total_loss += (beta_t / 2) * norm2_loss(dh_out - (t_h - mu_t_h).detach())
            total_loss += (beta_t / 2) * norm2_loss(dv_out - (t_v - mu_t_v).detach())
            total_loss.backward()
            optimizer.step()

        out = net(net_input)
        dh_out, dv_out = D(out, dh_dft, dv_dft)

        q_h = dh_out + mu_t_h
        q_v = dv_out + mu_t_v
        q_norm = torch.sqrt(torch.pow(q_h, 2) + torch.pow(q_v, 2))
        weight = torch.pow(torch.norm(out - img_noisy_torch), 2) / (6 * h * w)
        weight = (weight / torch.clamp(q_norm, min=1e-12)).detach().clone()
        q_norm[q_norm == 0] = weight[q_norm == 0] / beta_t
        q_norm = torch.clamp(q_norm - weight / beta_t, min=0) / torch.clamp(q_norm, min=1e-12)

        t_h = (q_norm * q_h).detach().clone()
        t_v = (q_norm * q_v).detach().clone()

        mu_t_h = (mu_t_h + (dh_out - t_h)).detach().clone()
        mu_t_v = (mu_t_v + (dv_out - t_v)).detach().clone()

        out_np = out.detach().cpu().numpy()[0]
        psnr_noisy = compare_psnr(y, out_np)
        psnr_gt = compare_psnr(img_clean_np, out_np)

        if n_channels == 1:
            ssim_gt, _ = ssim(img_clean_np.squeeze(0), out_np.squeeze(0), win_size=7, full=True, data_range=1.0, channel_axis=None)
            ssim_noisy, _ = ssim(y.squeeze(0), out_np.squeeze(0), win_size=7, full=True, data_range=1.0, channel_axis=None)
            sobel_gt = cv2.Sobel(img_clean_np.squeeze(0), cv2.CV_64F, 1, 1, ksize=5)
            sobel_out = cv2.Sobel(out_np.squeeze(0), cv2.CV_64F, 1, 1, ksize=5)
            dssim, _ = ssim(sobel_gt, sobel_out, win_size=7, full=True, data_range=1.0, channel_axis=None)
        else:
            ssim_gt, _ = ssim(img_clean_np.transpose(1, 2, 0), out_np.transpose(1, 2, 0), win_size=7, full=True, data_range=1.0, channel_axis=2)
            ssim_noisy, _ = ssim(y.transpose(1, 2, 0), out_np.transpose(1, 2, 0), win_size=7, full=True, data_range=1.0, channel_axis=2)
            sobel_gt = cv2.Sobel(img_clean_np.transpose(1, 2, 0), cv2.CV_64F, 1, 1, ksize=5)
            sobel_out = cv2.Sobel(out_np.transpose(1, 2, 0), cv2.CV_64F, 1, 1, ksize=5)
            dssim, _ = ssim(sobel_gt, sobel_out, win_size=7, full=True, data_range=1.0, channel_axis=2)

        metrics.append({
            "iteration": i,
            "PSNR_noisy": psnr_noisy,
            "PSNR_gt": psnr_gt,
            "SSIM_gt": ssim_gt,
            "SSIM_noisy": ssim_noisy,
            "DSSIM": dssim,
            "loss": total_loss.item()
        })

        print(
            'Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f'
            % (i, total_loss.item(), psnr_noisy, psnr_gt),
            '\r',
            end=''
        )

        if PLOT and ((i % show_every == 0) or (i == num_iter - 1)):
            plot_image_grid(
                [np.clip(out_np, 0, 1), y, img_clean_np],
                factor=figsize,
                index=i,
                view=verbose,
                prefix="ADMM-DIPWTV",
                tag=f"sigma{sigma}",
                tag1=f"image={ind}"
            )

    out_dump = [net, net_input]
    return metrics, out_dump