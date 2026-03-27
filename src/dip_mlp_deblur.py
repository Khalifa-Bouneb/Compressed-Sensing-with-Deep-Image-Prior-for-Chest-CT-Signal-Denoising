"""
Rohan Sanda 2024
Code adapted from Ren et al., 2020, "Neural Blind Deconvolution Using Deep Image Priors"

""" 
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from models_dip import skip
from models_dip import fcn
import cv2
import torch
import pickle
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim_sk

from utils import *
from torch.fft import fft2, ifft2

parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[21, 21], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="datasets/levin/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/levin/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results')
opt = parser.parse_args()
#print(opt)

def psf2otf(psf, shape):
    inshape = psf.shape
    psf = torch.nn.functional.pad(psf, (0, shape[-1] - inshape[-1], 0, shape[-2] - inshape[-2], 0, 0))

    # Circularly shift OTF so that the 'center' of the PSF is [0,0] element of the array
    psf = torch.roll(psf, shifts=(-int(inshape[-1] / 2), -int(inshape[-2] / 2)), dims=(-1, -2))

    # Compute the OTF
    otf = fft2(psf)

    return otf

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.FloatTensor

warnings.filterwarnings("ignore")

files_source = glob.glob(os.path.join(opt.data_path, '*.png'))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# start #image
INPUT = 'noise'
pad = 'reflection'
LR = 0.001
num_iter = opt.num_iter
reg_noise_std = 0.0001

path_to_image = "./eval_data_k13/img_1.png"
path_to_image_clean = "./eval_data_k13/gt_1.png"
imgname = "img_1"

_, imgs = get_image(path_to_image, -1) # load image and convert to np.
_, img_clean = get_image(path_to_image_clean, -1) # load image and convert to np.
y = np_to_torch(imgs).type(dtype)
ind = 1
opt.kernel_size = [13, 13]
img_size = imgs.shape
#print(img_size)
#print(type(img_size))
#print(imgname)

padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
opt.img_size[0], opt.img_size[1] = img_size[1], img_size[2]

kernel_height, kernel_width = opt.kernel_size
pad_height = kernel_height // 2
pad_width = kernel_width // 2


input_depth = 32

net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

net = skip( input_depth, 3,
            num_channels_down = [128, 128, 128, 128, 128],
            num_channels_up   = [128, 128, 128, 128, 128],
            num_channels_skip = [4, 4, 4, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

net = net.type(dtype)

n_k = 200
net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
net_input_kernel.squeeze_()
#print(f"net_input_kernel shape {net_input_kernel.shape}")

net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
net_kernel = net_kernel.type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)

# optimizer
optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.1)  

# initilization inputs
net_input_saved = net_input.detach().clone()
net_input_kernel_saved = net_input_kernel.detach().clone()

metrics = []

for step in (range(num_iter)):

    # input regularization
    net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

    # change the learning rate
    scheduler.step(step)
    optimizer.zero_grad()

    # get the network output
    out_x = net(net_input)
    out_k = net_kernel(net_input_kernel)

    out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])

    H = psf2otf(out_k_m, out_x.shape)
    out_y = ifft2(fft2(out_x) * H).real

    total_loss = mse(out_y,y)         
    #print(f"loss = {total_loss.item()}")

    total_loss.backward()
    optimizer.step()

    psrn_gt    = compare_psnr(img_clean, out_x.detach().cpu().numpy()[0]) 
    psrn_noisy = compare_psnr(imgs, out_x.detach().cpu().numpy()[0])
    ssim_gt, _    = ssim_sk(img_clean.transpose(1,2,0), out_x.detach().cpu().numpy()[0].transpose(1,2,0), win_size=7, data_range= 1.0, full=True, channel_axis=2)
    ssim_noisy, _ = ssim_sk(imgs.transpose(1,2,0), out_x.detach().cpu().numpy()[0].transpose(1,2,0), win_size=7, data_range= 1.0, full=True, channel_axis=2)
    
    sobel_gt = cv2.Sobel(img_clean.transpose(1,2,0), cv2.CV_64F, 1,1, ksize=5)
    sobel_out = cv2.Sobel(out_x.detach().cpu().numpy()[0].transpose(1,2,0), cv2.CV_64F, 1,1, ksize=5)
    dssim, _ = ssim_sk(sobel_gt, sobel_out, win_size=7, full=True, data_range=1.0, channel_axis=2)
    #print(f"PSNR_GT: {psrn_gt}")
    
    metrics.append({
        "iteration": step, 
        "PSNR_noisy": psrn_noisy, 
        "PSNR_gt": psrn_gt, 
        "SSIM_gt": ssim_gt,
        "SSIM_gt_noisy": ssim_noisy,
        "DSSIM": dssim,
        "loss": total_loss.item()  # Include the loss here
    })
    
    print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f SSIM_gt: %f' % (step, total_loss.item(), psrn_noisy, psrn_gt, ssim_gt), '\r', end='')

    if (step+1) % opt.save_frequency == 0:
        #print('Iteration %05d' %(step+1))
        #save_path = os.path.join(opt.save_path, '%s_x.png'%imgname)
        
        with open("results_rohan/deblur/" + f"model_smartDeblur_image={ind}.pkl", "wb") as f:
            pickle.dump(metrics, f)
            
        out_x_np = torch_to_np(out_x)
        plot_image_grid([np.clip(out_x_np, 0, 1), imgs], factor=13, nrow=1, prefix="DIP-MLP", tag=f"kernel={opt.kernel_size[0]}", index = step, tag1=f"image={ind}", task="deblur")
            
