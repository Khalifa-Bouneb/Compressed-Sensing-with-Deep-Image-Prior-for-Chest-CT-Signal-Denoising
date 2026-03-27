from __future__ import print_function
import matplotlib.pyplot as plt

import os

import numpy as np
from models import *

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *  # this is for adding noise
from utils.utils import *
import random   

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1
PLOT = True
sigma = 20 #70
sigma_ = sigma/255.

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


name = 'butterfly.png'
fname = f"data2/denoising/%s"%(name)

# Add synthetic noise
img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)  
    
# pay attention!
if img_np.shape[0]==2:
  img_np_temp = np.zeros((3,img_np.shape[1], img_np.shape[2]))
  img_np_temp[0,:,:] = img_np[0,:,:]  
  img_np_temp[1,:,:] = img_np[0,:,:]
  img_np_temp[2,:,:] = img_np[0,:,:]
  img_np = img_np_temp
if img_np.shape[0]==4:
  img_np = img_np[:-1,:,:]
    
img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

if PLOT:
    plot_image_grid([img_np, img_noisy_np], 4, 5);

print(img_np.shape)

INPUT = 'noise' 
pad = 'reflection'
OPT_OVER = 'net' 

reg_noise_std = 1./20 

show_every = 5

num_iter = 100 
input_depth = 3 
figsize = 1 
    
    
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128, 
              skip_n33u=128, 
              skip_n11=4, 
              num_scales=5,
              upsample_mode='bilinear').type(dtype)
    
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)


size = img_np.shape
h = size[-2]
w = size[-1]
Dh_psf = np.array([ [0, 0, 0], [1, -1, 0], [0, 0, 0]])
Dv_psf = np.array([ [0, 1, 0], [0, -1, 0], [0, 0, 0]])
Id_psf = np.array([[1]])

Id_DFT = torch.from_numpy(psf2otf(Id_psf, [h,w])).cuda()
Dh_DFT = torch.from_numpy(psf2otf(Dh_psf, [h,w])).cuda()
Dv_DFT = torch.from_numpy(psf2otf(Dv_psf, [h,w])).cuda()

DhT_DFT = torch.conj(Dh_DFT)
DvT_DFT = torch.conj(Dv_DFT)

LR = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype) 
u = 0*img_noisy_torch.detach().clone()
t_h = 0*img_noisy_torch.detach().clone()
t_v = 0*img_noisy_torch.detach().clone()

mu_f = torch.zeros(net_input.shape, device=0)
mu_t_h = torch.zeros(net_input.shape, device=0)
mu_t_v = torch.zeros(net_input.shape, device=0)

beta_t = 10
inner_iterations = 50

loss_values = []
fun_values = []
psnr_values = []
running_loss=0

for i in range(num_iter):
    
    if inner_iterations>1:
      optimizer = torch.optim.Adam(net.parameters(), lr=LR)

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
    weight              = torch.div(torch.pow(torch.norm(out-img_noisy_torch),2)/(6*h*w),q_norm)
    weight              = weight.detach().clone()
    q_norm[q_norm == 0] = weight[q_norm == 0]/beta_t
    q_norm              = torch.clamp( q_norm - weight/beta_t , min=0 )/q_norm
    t_h                 = (q_norm * q_h).detach().clone()
    t_v                 = (q_norm * q_v).detach().clone()

    #Ascent step: updating lagrangian parameter
    mu_t_h = (mu_t_h + (Dh_out- t_h)).detach().clone()
    mu_t_v = (mu_t_v + (Dv_out- t_v)).detach().clone()

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
    psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
    psnr_values.append(psrn_gt)
    
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=5)


out_np = torch_to_np(net(net_input))
plot_image_grid([np.clip(out_np, 0, 1), img_np,img_noisy_np], factor=10)


plt.plot(psnr_values)
plt.title(f"Best PSNR:%.2f - Last PSNR:%.2f"%(np.max(psnr_values), psnr_values[-1]))
plt.show()

plt.plot(loss_values)
plt.title("Running loss")
plt.show()


from PIL import Image

#Saving output image

folder = "ADMM_DIPWTV"
path = f"results/%s/%s_%s_%d"%(folder, folder, name[:-4], sigma)

out_np=255*out_np
print(np.amax(out_np))
u=out_np.transpose(1, 2, 0)
np.savetxt(f'%s.txt'%path,u[:,:,0])

u=u.astype(np.uint8)
print(u.shape)

im = Image.fromarray(u)
im.save(f'%s.png'%path)


#Saving output PSNR
folder = "ADMM_DIPWTV"
file_name = "ADMM_DIPWTV_psnr"
path = f"results/%s/%s_%s_%d"%(folder, file_name, name[:-4], sigma)
np.savetxt(f'%s.txt'%path,psnr_values)

#Saving noisy image
folder = "ADMM_DIPWTV"
file_name = "ADMM_DIPWTV_noisy"
path = f"results/%s/%s_%s_%d"%(folder, file_name, name[:-4], sigma)

img_noisy_np=255*img_noisy_np
u=img_noisy_np.transpose(1, 2, 0)

u=u.astype(np.uint8)
print(u.shape)

im = Image.fromarray(u)
im.save(f'%s.png'%path)