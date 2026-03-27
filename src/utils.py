import torch
import torch.nn as nn
import torchvision
import sys
import os

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2

def psf2otf(psf, shape):
    inshape = psf.shape
    psf = torch.nn.functional.pad(psf, (0, shape[-1] - inshape[-1], 0, shape[-2] - inshape[-2], 0, 0))

    # Circularly shift OTF so that the 'center' of the PSF is [0,0] element of the array
    psf = torch.roll(psf, shifts=(-int(inshape[-1] / 2), -int(inshape[-2] / 2)), dims=(-1, -2))

    # Compute the OTF
    otf = fft2(psf)

    return otf

def norm2_loss(x):
    '''Calculates euclidean loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
    '''
    vec = torch.pow(x[:,:,:,:], 2)
    
    return torch.sum(vec)

def D(x, Dh_DFT, Dv_DFT):
    x_DFT = torch.fft.fft2(x, dim=(-2,-1))
    Dh_x = torch.fft.ifft2(Dh_DFT*x_DFT, dim=(-2,-1)).real
    Dv_x = torch.fft.ifft2(Dv_DFT*x_DFT, dim=(-2,-1)).real
    return [Dh_x, Dv_x]
    

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        input_image image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input_image image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.all(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf_bak(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow=nrow, padding=0)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos', view = False, index=1, prefix="", suffix="", tag="", tag1="", task="denoise"):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    os.makedirs(f"./results/{task}/{prefix}/", exist_ok=True)
    output_path = f"./results/{task}/{prefix}/image_grid_img_{tag}_{index}_{suffix}_{tag1}.png"

    if images_np[0].shape[0] == 1:
        grid_to_save = np.clip(grid[0], 0, 1)
        Image.fromarray((grid_to_save * 255).astype(np.uint8), mode='L').save(output_path)
    else:
        grid_to_save = np.clip(grid.transpose(1, 2, 0), 0, 1)
        Image.fromarray((grid_to_save * 255).astype(np.uint8)).save(output_path)
    
    if view:
        plt.figure(figsize=(factor * len(images_np), factor))
        if images_np[0].shape[0] == 1:
            plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
        else:
            plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
        plt.axis('off')
        plt.show()

    return grid


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def add_noise(x, sigma=0.1):
    return np.clip(x + torch.randn_like(x) * sigma, 0, 1)

def add_speckle(x, sigma=0.1):
    noise = torch.randn_like(x) * sigma  # Generating Gaussian noise
    noisy_x = x + (x * noise)  # Adding multiplicative noise
    return torch.clamp(noisy_x, 0, 1) 

def downsample_image(img_tensor, factor=4):
    img_np = img_tensor.detach().cpu().numpy()
    height, width, channels = img_np.shape

    # Determine the number of pixels in the downsampled image
    N = height // factor * width // factor

    # Generate random masks
    masks = np.random.rand(N, height, width, channels)

    # Compute downsampled images
    downsampled_images_np = np.sum(masks * img_np, axis=(1, 2))

    # Convert the downsampled images to PyTorch tensor
    downsampled_images_tensor = torch.tensor(downsampled_images_np)

    return downsampled_images_np, downsampled_images_tensor
def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def dip_optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False
        
def get_random_subset(dataset, subset_size=50, custom=False):
    """Selects a random subset of the dataset with a fixed size."""
    # Generate a list of indices based on the dataset size
    all_indices = list(range(len(dataset)))
    
    # Shuffle the indices and take the first 'subset_size' indices
    np.random.shuffle(all_indices)
    subset_indices = all_indices[:subset_size]
    
    # Create a subset dataset using the selected indices
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    if custom:
        all_indices = list(range(100))
        np.random.shuffle(all_indices)
        subset_indices = all_indices[:subset_size]
        print(subset_indices)
        subset = torch.utils.data.Subset(dataset, subset_indices)
    return subset

from PIL import Image
import torch
import numpy as np

def pil_to_np1(img):
    """Convert a PIL Image to a NumPy array."""
    # Ensure the image is in [0, 1] if it was originally uint8
    img_np = np.array(img).astype(np.float32) / 255.0
    if img_np.ndim == 2:  # Grayscale
        img_np = img_np[np.newaxis, ...]  # Add channel dimension
    else:
        img_np = img_np.transpose(2, 0, 1)  # Convert HWC to CHW for RGB
    return img_np

def crop_to_square(image_array):
    """
    Crop the input image array to make it a square.
    """
    height, width = image_array.shape[:2]

    if height == width:
        return image_array  # Already square

    # Calculate the size of the square crop
    height, width = 256, 256
    crop_size = min(height, width)

    # Calculate starting point for cropping
    start_height = (height - crop_size) // 2
    start_width = (width - crop_size) // 2

    # Perform cropping
    cropped_image = image_array[start_height:start_height + crop_size, start_width:start_width + crop_size]

    return cropped_image

def rgb_to_grayscale(image_array):
    """
    Convert RGB image array to grayscale.
    """
    # Convert RGB to grayscale using luminance formula
    grayscale_image = np.mean(image_array, axis=2, keepdims=True)
    return grayscale_image

from skimage.filters import gaussian

def fspecial_gaussian_2d(size, sigma):   #this will create a 2d gaussian filter 
    kernel = np.zeros(tuple(size))
    kernel[size[0]//2, size[1]//2] = 1
    kernel = gaussian(kernel, sigma)
    return kernel/np.sum(kernel)  #no changing in the output image brightness

from scipy.signal import convolve2d

def blur_image(img_np, window_size=5, sigma=1.0):  #this will apply the gaussian filter(blur) into the image 
    """
    Apply Gaussian blurring to the input image tensor.
    """
    filt = fspecial_gaussian_2d((window_size, window_size), sigma)
    blurred_np = convolve2d(img_np, filt, mode='same', boundary='symm')
    return blurred_np

def process_image_tensor(img_tensor, d=32, blur=False):
    # Check if tensor is normalized to [0, 1]
    max_val = img_tensor.max()
    if max_val > 1:
        img_tensor = img_tensor / 255.0  # Assuming the tensor was in range [0, 255]

    # Identify the device of the input tensor (CPU or GPU)
    device = img_tensor.device
    
    # Move tensor to CPU and convert to PIL Image for processing
    # Ensure no overflow by keeping the tensor in [0, 1]
    if len(img_tensor.shape) == 2:
        img_pil = Image.fromarray((img_tensor.numpy() * 255).astype(np.uint8), 'L')
    else:
        img_pil = Image.fromarray((img_tensor.to('cpu').numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    
    # Apply cropping to make dimensions divisible by `d`
    img_cropped = crop_image(img_pil, d=d)
    
    # Convert cropped PIL Image back to tensor
    img_np = pil_to_np1(img_cropped)
    img_tensor_processed = torch.from_numpy(img_np).float().to(device)  # Move processed tensor back to the original device
    
    if blur:
        img_numpy = (img_tensor.numpy().transpose(1,2,0)*255).astype(np.uint8)
        img_np_gray = rgb_to_grayscale(img_numpy)
        img_np_gray = img_np_gray.squeeze(2)
        img_np_crop = crop_to_square(img_np_gray)
        img_tensor_processed = blur_image(img_np_crop)
        tot_img = Image.fromarray((img_tensor_processed*255).astype(np.uint8))
        return tot_img, torch.from_numpy(img_tensor_processed).float().to(device)  # Move processed tensor back to the original device, and return img_tensor_processed
    return img_cropped, img_tensor_processed


def process_subset(subset, d=32, blur=False):
    processed_images_pil = []
    processed_images_tensors = []
    for img_tensor in subset:
        # breakpoint()

        img_pil, img_tensor_processed = process_image_tensor(img_tensor, d=d, blur=blur)
        processed_images_pil.append(img_pil)
        processed_images_tensors.append(img_tensor_processed)
    return processed_images_pil, processed_images_tensors

def plot_metrics(data, small_image, im_num):
    iterations = [entry["iteration"] for entry in data]
    psnr_noisy = [entry["PSNR_noisy"] for entry in data]
    psnr_gt = [entry["PSNR_gt"] for entry in data]
    ssim_gt = [entry["SSIM_gt"] for entry in data]
    dssim = [entry["DSSIM"] for entry in data]
    loss = [entry["loss"] for entry in data]

    # Plot PSNR
    plt.figure()
    # plt.figimage(small_image, xo=10, yo=10, alpha=0.8)  # Embed small image
    plt.plot(iterations, psnr_noisy, label="PSNR_noisy")
    plt.plot(iterations, psnr_gt, label="PSNR_gt")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Iteration")
    plt.legend()
    plt.savefig(f"psnr_v_iteration_{im_num}.png")

    # Plot SSIM
    plt.figure()
    # plt.figimage(small_image, xo=10, yo=10, alpha=0.8)  # Embed small image
    plt.plot(iterations, ssim_gt, label="SSIM_gt")
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    plt.title("SSIM vs Iteration")
    plt.legend()
    plt.savefig(f"ssim_v_iteration_{im_num}.png")

    # Plot DSSIM
    plt.figure()
    # plt.figimage(small_image, xo=10, yo=10, alpha=0.8)  # Embed small image
    plt.plot(iterations, dssim, label="DSSIM")
    plt.xlabel("Iteration")
    plt.ylabel("DSSIM")
    plt.title("DSSIM vs Iteration")
    plt.legend()
    # plt.figimage(small_image, xo=0, yo=0, alpha=0.1)  # Embed small image
    plt.savefig(f"dssim_v_iteration_{im_num}.png")

    # Plot Loss
    plt.figure()
    # plt.figimage(small_image, xo=10, yo=10, alpha=0.8)  # Embed small image
    plt.plot(iterations, loss, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Iteration")
    plt.legend()
    # plt.figimage(small_image, xo=0, yo=0, alpha=0.1)  # Embed small image
    plt.savefig(f"loss_v_iteration_{im_num}.png")
