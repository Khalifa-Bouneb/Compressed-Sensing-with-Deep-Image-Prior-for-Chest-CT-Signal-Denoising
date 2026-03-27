import torch
from torch.fft import fft2, ifft2
import numpy as np
from .hw5_task2 import BSDS300Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from .models import Unet
import skimage.io
from skimage.metrics import structural_similarity as ssim
from .utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .config import *

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

class BlurredBSDS300Dataset(BSDS300Dataset):
    def __init__(self, root='./BSDS300', patch_size=32, split='train', use_patches=True,
                 kernel_size=7, sigma=2, return_kernel=True):
        super(BlurredBSDS300Dataset, self).__init__(root, patch_size, split)

        # trim images to even size
        self.images = self.images[..., :-1, :-1]
        self.kernel_size = kernel_size
        self.return_kernel = return_kernel

        # extract blur kernel (use an MNIST digit)
        self.kernel_dataset = MNIST('./', train=True, download=True,
                                    transform=Compose([Lambda(lambda x: np.array(x)),
                                                       ToTensor(),
                                                       Lambda(lambda x: x / torch.sum(x))]))

        kernels = torch.cat([x[0] for (x, _) in zip(self.kernel_dataset, np.arange(self.images.shape[0]))])
        kernels = torch.nn.functional.interpolate(kernels[:, None, ...], size=2*(kernel_size,))
        kernels = kernels / torch.sum(kernels, dim=(-1, -2), keepdim=True)
        self.kernel = kernels[[0]].repeat(kernels.shape[0], 1, 1, 1)

        # blur the images
        H = psf2otf(self.kernel, self.images.shape)
        self.blurred_images = ifft2(fft2(self.images) * H).real
        self.blurred_patches = self.patchify(self.blurred_images, patch_size)

        # save which blur kernel is used for each image
        self.patch_kernel = self.kernel.repeat(1, len(self.blurred_patches) // len(self.images), 1, 1)
        self.patch_kernel = self.patch_kernel.view(-1, *self.kernel.shape[-2:])

        # reshape kernel
        self.kernel = self.kernel.squeeze()
    
    
    def get_kernel(self, kernel_size, sigma):
        kernel = self.gaussian(kernel_size, sigma)
        kernel_2d = torch.matmul(kernel.unsqueeze(-1), kernel.unsqueeze(-1).t())
        return kernel_2d

    def __getitem__(self, idx):
        out = [self.blurred_images[idx][None, ...].to(device),
               self.images[idx][None, ...].to(device)]
        if self.return_kernel:
            out.append(self.kernel[[idx]].to(device))

        return out


def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)


def psf2otf(psf, shape):
    inshape = psf.shape
    psf = torch.nn.functional.pad(psf, (0, shape[-1] - inshape[-1], 0, shape[-2] - inshape[-2], 0, 0))

    # Circularly shift OTF so that the 'center' of the PSF is [0,0] element of the array
    psf = torch.roll(psf, shifts=(-int(inshape[-1] / 2), -int(inshape[-2] / 2)), dims=(-1, -2))

    # Compute the OTF
    otf = fft2(psf)

    return otf


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out



def wiener_deconv(x, kernel):
    snr = 100  # use this SNR parameter for your results
    H = psf2otf(kernel, x.shape).to(device)
    G = torch.conj(H) * 1/(1/snr + H*torch.conj(H)).to(device)
    return ifft2(fft2(x) * G).real


def load_models():
    model_deblur_denoise = Unet().to(device)
    model_deblur_denoise.load_state_dict(torch.load('pretrained/deblur_denoise.pth', map_location=device))

    model_denoise = Unet().to(device)
    model_denoise.load_state_dict(torch.load('pretrained/denoise.pth', map_location=device))

    return model_deblur_denoise, model_denoise

def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)

def evaluate_model(dataset_in, sigma_in=0.01):

    # create the dataset
    dataset = dataset_in

    # load the models
    model_deblur_denoise, model_denoise = load_models()

    # put into evaluation mode
    model_deblur_denoise.eval()
    model_denoise.eval()

    for sigma in [sigma_in]:
        psnrs_wiener = []
        ssim_weiner = []
        psnrs_deblur_denoise = []
        ssim_deblur_denoise = []
        psnrs_denoise = []
        ssim_denoise = []
        idx = 0
        for image, gt, kernel in dataset:
            # noisy_image = image + sigma * torch.randn_like(image)
            noisy_image=image

            # Apply Wiener deconvolution
            wiener_deconvolved = wiener_deconv(noisy_image, kernel)

            # Apply the neural network models
            #denoised_image = model_denoise(noisy_image)
            deblurred_denoised_image = model_deblur_denoise(noisy_image)
            
            # Hybrid approach
            wiener_then_denoised_image = model_denoise(wiener_deconvolved)

            # Calculate PSNR
            psnr_wiener = calc_psnr(wiener_deconvolved, gt)
            psnr_deblur_denoise = calc_psnr(deblurred_denoised_image, gt)
            psnr_denoise = calc_psnr(wiener_then_denoised_image, gt)
            
            # breakpoint()
            ssim_weiner.append(ssim(wiener_deconvolved.detach().numpy().squeeze(0).transpose(1,2,0), gt.detach().numpy().squeeze(0).transpose(1,2,0), full=True, channel_axis=2)[0])
            ssim_deblur_denoise.append(ssim(deblurred_denoised_image.detach().numpy().squeeze(0).transpose(1,2,0), gt.detach().numpy().squeeze(0).transpose(1,2,0), full=True, channel_axis=2)[0])
            ssim_denoise.append(ssim(wiener_then_denoised_image.detach().numpy().squeeze(0).transpose(1,2,0), gt.detach().numpy().squeeze(0).transpose(1,2,0), full=True, channel_axis=2)[0])

            psnrs_wiener.append(psnr_wiener)
            psnrs_deblur_denoise.append(psnr_deblur_denoise)
            psnrs_denoise.append(psnr_denoise)
            #, deblurred_denoised_image.detach().numpy().squeeze(0), wiener_then_denoised_image.detach().numpy().squeeze(0)
            plot_image_grid([noisy_image.detach().numpy().squeeze(0), wiener_deconvolved.detach().numpy().squeeze(0)], index = idx, task="deblur", prefix="task3", suffix="weiner_deconv")
            plot_image_grid([noisy_image.detach().numpy().squeeze(0), deblurred_denoised_image.detach().numpy().squeeze(0)], index = idx, task="deblur", prefix="task3", suffix="deblur_denoise")
            plot_image_grid([noisy_image.detach().numpy().squeeze(0), wiener_then_denoised_image.detach().numpy().squeeze(0)], index = idx, task="deblur", prefix="task3", suffix="denoise")

            skimage.io.imsave(f'wiener_sigma_{sigma}.png', (img_to_numpy(wiener_deconvolved)*255).astype(np.uint8))
            skimage.io.imsave(f'deblur_denoise_sigma_{sigma}.png', (img_to_numpy(deblurred_denoised_image)*255).astype(np.uint8))
            skimage.io.imsave(f'denoise_sigma_{sigma}.png', (img_to_numpy(wiener_then_denoised_image)*255).astype(np.uint8))         

            idx += 1
        return [psnrs_wiener, psnrs_deblur_denoise, psnrs_denoise, ssim_weiner, ssim_deblur_denoise, ssim_denoise]
        # print(f'Average PSNR for Wiener deconvolution at sigma {sigma}: {np.mean(psnrs_wiener)}')
        # print(f'Average PSNR for Deblur+Denoise model at sigma {sigma}: {np.mean(psnrs_deblur_denoise)}')
        # print(f'Average PSNR for Denoise model at sigma {sigma}: {np.mean(psnrs_denoise)}')        
            
            

def run_hw5_task3(dataset_in, sigma_in=0.01):
    metrics_l = evaluate_model(dataset_in, sigma_in=sigma_in)        
    return metrics_l
