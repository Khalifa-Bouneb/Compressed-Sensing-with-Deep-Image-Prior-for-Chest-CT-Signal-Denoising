import torch
from torch.fft import fft2, ifft2
import numpy as np
from torch.utils.data import Dataset
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

def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)

class BSDS300Dataset(Dataset):
    def __init__(self, root='./Dataset/BSDS300/BSDS300', patch_size=32, use_patches=True):
        files = self._resolve_image_files(root)
        
        self.use_patches = use_patches
        self.images = self.load_images(files)
        self.patches = self.patchify(self.images, patch_size)
        self.mean = torch.mean(self.patches)
        self.std = torch.std(self.patches)

    def _resolve_image_files(self, root, split):
        image_root = os.path.join(root, 'images')
        candidates = []

        if split is not None:
            candidates.append(os.path.join(image_root, split, '*'))

        candidates.append(os.path.join(image_root, '*'))

        files = []
        for pattern in candidates:
            files = sorted(fname for fname in glob(pattern) if os.path.isfile(fname))
            if files:
                return files

        searched = ", ".join(candidates)
        raise FileNotFoundError(
            f"No image files were found for BSDS300Dataset. "
            f"Searched: {searched}. Check the root path: {root}"
        )

    def load_images(self, files):
        out = []
        for fname in files:
            img = skimage.io.imread(fname)
            if img.shape[0] > img.shape[1]:
                img = img.transpose(1, 0, 2)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.
            out.append(torch.from_numpy(img))
        return torch.stack(out)
    
    def patchify(self, img_array, patch_size):
        # create patches from image array of size (N_images, 3, rows, cols)
        patches = img_array.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(patches.shape[0], 3, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, 3, patch_size, patch_size)
        return patches

    def __len__(self):
        if self.use_patches:
            return self.patches.shape[0]
        else:
            return self.images.shape[0]

    def __getitem__(self, idx):
        if self.use_patches:
            return self.patches[idx]
        else:
            return self.images[idx]

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
