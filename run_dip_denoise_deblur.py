import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import pickle

# set random seeds
torch.manual_seed(1)
np.random.seed(1)

torch.use_deterministic_algorithms(True)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from src.config import *
from src.models_dip import *
from src.utils import *
from src.denoise_deblur_dip import *
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

####################################################################
class BSDS300Dataset(Dataset):
    def __init__(self, root='./Dataset/BSDS300/BSDS300', patch_size=32, split='train', use_patches=True):
        files = sorted(glob(os.path.join(root, 'images', split, '*')))
        
        self.use_patches = use_patches
        self.images = self.load_images(files)
        self.patches = self.patchify(self.images, patch_size)
        self.mean = torch.mean(self.patches)
        self.std = torch.std(self.patches)

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
        
class BlurredBSDS300Dataset(BSDS300Dataset):
    def __init__(self, root='./Dataset/BSDS300/BSDS300', patch_size=32, split='train', use_patches=True,
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
        out = [self.blurred_images[idx][None, ...],
               self.images[idx][None, ...]]
        if self.return_kernel:
            out.append(self.kernel[[idx]])

        return out
        
####################################################################

def run_method(dataset, dataset_name="BSDS300", task="denoise", method= "DIP", fsavepath="./results", verbose=False, sigma=0.01):
        if not os.path.exists(fsavepath):
            os.makedirs(fsavepath)
            
        print(f"Running {method} on {dataset_name} for {task}")
            
        if dataset_name == "BSDS300":   
            subset = get_random_subset(dataset, subset_size=1)
            
            processed_pils, processed_tensors = process_subset(subset)
            n = len(processed_pils)
            print(n)
            if task == "denoise":
                if not os.path.exists(fsavepath + "/denoise/" + method):
                    os.makedirs(fsavepath + "/denoise/" + method)
                
            
                for i in range(n):
                    img_pil = processed_pils[i]
                    img_tensor = processed_tensors[i]
                    img_np = processed_tensors[i].numpy()
                    
                    sigma = ALL_PARAMS[method]["sigma"]
                    img_noisy_np = add_noise(img_tensor, sigma).numpy()    
                    metrics_all = []
                    
                    if method == "DIP":
                        metrics, output = dip_single(img_pil, img_np, img_noisy_np, i+1, verbose=verbose)
                        with open(fsavepath + "/denoise/" + method + f"/model_image={i+1}.pkl", "wb") as f:
                            pickle.dump(output, f)
                    
                    else:
                        assert False
                    
                    metrics_all.append(metrics)
                    
                    # Save metrics
                    os.makedirs(fsavepath + "/denoise/" + method, exist_ok=True)
                    with open(fsavepath + "/denoise/" + method + f"/metrics_image={i+1}.pkl", "wb") as f:
                        pickle.dump(metrics, f)

            elif task == "deblur":
                dataset = BlurredBSDS300Dataset(split='test', kernel_size=7)
                subset = dataset
                os.makedirs(fsavepath + "/deblur/" + method, exist_ok=True)
                idx=0
                for img, gt, kernel in subset:
                    gt_pil = TF.to_pil_image(gt.squeeze(0))
                    gt_pil.save(fsavepath + "/deblur/" + f"gt{idx}.png")
                    img = img + SMART_DEBLUR_DIP_PARAMS["sigma"]*torch.randn_like(img)
                    idx += 1
             
                if method == "DIP":
                    idx = 0
                    for img, gt, kernel in subset:
                        idx += 1
                        size1 = kernel.shape[-1]
                        img_np = img.squeeze(0).numpy()
                        gt_pil = TF.to_pil_image(gt.squeeze(0))
                        gt_np = gt.squeeze(0).numpy()
                        metrics, output = dip_single(gt_pil, gt_np, img_np, idx, deblur=True, verbose=verbose)
                        with open(fsavepath + "/deblur/" + method + f"/model_image={idx}.pkl", "wb") as f:
                            pickle.dump(output, f)
                        with open(fsavepath + "/deblur/" + method + f"/metrics_image={idx}.pkl", "wb") as f:
                            pickle.dump(metrics, f)
            
        print(f"Finished running {method} on {dataset_name} for {task}")

if __name__ == "__main__":
    dataset = BSDS300Dataset(split="test", use_patches=False)
    run_method(dataset, task="deblur", method="DIP")
