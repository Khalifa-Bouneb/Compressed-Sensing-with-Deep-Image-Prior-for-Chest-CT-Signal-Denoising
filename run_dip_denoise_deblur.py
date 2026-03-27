import torch
import torch.nn as nn
from torch.utils.data import Dataset
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
from src.denoise_dip_tv import *
from src.denoise_dip_tvw import *
from src.baselines import bm3d_denoise
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

from src.hw5_task2 import run_hw5
from src.bm3d_wrapper import bm3d_single
from src.hw5_task3 import *

methods_denoise = ["DCNN", "BM3D", "DIP", "ADMM-DIP", "ADMM-DIPWTV"]
methods_hw_deblur = ["weiner", "deblur_denoise", "denoise"]
methods_deblur = ["task3", "smart_dip_deblur", "dip", "admm_dip"]
                
##############
class BSDS300Dataset(Dataset):
    def __init__(self, root='./Dataset/BSDS300/BSDS300', patch_size=32, use_patches=True):
        files = self._resolve_image_files(root)
        
        self.use_patches = use_patches
        self.images = self.load_images(files)
        self.patches = self.patchify(self.images, patch_size)
        self.mean = torch.mean(self.patches)
        self.std = torch.std(self.patches)

    def _resolve_image_files(self, root, split=None):
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
        
class BlurredBSDS300Dataset(BSDS300Dataset):
    def __init__(self, root='./Dataset/BSDS300/BSDS300', patch_size=32, split=None, use_patches=True,
                 kernel_size=7, sigma=2, return_kernel=True):
        super(BlurredBSDS300Dataset, self).__init__(root=root, patch_size=patch_size, use_patches=use_patches)

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
        
##############

def run_method(dataset, dataset_name="BSDS300", task="denoise", method= "DIP", fsavepath="../results", verbose=False, sigma=0.01):
        if not os.path.exists(fsavepath):
            os.makedirs(fsavepath)
            
        print(f"Running {method} on {dataset_name} for {task}")
            
        if dataset_name == "BSDS300":   
            subset = get_random_subset(dataset, subset_size=1) ####CHANGE THIS TO 1 FOR FINAL RUN
            
            processed_pils, processed_tensors = process_subset(subset)     
            n = len(processed_pils)
            if task == "denoise":
                if not os.path.exists(fsavepath + "/denoise/" + method):
                    os.makedirs(fsavepath + "/denoise/" + method)
                    
                if method == "DCNN":
                    metrics, model = run_hw5(subset)
                    with open(fsavepath + "/denoise/" + method + "/SPECKLE_model_sigma=0.10.pkl", "wb") as f:
                        pickle.dump(model, f)
                        
                    with open(fsavepath + "/denoise/" + method + "/SPECKLE_metrics_sigma=0.10.pkl", "wb") as f:
                        pickle.dump(metrics, f)
                
                else:
                    for i in range(n):
                        img_pil = processed_pils[i]
                        img_tensor = processed_tensors[i]
                        img_np = processed_tensors[i].numpy()
                        
                        sigma = ALL_PARAMS[method]["sigma"]
                        img_noisy_np = add_noise(img_tensor, sigma).numpy()
                        metrics_all = []
                        
                        if method == "BM3D":
                            metrics, output = bm3d_single(img_pil, img_np, img_noisy_np, i+1, verbose=verbose)
                        elif method == "DIP":
                            metrics, output = dip_single(img_pil, img_np, img_noisy_np, i+1, verbose=verbose)
                            with open(fsavepath + "/denoise/" + method + f"/model_image={i+1}.pkl", "wb") as f:
                                pickle.dump(output, f)
                        elif method == "ADMM-DIP":
                            metrics, output = admm_dip_single(img_pil, img_np, img_noisy_np, i+1, verbose=verbose)
                            with open(fsavepath + "/denoise/" + method + f"/model_image={i+1}.pkl", "wb") as f:
                                pickle.dump(output, f)
                        elif method == "ADMM-DIPWTV":
                            metrics, output = admm_dip_wtv_single(img_pil, img_np, img_noisy_np, i+1, verbose=verbose)
                            with open(fsavepath + "/denoise/" + method + f"/model_image={i+1}.pkl", "wb") as f:
                                pickle.dump(output, f)
                        else :
                            assert False
                        
                        metrics_all.append(metrics)
                        
                        # Save metrics
                        os.makedirs(fsavepath + "/denoise/" + method, exist_ok=True)
                        with open(fsavepath + "/denoise/" + method + f"/metrics_image={i+1}.pkl", "wb") as f:
                            pickle.dump(metrics, f)
            elif task == "deblur":
                dataset = BlurredBSDS300Dataset(split='test', kernel_size=7)
                subset = dataset
                idx=0
                for img, gt, kernel in subset:
                    gt_pil = TF.to_pil_image(gt.squeeze(0))
                    gt_pil.save(fsavepath + "/deblur/" + f"gt{idx}.png")
                    img = img + SMART_DEBLUR_DIP_PARAMS["sigma"]*torch.randn_like(img)
                    idx += 1
                if method in methods_hw_deblur:
                    method = "task3"
                    if not os.path.exists(fsavepath + "/deblur/" + method):
                        os.makedirs(fsavepath + "/deblur/" + method)  
                        
                    metrics = run_hw5_task3(subset, sigma_in=sigma)
                    print(metrics)
                    with open(fsavepath + "/deblur/" + method + f"/metrics_image_tot_hw5_task3.pkl", "wb") as f:
                        pickle.dump(metrics, f)
                    # with open(fsavepath + "/deblur/" + method + f"/out_dump_hw5_task3.pkl", "wb") as f:
                    #     pickle.dump(out_dump, f)
                elif method == "SDDP":
                    ind = 0
                    if not os.path.exists(fsavepath + "/deblur/" + method):
                        os.makedirs(fsavepath + "/deblur/" + method)  
                        
                    for img, gt, kernel in subset:
                        size1 = kernel.shape[-1]
                        img_np = img.squeeze(0).numpy()
                        gt_pil = TF.to_pil_image(gt.squeeze(0))
                        gt_np = gt.squeeze(0).numpy()
                        #metrics, out_dump = smart_dip_deblur(gt_pil, gt_np, img_np, size1, ind, fsave=fsavepath + "/deblur/" + method, verbose=verbose)
                        
                        os.makedirs(fsavepath + "/deblur/" + method, exist_ok=True)
                        with open(fsavepath + "/deblur/" + method + f"/metrics_image={ind+1}.pkl", "wb") as f:
                            pickle.dump(metrics, f)
                        
                        ind += 1
                elif method == "DIP":
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
                elif method == "all":
                    idx = 0
                    for img, gt, kernel in subset:
                        idx += 1
                        size1 = kernel.shape[-1]
                        img_np = img.squeeze(0).numpy()
                        gt_pil = TF.to_pil_image(gt.squeeze(0))
                        gt_np = gt.squeeze(0).numpy()
                        
                        for method in methods_deblur:
                            if method == "task3":
                                if not os.path.exists(fsavepath + "/deblur/" + method):
                                    os.makedirs(fsavepath + "/deblur/" + method)  
                                    
                                metrics, out_dump = run_hw5_task3(subset, sigma_in=sigma)
                                print(metrics)
                                with open(fsavepath + "/deblur/" + method + f"/metrics_image_tot_hw5_task3.pkl", "wb") as f:
                                    pickle.dump(metrics, f)
                                with open(fsavepath + "/deblur/" + method + f"/out_dump_hw5_task3.pkl", "wb") as f:
                                    pickle.dump(out_dump, f)
                            else:
                                if not os.path.exists(fsavepath + "/deblur/" + method):
                                    os.makedirs(fsavepath + "/deblur/" + method)  
                                if method == "SDDP":
                                    continue
                                if method == "DIP":
                                    metrics, output = dip_single(gt_pil, gt_np, img_np, idx, deblur=True, verbose=verbose)
                                    with open(fsavepath + "/deblur/" + method + f"/model_image={idx}.pkl", "wb") as f:
                                        pickle.dump(output, f)
                                    with open(fsavepath + "/deblur/" + method + f"/metrics_image={idx}.pkl", "wb") as f:
                                        pickle.dump(metrics, f)
                        
        print(f"Finished running {method} on {dataset_name} for {task}")
            
def run_all_on_dataset(dataset, dataset_name="BSDS300", task="denoise", fsavepath="../results"):
    if not os.path.exists(fsavepath):
        os.makedirs(fsavepath)
    
    if dataset_name == "BSDS300":   
        subset = get_random_subset(dataset, subset_size=5)
        processed_pils, processed_tensors = process_subset(subset)
        n = len(processed_pils)
        
        if task == "denoise":
            for method in methods_denoise:
                run_method(dataset, dataset_name, task, method, fsavepath)
                               
            
