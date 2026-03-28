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
from .config import *
from .utils import *
from skimage.metrics import structural_similarity as ssim

# set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

matplotlib.rcParams['figure.raise_window'] = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class DnCNN(nn.Module):
    """
    Network architecture from this reference. Note that we omit batch norm
    since we are using a shallow network to speed up training times.

    @article{zhang2017beyond,
      title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
      author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
      journal={IEEE Transactions on Image Processing},
      year={2017},
      volume={26},
      number={7},
      pages={3142-3155},
    }
    """

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, kernel_size=3,
                 hidden_layers=3, use_bias=True):
        super(DnCNN, self).__init__()

        self.use_bias = use_bias

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
        layers.append(torch.nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            layers.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='same', bias=use_bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)


def add_noise(x, sigma=0.1):
    return x + torch.randn_like(x) * sigma


def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out


def plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
                 val_losses, val_psnrs, val_iters, train_dataset,
                 val_dataset, val_dataloader):

    with torch.no_grad():
        model.eval()

        # evaluate on training dataset sample
        train_dataset.use_patches = False
        train_image = train_dataset[0][None, ...].to(device)
        train_dataset.use_patches = True

        noisy_train_image = add_noise(train_image, sigma=sigma)
        denoised_train_image = model(noisy_train_image)

        # evaluate on validation dataset sample
        val_dataset.use_patches = False
        val_image = val_dataset[6][None, ...].to(device)
        val_dataset.use_patches = True
        
        #val_patch_samples = iter(val_dataloader).next().to(device)
        
        for val_patch_samples in val_dataloader:
            val_patch_samples = val_patch_samples.to(device)
            break

        # calculate validation metrics
        noisy_val_patch_samples = add_noise(val_patch_samples, sigma=sigma)
        denoised_val_patch_samples = model(noisy_val_patch_samples)
        val_loss = torch.mean((val_patch_samples - denoised_val_patch_samples)**2)
        val_psnr = calc_psnr(denoised_val_patch_samples, val_patch_samples)

        val_losses.append(val_loss.item())
        val_psnrs.append(val_psnr)
        val_iters.append(idx)

        noisy_val_image = add_noise(val_image, sigma=sigma)
        denoised_val_image = model(noisy_val_image)

    plt.clf()
    plt.subplot(241)
    plt.plot(losses, label='Train loss')
    plt.plot(val_iters, val_losses, '.', label='Val. loss')
    plt.yscale('log')
    plt.legend()
    plt.title('loss')

    plt.subplot(245)
    plt.plot(psnrs, label='Train PSNR')
    plt.plot(val_iters, val_psnrs, '.', label='Val. PSNR')
    plt.plot(baseline_psnrs, label='Baseline PSNR')
    plt.ylim((0, 32))
    plt.legend()
    plt.title('psnr')

    plt.subplot(242)
    plt.imshow(img_to_numpy(train_image))
    plt.ylabel('Training Set')
    plt.title('GT')

    plt.subplot(243)
    plt.imshow(img_to_numpy(noisy_train_image))
    plt.title('Noisy Image')

    plt.subplot(244)
    plt.imshow(img_to_numpy(denoised_train_image))
    plt.title('Denoised Image')

    plt.subplot(246)
    plt.imshow(img_to_numpy(val_image))
    plt.ylabel('Validation Set')
    plt.title('GT')

    plt.subplot(247)
    plt.imshow(img_to_numpy(noisy_val_image))
    plt.title('Noisy Image')

    plt.subplot(248)
    plt.imshow(img_to_numpy(denoised_val_image))
    plt.title('Denoised Image')

    plt.tight_layout()
    plt.pause(0.1)


def train(sigma=0.1, use_bias=True, hidden_channels=32, epochs=2, batch_size=32, plot_every=200):

    print(f'==> Training on noise level {sigma:.02f} | use_bias: {use_bias} | hidden_channels: {hidden_channels}')

    # create datasets
    train_dataset = BSDS300Dataset(patch_size=32, split='train', use_patches=True)
    val_dataset = BSDS300Dataset(patch_size=32, split='test', use_patches=True)

    # create dataloaders & seed for reproducibility
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DnCNN(use_bias=use_bias, hidden_channels=hidden_channels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    psnrs = []
    baseline_psnrs = []
    val_losses = []
    val_psnrs = []
    val_iters = []
    idx = 0

    pbar = tqdm(total=len(train_dataset) * epochs // batch_size)
    for epoch in range(epochs):
        for sample in train_dataloader:

            model.train()
            sample = sample.to(device)

            # add noise
            noisy_sample = add_noise(sample, sigma=sigma)

            # denoise
            denoised_sample = model(noisy_sample)

            # loss function
            loss = torch.mean((denoised_sample - sample)**2)
            psnr = calc_psnr(denoised_sample, sample)
            baseline_psnr = calc_psnr(noisy_sample, sample)

            losses.append(loss.item())
            psnrs.append(psnr)
            baseline_psnrs.append(baseline_psnr)

            # update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            # plot results
            # if not idx % plot_every:
            #     # # plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
            #     #              val_losses, val_psnrs, val_iters, train_dataset,
            #     #              val_dataset, val_dataloader)

            # idx += 1
            pbar.update(1)

    pbar.close()
    return model


def evaluate_model(model, sigma=0.1, output_filename='out.png', comp_mode=False, subset = None):
    dataset = BSDS300Dataset(patch_size=32, split='test', use_patches=False)
    
    if comp_mode:
        dataset = subset
        
    model.eval()

    psnrs = []
    ssims = []
    breakpoint()
    for idx, image in enumerate(dataset):
        print(idx)
        breakpoint()
        image = image[None, ...].to(device)  # add batch dimension
        noisy_image = add_speckle(image, sigma)
        denoised_image = model(noisy_image)
        psnr = calc_psnr(denoised_image, image)
        psnrs.append(psnr)
        
        # breakpoint()
        ssim_out = ssim(img_to_numpy(denoised_image), img_to_numpy(image), channel_axis=2)
        ssims.append(ssim_out)
        
        # include the tiger image in your homework writeup
        if comp_mode:
            plot_image_grid([img_to_numpy(image).transpose(2,1,0), img_to_numpy(noisy_image).transpose(2,1,0), img_to_numpy(denoised_image).transpose(2,1,0)], prefix = 'DCNN', index=idx+1, task = 'denoise', suffix=f"SPECKLE_sigma={sigma:.02f}")
            skimage.io.imsave(output_filename, (img_to_numpy(denoised_image)*255).astype(np.uint8))

    return np.mean(psnrs), psnrs, ssims, model


from .config import *

def run_hw5(subset):
    noise_levels = [0.1]
    metrics = []
    
    model_configs = [
        {'use_bias': DCNN_PARAMS['use_bias'], 'hidden_channels': DCNN_PARAMS['hidden_channels']}
    ]
    
    psnr_results = {sigma: {str(config): None for config in model_configs} for sigma in noise_levels}
    ssim_results = {sigma: {str(config): None for config in model_configs} for sigma in noise_levels}
    
    for config in model_configs:
        for sigma in noise_levels:
            model = train(use_bias=config['use_bias'], hidden_channels=config['hidden_channels'])
            avg_psnr, psnrs, ssim, model_out = evaluate_model(model, sigma=sigma, output_filename=f'out_{sigma}_{str(config)}.png', comp_mode=True, subset = subset)
            psnr_results[sigma][str(config)] = avg_psnr
            ssim_results[sigma][str(config)] = np.mean(ssim)
            
            print(f"==> Model {config} | Noise level {sigma:.02f} | PSNRs: {psnrs}")
            print(f"==> Model {config} | Noise level {sigma:.02f} | PSNR: {avg_psnr:.02f}")
            
            with open(f'results/denoise/DCNN/metrics_sigma{sigma}.txt', 'a') as f:
                f.write(f"==> Model {config} | Noise level {sigma:.02f} | PSNR: {avg_psnr:.02f}\n")
                f.write(f"==> Model {config} | Noise level {sigma:.02f} | PSNRs: {psnrs}\n")
            
    return [psnr_results, ssim_results], model_out
            


if __name__ == '__main__':
    
    noise_levels = [0.05, 0.1, 0.2]

    model_configs = [
        {'use_bias': True, 'hidden_channels': 32},
        {'use_bias': True, 'hidden_channels': 64},
        {'use_bias': False, 'hidden_channels': 32},
        {'use_bias': False, 'hidden_channels': 64}
    ]
    
    psnr_results = {sigma: {str(config): None for config in model_configs} for sigma in noise_levels}
    
    for config in model_configs:
        for sigma in noise_levels:
            model = train(use_bias=config['use_bias'], hidden_channels=config['hidden_channels'])
            avg_psnr, psnrs = evaluate_model(model, sigma=sigma, output_filename=f'out_{sigma}_{str(config)}.png', comp_mode=True)
            psnr_results[sigma][str(config)] = avg_psnr
            
            print(f"==> Model {config} | Noise level {sigma:.02f} | PSNRs: {psnrs}")
            print(f"==> Model {config} | Noise level {sigma:.02f} | PSNR: {avg_psnr:.02f}")
            
    with open('psnr_results.txt', 'w') as f:
        for sigma, configs in psnr_results.items():
            for config, psnr in configs.items():
                f.write(f'Noise level: {sigma:.02f} | Model: {config} | PSNR: {psnr:.02f}\n')
    
