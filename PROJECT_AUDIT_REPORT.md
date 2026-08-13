# Full Project Audit

## Executive verdict

The repository currently contains a reasonably coherent implementation of:

- Classical Gaussian denoising baselines.
- Vanilla Deep Image Prior for single-image denoising.
- DIP combined with fixed and adaptive total variation.
- Supporting CNN/U-Net architectures.

However, several advertised capabilities are not yet valid end to end:

- The `DIP` deblurring branch does not mathematically perform deblurring.
- The supervised DnCNN path currently stops at breakpoints and has a dataset-constructor error.
- The pretrained U-Net deblurring path cannot run because its weights are absent.
- The blind DIP deconvolution script is a standalone prototype with missing data and incompatible imports.
- The repository does not currently implement compressed sensing or chest CT reconstruction despite its title.
- Different methods use different corruption models, samples, and evaluation conventions, so current benchmark comparisons are not scientifically fair.

The vanilla DIP denoising path is the strongest working core. The OpenCV image-layout error in that path has been fixed.

---

## 1. Repository structure

The project contains approximately 5,200 Python lines and 470 notebook code lines.

| Component | Purpose | Current state |
|---|---|---|
| `run_dip_denoise_deblur.py` | Main dataset and method dispatcher | Partially functional |
| `src/denoise_deblur_dip.py` | Vanilla DIP | Denoising works; deblurring objective is incorrect |
| `src/denoise_dip_tv.py` | DIP + TV via ADMM | Mostly coherent, but inefficient and RGB-only |
| `src/denoise_dip_tvw.py` | DIP + adaptive weighted TV | Mostly coherent |
| `src/models_dip/` | DIP generator architectures | Main skip model works; several alternatives contain defects |
| `src/hw5_task2.py` | Supervised DnCNN | Currently non-runnable |
| `src/hw5_task3.py` | Wiener/pretrained U-Net deblurring | Missing pretrained weights |
| `src/dip_mlp_deblur.py` | Blind DIP deconvolution prototype | Standalone and currently non-runnable |
| `src/degrade.py` | Gaussian noise and blur generation | Generally usable |
| `src/baselines.py` | Wiener, BM3D, TV, Richardson-Lucy | Generally usable |
| `src/metrics.py` | PSNR, SSIM, gradient DSSIM, MSE | Standalone definitions are coherent |
| `src/utils.py` | FFT, noise, conversion, plotting | Useful but oversized and inconsistent |
| `notebooks/` | Classical baseline exploration | Dataset paths do not match this checkout |
| `vanilla_dip.ipynb` | DIP experiment launcher | Main practical entry point |

All Python files pass static compilation. There are no automated tests.

---

## 2. Mathematical problem formulation

### 2.1 General inverse problem

The standard imaging model is

$$
y = A x + \eta,
$$

where:

- $x\in\mathbb{R}^{C\times H\times W}$ is the unknown clean image,
- $A$ is the forward degradation operator,
- $\eta$ is noise,
- $y$ is the observed image.

The task changes according to $A$.

### Denoising

$$
A=I,\qquad y=x+\eta.
$$

### Non-blind deblurring

$$
Ax=h*x,\qquad y=h*x+\eta,
$$

where $h$ is a known point-spread function and $*$ denotes convolution.

### Blind deblurring

Both $x$ and $h$ are unknown:

$$
y=h*x+\eta.
$$

### Compressed sensing

A compressed-sensing problem would require a non-square measurement operator such as

$$
y=\Phi x+\eta,\qquad m=\dim(y)\ll n=\dim(x),
$$

or, for CT,

$$
y=\mathcal{R}x+\eta,
$$

where $\mathcal{R}$ is a sparse-view or limited-angle Radon transform.

The current repository does not implement $\Phi$, a Radon transform, sinograms, DICOM/NIfTI loading, or CT reconstruction. It is currently a natural RGB image restoration project using five BSDS images.

---

## 3. Dataset and preprocessing

### 3.1 Actual dataset

The repository contains five images:

- `101085.jpg`
- `101087.jpg`
- `3096.jpg`
- `37073.jpg`
- `8023.jpg`

They are stored in:

```text
Dataset/BSDS300/BSDS300/images/
```

This is not a complete BSDS300 dataset.

The loader rotates portrait images by transposing the spatial axes, making every image $321\times481$, then converts

$$
H\times W\times C \longrightarrow C\times H\times W
$$

and normalizes

$$
x_{\mathrm{float}}=\frac{x_{\mathrm{uint8}}}{255}.
$$

The DIP preprocessing center-crops images to dimensions divisible by 32:

$$
321\times481 \longrightarrow 320\times480.
$$

That is appropriate for a five-scale encoder-decoder because

$$
320/2^5=10,\qquad480/2^5=15.
$$

### 3.2 Patches

For supervised training, non-overlapping $32\times32$ patches are extracted using `unfold`.

For each image, approximately

$$
\left\lfloor \frac{320}{32}\right\rfloor
\left\lfloor \frac{480}{32}\right\rfloor
=10\cdot15=150
$$

patches are available after the effective boundary truncation.

Problems:

- Patches are calculated even when `use_patches=False`.
- Dataset logic is duplicated in three files.
- The code assumes RGB input.
- There is no explicit train/validation/test metadata.
- The five local files provide far too little data for credible supervised DnCNN training.

---

## 4. Noise models

### 4.1 Additive Gaussian noise

The modular degradation code implements

$$
\eta\sim\mathcal{N}(0,\sigma^2I),
\qquad
y=\operatorname{clip}(x+\eta,0,1).
$$

This is the standard synthetic Gaussian denoising setup.

Clipping changes the noise distribution near 0 and 1, so after clipping

$$
\mathbb{E}[y-x]\neq0
$$

for saturated pixels. This is conventional in experiments but should be documented.

### 4.2 Speckle noise

The main DIP runner instead uses multiplicative noise:

$$
n\sim\mathcal{N}(0,\sigma^2I),
\qquad
y=x+x\odot n=x\odot(1+n).
$$

Therefore,

$$
\operatorname{Var}(y_i\mid x_i)=\sigma^2x_i^2.
$$

Bright pixels receive more noise than dark pixels. This is signal-dependent noise, unlike additive white Gaussian noise.

This matters because:

- BM3D is configured as if $\sigma$ describes additive Gaussian noise.
- DnCNN is trained using additive Gaussian noise but evaluated using speckle noise.
- The separate baseline script uses additive Gaussian noise.
- Reported numbers from these pipelines are not directly comparable.

A fair experiment must generate one fixed observation $y_i$ per image and provide that identical array to every method.

---

## 5. Vanilla Deep Image Prior

### 5.1 DIP model

DIP represents an image using an untrained convolutional generator:

$$
x_\theta=f_\theta(z),
$$

where:

- $z$ is fixed random noise,
- $\theta$ contains randomly initialized network weights,
- no dataset-level model training is required.

The code initializes

$$
z_{c,h,w}\sim\mathcal{U}(0,0.1).
$$

At each iteration it perturbs the input:

$$
z_t=z+\xi_t,
\qquad
\xi_t\sim\mathcal{N}(0,\sigma_z^2I),
\qquad
\sigma_z=\frac{1}{40}=0.025.
$$

The regularized objective is approximately

$$
\theta^\star
=
\arg\min_\theta
\mathbb{E}_{\xi}
\left[
\frac{1}{N}
\left\|
f_\theta(z+\xi)-y
\right\|_2^2
\right].
$$

The generator architecture itself provides the prior: convolution, downsampling, upsampling, locality, and skip connections cause natural image structure to be fitted earlier than high-frequency noise.

### 5.2 Architecture

The active generator is the DIP `skip` network:

- 5 encoder scales.
- 128 downsampling features per scale.
- 128 upsampling features per scale.
- 4 skip features for denoising.
- LeakyReLU activations.
- Bilinear upsampling.
- Sigmoid output.

Therefore,

$$
f_\theta(z)\in[0,1]^{C\times H\times W}.
$$

At scale $s$, the network conceptually computes

$$
e_{s+1}
=
\phi\left(
W_{s,2}*
\phi(W_{s,1}*e_s)
\right),
$$

then combines decoder and skip features:

$$
d_s
=
\phi\left(
W_s^{\mathrm{up}}*
\operatorname{concat}(S_s(e_s),\operatorname{up}(d_{s+1}))
\right).
$$

The shape-aware concatenation crops branches to their common central size, which helps handle odd spatial dimensions.

### 5.3 Early stopping

The configured stopping score is ground-truth PSNR:

$$
s_t=\operatorname{PSNR}(x,f_{\theta_t}(z)).
$$

Training stops if the score fails to improve by `0.001` for 20 iterations.

This is an oracle stopping rule because it uses the unknown clean image. It is acceptable for retrospective benchmark analysis, but not for a deployable unsupervised DIP system.

More importantly, the code always records and restores the best ground-truth checkpoint, even when early stopping is disabled. Consequently, final DIP results use test ground truth for model selection.

A valid unsupervised stopping rule should depend only on the observation, for example:

- residual stabilization,
- exponential moving variance,
- SURE for Gaussian noise,
- discrepancy principle,
- self-validation through held-out pixels,
- frequency-based overfitting detection.

---

## 6. The deblurring defect

The current `deblur=True` branch only changes

$$
\text{skip channels}:4\rightarrow16.
$$

It does not alter the loss. It still minimizes

$$
\min_\theta
\left\|
f_\theta(z)-y
\right\|_2^2.
$$

If $y=h*x+\eta$, the model is encouraged to reproduce the blurred image itself. It is not encouraged to produce an image whose blurred version matches $y$.

Correct non-blind DIP deblurring requires

$$
\theta^\star
=
\arg\min_\theta
\left\|
h*f_\theta(z)-y
\right\|_2^2.
$$

Using FFT-based circular convolution:

$$
\widehat{h*x}
=
\mathcal{F}^{-1}
\left[
\mathcal{F}(h)\odot\mathcal{F}(x)
\right].
$$

The dataset already returns a kernel, but `dip_single` never receives or applies it. Therefore the current `task="deblur", method="DIP"` result should not be described as DIP deblurring.

This is the most important mathematical issue in the repository.

---

## 7. DIP + Total Variation through ADMM

### 7.1 TV regularization

For an image $x$, isotropic total variation is

$$
\operatorname{TV}(x)
=
\sum_p
\sqrt{
(D_hx)_p^2+(D_vx)_p^2
},
$$

where $D_h$ and $D_v$ are horizontal and vertical finite differences.

The desired problem is

$$
\min_\theta
\left\|
f_\theta(z)-y
\right\|_2^2
+
\lambda\operatorname{TV}(f_\theta(z)).
$$

Introduce split variables

$$
t_h=D_hf_\theta(z),\qquad
t_v=D_vf_\theta(z).
$$

The constrained form becomes

$$
\min_{\theta,t_h,t_v}
\|f_\theta(z)-y\|_2^2
+
\lambda\sum_p\sqrt{t_{h,p}^2+t_{v,p}^2}
$$

subject to

$$
t_h=D_hf_\theta(z),\qquad
t_v=D_vf_\theta(z).
$$

### 7.2 Network update

The implementation's scaled ADMM network step is

$$
\theta^{k+1}
=
\arg\min_\theta
\|f_\theta(z)-y\|_2^2
+
\frac{\beta}{2}
\|D_hf_\theta(z)-t_h^k+\mu_h^k\|_2^2
+
\frac{\beta}{2}
\|D_vf_\theta(z)-t_v^k+\mu_v^k\|_2^2.
$$

This is structurally correct.

The derivative filters are applied in the Fourier domain:

$$
D_hx
=
\mathcal{F}^{-1}
\left[
\widehat{D_h}\odot\widehat{x}
\right],
$$

and similarly for $D_v$. This implies periodic boundary conditions.

### 7.3 TV proximal update

Define

$$
q_p=
\begin{bmatrix}
(D_hx)_p+\mu_{h,p}\\
(D_vx)_p+\mu_{v,p}
\end{bmatrix}.
$$

The isotropic soft-thresholding step is

$$
t_p
=
\max\left(
1-\frac{\lambda}{\beta\|q_p\|_2},
0
\right)q_p.
$$

The implementation follows this formula.

The dual update is

$$
\mu^{k+1}
=
\mu^k+Dx^{k+1}-t^{k+1}.
$$

That is also correct.

### 7.4 Implementation concerns

The fixed-TV implementation has several weaknesses:

- It defaults to three output channels and does not properly support grayscale.
- It creates an LBFGS optimizer inside every outer ADMM iteration.
- Each call to `LBFGS.step()` can invoke its closure many times.
- The code then calls that step inside another ten-iteration loop.
- Actual network evaluations can therefore greatly exceed the apparent iteration count.
- `reg_noise_std`, `OPTIMIZER`, and several configuration values are loaded but unused.
- The finite-difference FFTs originate from float64 NumPy arrays, producing avoidable complex128 computation.
- Reproducibility is explicitly disabled with `torch.use_deterministic_algorithms(False)`.

---

## 8. Adaptive weighted TV

The weighted method replaces a constant $\lambda$ with pixel-dependent weights:

$$
\operatorname{WTV}(x)
=
\sum_p w_p\|\nabla x_p\|_2.
$$

The code estimates

$$
\rho
=
\frac{\|x-y\|_2^2}{6HW},
$$

then uses approximately

$$
w_p
=
\frac{\rho}{\|\nabla x_p+\mu_p\|_2+\varepsilon}.
$$

Its shrinkage becomes

$$
t_p
=
\max\left(
1-\frac{w_p}{\beta\|q_p\|_2},
0
\right)q_p.
$$

Because $w_p$ is inversely related to gradient strength:

- flat regions receive stronger regularization,
- strong edges receive weaker regularization.

This is the intended qualitative behavior of adaptive TV.

Concerns:

- The denominator $6HW$ implicitly assumes a particular RGB/noise derivation and is not adjusted for grayscale.
- The optimizer is unnecessarily reconstructed every outer iteration.
- Configured input regularization is unused.
- Ground truth is still used for reporting but not for the ADMM objective.

---

## 9. Supervised DnCNN

The DnCNN path learns a residual denoiser. If $R_\psi(y)$ estimates noise, the output is

$$
\hat{x}=y-R_\psi(y).
$$

Training minimizes

$$
\psi^\star
=
\arg\min_\psi
\frac{1}{B}
\sum_{i=1}^{B}
\left\|
y_i-R_\psi(y_i)-x_i
\right\|_2^2.
$$

The architecture is shallow:

- Initial convolution + ReLU.
- Three hidden convolution + ReLU blocks.
- Final RGB convolution.
- Residual subtraction.

This is conceptually valid, but the path cannot currently complete:

1. `BSDS300Dataset.__init__` does not accept `split`, but training calls it with `split='train'` and `split='test'`.
2. Two unconditional `breakpoint()` calls stop evaluation.
3. Training uses additive Gaussian noise.
4. Evaluation uses multiplicative speckle noise.
5. The local dataset has only five images and no real split.
6. The `sigma` passed to `train()` from `run_hw5` is not forwarded explicitly.
7. Output image layout uses `transpose(2,1,0)`, which swaps spatial axes as well as channels.
8. The command-line `__main__` expects a different return signature from `evaluate_model`.

So DnCNN results from this repository are not currently reproducible.

---

## 10. Deblurring dataset and pretrained U-Net path

### 10.1 Synthetic blur

The blurred dataset constructs a PSF from an MNIST digit and applies circular convolution:

$$
y
=
\mathcal{F}^{-1}
\left[
\mathcal{F}(h)\odot\mathcal{F}(x)
\right].
$$

The first MNIST-derived kernel is repeated for every BSDS image.

Issues:

- The constructor parameter `sigma` is unused.
- All images receive the same first MNIST kernel.
- Creating the dataset can trigger an MNIST download.
- Blurred patches are calculated but never returned.
- `__getitem__` always returns full images regardless of `use_patches`.
- This is an unusual blur model and should be clearly distinguished from Gaussian, motion, or CT system blur.

### 10.2 Noise is discarded

The deblur runner contains:

```python
img = img + sigma * torch.randn_like(img)
```

but `img` is only a local loop variable. It does not update the dataset.

Later methods iterate over the original dataset again, so they receive noise-free blurred images.

Thus the intended model

$$
y=h*x+\eta
$$

actually becomes

$$
y=h*x
$$

for the downstream methods.

### 10.3 Pretrained U-Net path

The U-Net branch compares:

1. Wiener deconvolution.
2. A pretrained direct deblur-and-denoise U-Net.
3. Wiener followed by a pretrained denoising U-Net.

The Wiener filter is

$$
\widehat{X}(\omega)
=
\frac{H^\ast(\omega)}
{|H(\omega)|^2+1/\operatorname{SNR}}
Y(\omega),
$$

with fixed $\operatorname{SNR}=100$.

The repository does not contain:

```text
pretrained/deblur_denoise.pth
pretrained/denoise.pth
```

so this path raises `FileNotFoundError`.

The supplied `sigma_in` is also ineffective because noise injection is commented out.

---

## 11. Blind DIP deconvolution prototype

The intended model in `dip_mlp_deblur.py` is mathematically useful:

$$
x_\theta=f_\theta(z_x),
\qquad
h_\phi=g_\phi(z_h),
$$

with a softmax-constrained kernel:

$$
h_{\phi,i}
=
\frac{e^{a_i}}{\sum_j e^{a_j}},
\qquad
h_{\phi,i}\geq0,
\qquad
\sum_i h_{\phi,i}=1.
$$

The blind objective is

$$
(\theta^\star,\phi^\star)
=
\arg\min_{\theta,\phi}
\left\|
h_\phi*f_\theta(z_x)-y
\right\|_2^2.
$$

This is the correct basic blind-DIP formulation.

But the file is not integrated into the package:

- It uses `from models_dip` and `from utils` rather than package-relative imports.
- It parses command-line arguments during import.
- It executes training during import.
- It uses hard-coded missing paths under `eval_data_k13/`.
- It saves into a missing `results_rohan/deblur` directory.
- It always uses CPU tensors.
- It suppresses every warning.
- Its configured data-path file list is calculated but ignored.
- It is never invoked successfully by the main runner.
- The runner's `SDDP` call is commented out and then tries to save an undefined `metrics` variable.

This code should be converted into a callable function before being considered part of the application.

---

## 12. Classical baselines

### 12.1 Wiener

The FFT Wiener estimate is

$$
\widehat{X}
=
\frac{H^\ast}{|H|^2+K}Y,
$$

where $K$ approximates the noise-to-signal ratio.

The modular implementation supports:

- local Wiener denoising when no kernel is provided,
- skimage Wiener deconvolution with a known kernel,
- a custom FFT demonstration.

Problems:

- `noise_var` and `snr` parameters are accepted but unused.
- The custom FFT version converts RGB images to grayscale.
- `restore_image` uses `balance=10\sigma`, which is not generally equivalent to the theoretically expected $\sigma^2/\sigma_x^2$.

### 12.2 BM3D

BM3D approximately performs:

1. Similar-patch search.
2. Grouping into 3-D stacks.
3. Transform-domain shrinkage.
4. Inverse transformation and aggregation.
5. A second Wiener-like collaborative stage.

The modular BM3D wrapper is appropriate for additive white Gaussian noise.

It is not statistically matched to the main runner's speckle noise. Passing the same numerical $\sigma$ does not make those noise models equivalent.

### 12.3 Total variation baseline

The baseline solves approximately

$$
\hat{x}
=
\arg\min_x
\frac{1}{2}\|x-y\|_2^2
+
\lambda\operatorname{TV}(x)
$$

using Chambolle's algorithm.

This is a sound baseline for piecewise-smooth images, though it may produce staircase or cartoon-like artifacts.

### 12.4 Richardson-Lucy

The iterative update is approximately

$$
x^{k+1}
=
x^k
\odot
\left[
h^\dagger*
\frac{y}{h*x^k+\varepsilon}
\right],
$$

where $h^\dagger$ is the flipped kernel.

This is appropriate primarily for Poisson-noise deconvolution and can amplify noise if over-iterated.

---

## 13. Metrics

### 13.1 MSE

$$
\operatorname{MSE}(x,\hat{x})
=
\frac{1}{N}
\sum_{i=1}^{N}(x_i-\hat{x}_i)^2.
$$

Lower is better.

### 13.2 PSNR

$$
\operatorname{PSNR}(x,\hat{x})
=
10\log_{10}
\left(
\frac{L^2}{\operatorname{MSE}(x,\hat{x})}
\right),
$$

where $L=1$ for normalized images.

Higher is better.

The DIP implementation sometimes uses

$$
L=\max(y)
$$

instead of $L=1$. This can make PSNR values incomparable if an observation does not reach exactly 1.

The field named `PSNR_noisy` is also potentially misleading. It measures

$$
\operatorname{PSNR}(y,\hat{x}),
$$

not the quality of the noisy image relative to ground truth.

### 13.3 SSIM

For local means $\mu_x,\mu_y$, standard deviations $\sigma_x,\sigma_y$, and covariance $\sigma_{xy}$,

$$
\operatorname{SSIM}(x,y)
=
\frac{
(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)
}{
(\mu_x^2+\mu_y^2+C_1)
(\sigma_x^2+\sigma_y^2+C_2)
}.
$$

Higher is better.

The standalone metric implementation correctly handles RGB using the last channel axis.

### 13.4 DSSIM inconsistency

The standalone metrics module computes gradient magnitude:

$$
g(x)=\sqrt{(D_hx)^2+(D_vx)^2},
$$

then defines

$$
\operatorname{DSSIM}(x,\hat{x})
=
\frac{1-\operatorname{SSIM}(g(x),g(\hat{x}))}{2}.
$$

Lower is better.

The DIP, BM3D, and ADMM wrappers instead calculate SSIM directly on

```python
cv2.Sobel(image, dx=1, dy=1)
```

and store the result as `DSSIM`.

That quantity is:

- a mixed derivative response, not gradient magnitude,
- still an SSIM value,
- not transformed by $(1-\mathrm{SSIM})/2$,
- higher-is-better despite being labeled lower-is-better,
- evaluated with `data_range=1` even though Sobel responses need not lie in a unit interval.

Therefore the current `DSSIM` plots and comparisons are not trustworthy. All methods should call the same function from `src.metrics`.

---

## 14. Runner and experimental validity problems

### Critical issues

#### 1. DIP deblur does not include the blur operator

The result is not deconvolution.

#### 2. Deblur noise is created and discarded

The observation used later is noise-free.

#### 3. Ground-truth leakage

DIP selects its final checkpoint using ground-truth PSNR.

#### 4. DnCNN is non-runnable

It has invalid constructor arguments and active breakpoints.

#### 5. Pretrained deblur models are absent

The U-Net path cannot start.

#### 6. Blind deblur prototype is disconnected

The `SDDP` branch saves an undefined variable.

### High-priority issues

#### 7. The user-supplied `sigma` is ignored

Inside `run_method`:

$$
\sigma\leftarrow\texttt{ALL\_PARAMS[method]["sigma"]}.
$$

Thus the function argument does not control denoising corruption.

#### 8. `run_all_on_dataset` does not use its selected subset

It selects five samples, discards them, and calls `run_method`, which independently selects one random sample for each method.

As a result, methods may be evaluated on different images.

#### 9. Method names do not agree

The deblur method list contains lowercase names such as `"dip"`, but branches compare against uppercase `"DIP"` and `"SDDP"`.

The `"all"` path therefore skips expected branches.

#### 10. The `"all"` task3 branch has an invalid return assumption

`run_hw5_task3` returns six metric lists, but the runner attempts:

```python
metrics, out_dump = run_hw5_task3(...)
```

which cannot unpack the result into two values.

#### 11. Dataset paths disagree

The actual local path is:

```text
Dataset/BSDS300/BSDS300/images/
```

But the classical notebook and command-line runner expect:

```text
Dataset/BSDS300/images/test/
```

That directory does not exist.

#### 12. Results are split across directories

`run_method(..., fsavepath="../results")` writes serialized outputs outside the repository, while `plot_image_grid` always writes under:

```text
./results/
```

One experiment can therefore produce artifacts in two different result trees.

---

## 15. Model-library findings

### Active skip network

The active skip architecture is coherent and is the best-maintained model in `models_dip`.

### DCGAN

The DCGAN wrapper expects an output size. `get_net(..., NET_TYPE="dcgan")` does not provide one, so its forward pass can attempt interpolation with `size=None`.

### Texture network

The texture-network convolution uses floating-point padding:

```python
padding=(kernel_size - 1) / 2
```

Modern PyTorch expects integer padding. This branch is likely broken.

### DIP U-Net

When `more_layers > 0`, the forward pass accesses `self.more`, which is undefined. The default `more_layers=0` avoids the defect.

### Residual network

A rare shape-correction branch uses `/` when calculating slice indices, producing floats. It should use integer division.

Its custom `eval()` behavior also differs from standard `nn.Module.eval()` semantics in one residual helper.

### Supervised U-Net

The separate U-Net used for pretrained deblurring is structurally sensible, but:

- it returns `out_layer + x`, so input and output channel counts must match,
- its output is not bounded to $[0,1]$,
- it depends entirely on absent checkpoints.

---

## 16. Reproducibility and engineering quality

### Positive aspects

- Seeds are set in most experiment modules.
- Device selection usually supports CPU and CUDA.
- DIP output is bounded through sigmoid.
- FFT-based PSF conversion is implemented.
- Parameters are partly centralized in `config.py`.
- Current Python sources compile.
- The earlier OpenCV channel-layout defect is fixed.

### Problems

#### Conflicting deterministic settings

Some modules call:

```python
torch.use_deterministic_algorithms(True)
```

while ADMM modules call:

```python
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = True
```

Import order can change global behavior.

#### Import-time side effects

Imports:

- reset global random seeds,
- print device names,
- modify Matplotlib state,
- modify deterministic PyTorch state,
- monkey-patch `torch.nn.Module.add`.

Configuration is copied into module-level variables at import, so changing `DIP_PARAMS` after importing a module may not affect execution.

#### CPU fallback defect

DIP rollback explicitly calls `.cuda()`. It fails on CPU-only machines.

The rollback also happens after gradients have already been computed, so returning `total_loss * 0` does not cancel the gradients already stored in parameters.

#### Fragile serialization

Entire model objects and inputs are pickled. Pickles are:

- Python-version dependent,
- sensitive to code relocation,
- potentially unsafe to load from untrusted sources,
- less portable than `state_dict`.

#### Dependency reproducibility

`requirements.txt` specifies lower bounds only, such as `opencv-python>=4.8.0`. There is no lock file or tested version matrix. This allowed OpenCV 5 to be installed despite code written against older behavior.

#### Documentation and tests

- README contains only the project title.
- There is no test suite.
- There are no CLI examples for DIP.
- There is no experiment manifest.
- There is no description of metric conventions.
- There is no record of dataset image IDs used per result.

---

## 17. Relationship to chest CT and compressed sensing

At present, the title overstates the implementation.

The code does not include:

- CT images,
- Hounsfield-unit processing,
- DICOM or NIfTI loading,
- window/level selection,
- sinogram generation,
- Radon or fan-beam projection,
- sparse-view sampling,
- limited-angle masks,
- filtered backprojection,
- CT-specific forward operators,
- data consistency against projection measurements.

The unused `downsample_image` utility is not a valid compressed-sensing pipeline. It constructs large random masks and sums image pixels but is not connected to reconstruction.

To become a CT compressed-sensing project, the core DIP objective should resemble

$$
\theta^\star
=
\arg\min_\theta
\left\|
M\mathcal{R}f_\theta(z)-y
\right\|_2^2
+
\lambda R(f_\theta(z)),
$$

where:

- $\mathcal{R}$ is the CT projection operator,
- $M$ selects measured views,
- $R$ could be TV or weighted TV.

For deblurring alone, use

$$
\theta^\star
=
\arg\min_\theta
\left\|
h*f_\theta(z)-y
\right\|_2^2
+
\lambda R(f_\theta(z)).
$$

---

## 18. Recommended repair order

### Phase 1: Establish a valid benchmark

1. Create one canonical dataset class.
2. Define one observation generator.
3. Store the exact clean image ID, noise seed, noise type, $\sigma$, and kernel.
4. Give every method the same observation.
5. Honor the `sigma` function argument.
6. Use one metrics implementation.
7. Save all outputs under one experiment directory.
8. Evaluate all methods on the same fixed subset.

### Phase 2: Correct deblurring

Change DIP from

$$
\|f_\theta(z)-y\|_2^2
$$

to

$$
\|h*f_\theta(z)-y\|_2^2.
$$

Pass the kernel explicitly into `dip_single` or create a separate `dip_deblur_single`.

### Phase 3: Remove test-ground-truth leakage

Use ground truth only after optimization for final evaluation. Replace oracle early stopping with an observation-only criterion.

### Phase 4: Repair or remove broken methods

- Remove DnCNN breakpoints.
- Correct dataset split handling.
- Train and evaluate DnCNN on the same noise model.
- Supply or document pretrained U-Net weights.
- Convert blind DIP code into an import-safe function.
- Normalize all method names through an enum or validated lowercase identifiers.

### Phase 5: Improve metrics and results

For each method/image, save:

```text
image_id
task
method
noise_type
sigma
kernel
seed
iteration
measurement_loss
PSNR_gt
SSIM_gt
DSSIM_gt
runtime
```

Report mean and standard deviation:

$$
\bar{m}
=
\frac{1}{N}\sum_{i=1}^{N}m_i,
\qquad
s_m
=
\sqrt{
\frac{1}{N-1}
\sum_{i=1}^{N}(m_i-\bar{m})^2
}.
$$

With only five images, individual results should also be shown.

### Phase 6: Add CT compressed sensing

Only after the restoration benchmark is stable:

1. Load grayscale CT slices.
2. Normalize HU values using a documented window.
3. Implement a differentiable projection operator.
4. Generate sparse-view measurements.
5. Optimize data consistency in sinogram space.
6. Compare against FBP, TV reconstruction, and supervised baselines.

---

## Final assessment

The repository is best described today as an experimental natural-image denoising codebase centered on DIP, with incomplete deblurring and supervised extensions.

The strongest valid method is vanilla DIP denoising:

$$
\min_\theta \|f_\theta(z)-y\|_2^2,
$$

plus its ADMM-TV variants.

The most urgent correction is the deblurring objective:

$$
\boxed{
\min_\theta
\|h*f_\theta(z)-y\|_2^2
}
$$

because without the forward blur operator, the current deblur experiment cannot recover sharp content in a mathematically data-consistent way.

This report is based on all tracked Python files, both notebooks, static compilation, actual dataset inspection, and verification of missing data and checkpoint paths.
