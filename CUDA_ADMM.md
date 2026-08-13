# CUDA-accelerated ADMM operators

The ADMM-DIP-TV and ADMM-DIP-WTV implementations use custom C++/CUDA kernels
for the image-space operations inside their optimization loops.

## Accelerated operations

The extension implements the periodic finite differences

$$
(D_hx)_{i,j}=x_{i,j+1}-x_{i,j},\qquad
(D_vx)_{i,j}=x_{i+1,j}-x_{i,j},
$$

their adjoint for autograd, and a fused kernel for isotropic TV shrinkage and
the scaled-dual update:

$$
q_p=D x_p+\mu_p,
$$

$$
t_p=\max\left(1-\frac{\tau_p}{\lVert q_p\rVert_2},0\right)q_p,
$$

$$
\mu_p^{k+1}=\mu_p^k+D x_p-t_p.
$$

This replaces the previous FFT/IFFT derivative calculation and several
separate elementwise PyTorch launches.

## Requirements

Runtime compilation requires all of the following:

- An NVIDIA GPU visible to PyTorch.
- An NVIDIA driver compatible with the installed PyTorch CUDA runtime.
- A CUDA toolkit containing `nvcc`.
- A supported host C++ compiler.
- Ninja, which PyTorch uses to build extensions.

Check the environment with:

```python
import torch
from torch.utils.cpp_extension import CUDA_HOME
from src.admm_cuda import cuda_extension_status

print("CUDA available:", torch.cuda.is_available())
print("CUDA toolkit:", CUDA_HOME)
print("ADMM extension:", cuda_extension_status())
```

The extension compiles lazily during the first ADMM call that receives a CUDA
tensor. Later calls in the same environment reuse PyTorch's build cache.

To see compiler output, set this before importing the project:

```python
import os
os.environ["DIP_CUDA_EXT_VERBOSE"] = "1"
```

To disable the extension and use the equivalent PyTorch implementation:

```python
import os
os.environ["DIP_DISABLE_CUDA_EXT"] = "1"
```

## Scope and expected speedup

The extension accelerates ADMM image operators. It does not replace the DIP
network, PyTorch autograd, Adam, or LBFGS. Those components contain most of the
runtime, especially in the fixed-TV implementation where LBFGS may evaluate
its closure repeatedly. Moving the complete optimizer loop to native code
would require porting the generator and optimizer to LibTorch and would no
longer interoperate naturally with the current Python model.

For a larger runtime reduction, use fewer inner iterations during development
and prefer Adam over nested LBFGS steps. The CUDA kernels and reduced iteration
counts address different sources of overhead and can be used together.
