"""CUDA-accelerated ADMM image operators with transparent PyTorch fallback.

The extension is loaded lazily only when the first CUDA tensor is received.
Set ``DIP_DISABLE_CUDA_EXT=1`` to force the PyTorch implementation.
"""

import os
import warnings
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

if __package__:
    from .admm_profile import profile_region
else:
    from admm_profile import profile_region


_EXTENSION = None
_EXTENSION_ERROR = None


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None or _EXTENSION_ERROR is not None:
        return _EXTENSION
    if os.environ.get("DIP_DISABLE_CUDA_EXT") == "1":
        _EXTENSION_ERROR = RuntimeError("disabled by DIP_DISABLE_CUDA_EXT")
        return None
    if not torch.cuda.is_available() or CUDA_HOME is None:
        _EXTENSION_ERROR = RuntimeError(
            "CUDA extension requires an active NVIDIA GPU and a CUDA toolkit with nvcc"
        )
        return None

    source_dir = Path(__file__).resolve().parent / "cuda_ops"
    try:
        with profile_region("CUDA_EXTENSION_LOAD_OR_COMPILE"):
            build_dir = source_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            _EXTENSION = load(
                name="dip_admm_cuda",
                sources=[
                    str(source_dir / "admm_ops.cpp"),
                    str(source_dir / "admm_ops_cuda.cu"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=os.environ.get("DIP_CUDA_EXT_VERBOSE") == "1",
                build_directory=str(build_dir),
            )
    except Exception as error:  # Fall back without breaking CPU/notebook runs.
        _EXTENSION_ERROR = error
        warnings.warn(
            f"Could not build the ADMM CUDA extension; using PyTorch fallback: {error}",
            RuntimeWarning,
        )
    return _EXTENSION


class _PeriodicGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image):
        (Dh_out, Dv_out) = tuple(_load_extension().gradient(image))
        # ctx.save_for_backward(image)
        return Dh_out, Dv_out

    @staticmethod
    def backward(ctx, grad_h, grad_v):
        return _load_extension().divergence(grad_h.contiguous(), grad_v.contiguous())


def cuda_extension_available():
    """Return whether the compiled extension is active in this process."""
    return _EXTENSION is not None


def cuda_extension_status():
    """Return a human-readable extension status without forcing compilation."""
    if _EXTENSION is not None:
        return "loaded"
    if _EXTENSION_ERROR is not None:
        return f"fallback ({_EXTENSION_ERROR})"
    if not torch.cuda.is_available():
        return "fallback (CUDA device unavailable)"
    if CUDA_HOME is None:
        return "fallback (CUDA toolkit/nvcc unavailable)"
    return "available; will compile on first CUDA use"


def periodic_gradient(image):
    """Compute the current ADMM forward differences with periodic boundaries."""
    if image.is_cuda and _load_extension() is not None:
        return _PeriodicGradient.apply(image)
    return (
        torch.roll(image, shifts=-1, dims=-1) - image,
        torch.roll(image, shifts=-1, dims=-2) - image,
    )


@torch.no_grad()
def shrink_and_update_dual(grad_h, grad_v, dual_h, dual_v, threshold,
                           epsilon=1e-12):
    """Apply isotropic shrinkage and update both scaled ADMM dual variables."""
    threshold = torch.as_tensor(threshold, device=grad_h.device, dtype=grad_h.dtype)
    extension = _load_extension() if grad_h.is_cuda else None
    if extension is not None:
        return tuple(extension.shrink_dual(
            grad_h, grad_v, dual_h, dual_v, threshold, epsilon
        ))

    q_h = grad_h + dual_h
    q_v = grad_v + dual_v
    q_norm = torch.sqrt(q_h.square() + q_v.square())
    scale = torch.clamp(q_norm - threshold, min=0) / torch.clamp(
        q_norm, min=epsilon
    )
    split_h = scale * q_h
    split_v = scale * q_v
    next_dual_h = dual_h + grad_h - split_h
    next_dual_v = dual_v + grad_v - split_v
    return split_h, split_v, next_dual_h, next_dual_v
