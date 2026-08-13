import torch
from torch.utils.cpp_extension import CUDA_HOME

print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA toolkit:", CUDA_HOME)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(1))