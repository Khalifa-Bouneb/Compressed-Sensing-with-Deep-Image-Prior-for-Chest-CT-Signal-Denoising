import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming common.py functions and classes are accessible from this context
from .common import *

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_channels, img_size, pad, upsample_mode):
        super(SimpleCNN, self).__init__()
        # Define network layers using the 'conv' function with specified padding
        self.conv1 = conv(input_channels, 64, kernel_size=3, stride=1, bias=True, pad=pad)
        self.conv2 = conv(64, 128, kernel_size=3, stride=1, bias=True, pad=pad)
        self.conv3 = conv(128, 256, kernel_size=3, stride=1, bias=True, pad=pad)
        self.conv4 = conv(256, output_channels, kernel_size=3, stride=1, bias=True, pad=pad)

        # Define an upsampling layer if specified
        if upsample_mode == 'nearest':
            self.upsample = nn.Upsample(size=img_size, mode='nearest')
        elif upsample_mode == 'bilinear':
            self.upsample = nn.Upsample(size=img_size, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.upsample(x)  # Upsample to the target size
        return x

# Example usage
# input_channels = 32  # Number of input channels
# output_channels = 3  # Desired number of output channels (for RGB images)
# img_size = (320, 480)  # Spatial dimensions of the output image (H, W)
# pad = 'reflection'  # Specify the padding mode, e.g., 'zero', 'reflection'
# upsample_mode = 'bilinear'  # Specify the upsampling mode, e.g., 'nearest', 'bilinear'
# model = SimpleCNN(input_channels, output_channels, img_size, pad, upsample_mode)

# Assuming you have an input tensor 'z' with the correct dimensions
# z = torch.randn(1, 32, 320, 480)  # Example input tensor
# output = model(z)
