"""
UNet architecture for Deep Image Prior.
Based on the original DIP paper: https://dmitryulyanov.github.io/deep_image_prior
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Downsampling block with convolutional layers."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with convolutional layers."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet model for Deep Image Prior.
    
    Args:
        input_channels: Number of input channels (default: 32 for random noise)
        output_channels: Number of output channels (default: 1 for grayscale CT)
        num_channels_down: List of channel sizes for downsampling path
        num_channels_up: List of channel sizes for upsampling path
        num_channels_skip: List of channel sizes for skip connections
    """
    
    def __init__(
        self,
        input_channels=32,
        output_channels=1,
        num_channels_down=[128, 128, 128, 128, 128],
        num_channels_up=[128, 128, 128, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4]
    ):
        super(UNet, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_channels_down[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels_down[0]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(len(num_channels_down) - 1):
            self.down_blocks.append(
                DownBlock(num_channels_down[i], num_channels_down[i + 1])
            )
        
        # Skip connection convolutions
        self.skip_convs = nn.ModuleList()
        for i in range(len(num_channels_skip)):
            self.skip_convs.append(
                nn.Conv2d(num_channels_down[i], num_channels_skip[i], kernel_size=1)
            )
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in range(len(num_channels_up) - 1):
            in_ch = num_channels_up[i] + num_channels_skip[i]
            self.up_blocks.append(
                UpBlock(in_ch, num_channels_up[i + 1])
            )
        
        # Final output convolution
        final_in_channels = num_channels_up[-1] + num_channels_skip[-1]
        self.output_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, output_channels, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        """
        Forward pass of the UNet.
        
        Args:
            x: Input tensor of shape (batch, input_channels, H, W)
            
        Returns:
            Output tensor of shape (batch, output_channels, H, W)
        """
        # Initial convolution
        x = self.input_conv(x)
        
        # Downsampling with skip connections
        skip_connections = []
        for i, down_block in enumerate(self.down_blocks):
            skip = self.skip_convs[i](x)
            skip_connections.append(skip)
            x = down_block(x)
        
        # Final skip connection
        skip = self.skip_convs[-1](x)
        skip_connections.append(skip)
        
        # Upsampling with skip connections
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 2)]
            x = up_block(x, skip)
        
        # Final output
        skip = skip_connections[0]
        x = torch.cat([x, skip], dim=1)
        x = self.output_conv(x)
        
        return x


def get_input_noise(input_depth, spatial_size, noise_type='uniform', var=0.1):
    """
    Generate random noise input for Deep Image Prior.
    
    Args:
        input_depth: Number of channels for the noise
        spatial_size: Tuple of (height, width)
        noise_type: Type of noise ('uniform' or 'normal')
        var: Variance for normal noise
        
    Returns:
        Noise tensor of shape (1, input_depth, height, width)
    """
    if noise_type == 'uniform':
        noise = torch.rand(1, input_depth, spatial_size[0], spatial_size[1])
    elif noise_type == 'normal':
        noise = torch.randn(1, input_depth, spatial_size[0], spatial_size[1]) * var
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noise
