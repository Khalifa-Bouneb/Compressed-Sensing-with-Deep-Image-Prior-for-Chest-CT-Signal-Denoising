from .skip import skip
from .unet import UNet

import torch.nn as nn


def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3,
            act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128,
            skip_n11=4, num_scales=5, downsample_mode='stride'):
    """Instantiate a generator network by name.

    Supported NET_TYPE values: 'skip', 'UNet', 'identity'.
    """
    if NET_TYPE == 'skip':
        net = skip(
            input_depth, n_channels,
            num_channels_down=[skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d,
            num_channels_up=[skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u,
            num_channels_skip=[skip_n11] * num_scales if isinstance(skip_n11, int) else skip_n11,
            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun,
        )

    elif NET_TYPE == 'UNet':
        net = UNet(
            num_input_channels=input_depth, num_output_channels=n_channels,
            feature_scale=4, more_layers=0, concat_x=False,
            upsample_mode=upsample_mode, pad=pad,
            norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True,
        )

    elif NET_TYPE == 'identity':
        assert input_depth == n_channels
        net = nn.Sequential()

    else:
        raise ValueError(f"Unknown NET_TYPE: {NET_TYPE!r}. Choose from 'skip', 'UNet', 'identity'.")

    return net
