from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet
from .simple_cnn import SimpleCNN
from .DCGAN import dcgan

import torch.nn as nn

def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    if NET_TYPE == 'ResNet':
        net = ResNet(
            num_input_channels=input_depth,
            num_output_channels=n_channels,
            num_blocks=10,
            num_channels=16,
            need_residual=True,
            act_fun=act_fun,
            need_sigmoid=True,
            norm_layer=nn.BatchNorm2d,
            pad=pad,
        )
        
    elif NET_TYPE == 'CNN-ATTN':
        net = SimpleCNN(input_depth, output_channels=3, img_size=(320, 480), pad=pad, upsample_mode=upsample_mode)
    elif NET_TYPE == 'skip':
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios = [32, 16, 8, 4, 2, 1], fill_noise=False,pad=pad)

    elif NET_TYPE =='UNet':
        net = UNet(num_input_channels=input_depth, num_output_channels=3, 
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)
    elif NET_TYPE == 'identity':
        assert input_depth == 3
        net = nn.Sequential()
    elif NET_TYPE == 'dcgan':
        net = dcgan(
        inp=input_depth,
        n_channels=n_channels,
        need_sigmoid=True,
        need_bias=True,
        upsample_mode=upsample_mode
    )
    else:
        assert False

    return net

