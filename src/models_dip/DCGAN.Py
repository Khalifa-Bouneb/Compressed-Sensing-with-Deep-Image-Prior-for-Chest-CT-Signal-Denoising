import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResizeToInput(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, out, ref):
        if out.shape[-2:] == ref.shape[-2:]:
            return out
        if self.mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            return F.interpolate(out, size=ref.shape[-2:], mode=self.mode, align_corners=False)
        return F.interpolate(out, size=ref.shape[-2:], mode=self.mode)


class _DCGANWrapper(nn.Module):
    def __init__(self, model, output_size, resize_mode='bilinear'):
        super().__init__()
        self.model = model
        self.output_size = output_size          # (H, W) of target image
        self.resizer = _ResizeToInput(mode=resize_mode)

    def forward(self, x):
        out = self.model(x)
        if out.shape[-2:] != self.output_size:
            out = F.interpolate(out, size=self.output_size, mode=self.resizer.mode,
                                align_corners=False if self.resizer.mode == 'bilinear' else None)
        return out

def dcgan(inp=2,
          ndf=32,
          num_ups=8,
          n_channels=3,
          need_sigmoid=True,
          need_bias=True,
          upsample_mode='nearest',
          need_convT=False,
          output_size=None):

    layers = [
        nn.ConvTranspose2d(inp, ndf, kernel_size=3, stride=1, padding=0, bias=need_bias),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2, inplace=True)
    ]

    for _ in range(num_ups - 1):
        if need_convT:
            layers += [
                nn.ConvTranspose2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=need_bias),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            layers += [
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True)
            ]

    if need_convT:
        layers += [
            nn.ConvTranspose2d(ndf, n_channels, 4, 2, 1, bias=need_bias)
        ]
    else:
        layers += [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(ndf, n_channels, kernel_size=3, stride=1, padding=1, bias=need_bias)
        ]

    if need_sigmoid:
        layers += [nn.Sigmoid()]

    model = nn.Sequential(*layers)
    return _DCGANWrapper(model, output_size=output_size, resize_mode=upsample_mode)

