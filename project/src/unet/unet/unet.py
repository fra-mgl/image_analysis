"""
Filename: unet.py
Description: implementation of U-Net model
Author: Image-inativi
Date: May 21st 2025
Note: source ---
"""

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,dilation=2, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x



class UNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=64, dimensions=14, save_intermediates=False):
        super(UNet, self).__init__()

        self.save_intermediates = save_intermediates
        self.intermediate_outputs = {}

        # Layers
        self.conv1 = DoubleConv(input_channels, base_channels)
        self.down1 = DownLayer(base_channels, base_channels * 2)
        self.down2 = DownLayer(base_channels * 2, base_channels * 4)
        self.down3 = DownLayer(base_channels * 4, base_channels * 8)
        self.up1 = UpLayer(base_channels * 8, base_channels * 4)
        self.up2 = UpLayer(base_channels * 4, base_channels * 2)
        self.up3 = UpLayer(base_channels * 2, base_channels)

        self.last_conv = nn.Conv2d(base_channels, dimensions, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        if self.save_intermediates: self.intermediate_outputs['conv1'] = x1.detach()

        x2 = self.down1(x1)
        if self.save_intermediates: self.intermediate_outputs['down1'] = x2.detach()

        x3 = self.down2(x2)
        if self.save_intermediates: self.intermediate_outputs['down2'] = x3.detach()

        x4 = self.down3(x3)
        if self.save_intermediates: self.intermediate_outputs['down3'] = x4.detach()

        x = self.up1(x3, x4)
        if self.save_intermediates: self.intermediate_outputs['up1'] = x.detach()

        x = self.up2(x2, x)
        if self.save_intermediates: self.intermediate_outputs['up2'] = x.detach()

        x = self.up3(x1, x)
        if self.save_intermediates: self.intermediate_outputs['up3'] = x.detach()

        x = self.last_conv(x)
        if self.save_intermediates: self.intermediate_outputs['output'] = x.detach()

        return x

    def get_intermediate_outputs(self):
        return self.intermediate_outputs if self.save_intermediates else None

