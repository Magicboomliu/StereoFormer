from __future__ import print_function, division
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import init


class RB(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, change_channel_nb=False):
        super(RB, self).__init__()
        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch),
            relu(*param),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(out_ch))
        if change_channel_nb:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True),
                norm_layer(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()
        self.relu_out = relu(*param)
    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = x + self.shortcut(identity)
        x = self.relu_out(x)
        return x
class Up(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, norm_layer, leaky=True, upsample='deconv'):
        super(Up, self).__init__()
        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]
        if upsample == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(out_ch),
                relu(*param)
            )
        elif upsample == 'deconv':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                norm_layer(out_ch),
                relu(*param)
            )
        else:
            assert 'Upsample is not in [bilinear, deconv]'
    def forward(self, x):
        x = self.up(x)
        return x
class UNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, norm_layer=nn.BatchNorm2d, downsample='conv', upsample='deconv',
                 leaky=True, nb_channel=32):
        super(UNet, self).__init__()
        filters = [nb_channel, nb_channel * 2, nb_channel * 4, nb_channel * 8]
        self.filters = filters
        self.input_nc = input_nc
        if downsample == 'maxpool':
            self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        elif downsample == 'conv':
            self.downsample1 = nn.Conv2d(self.filters[0], self.filters[0], kernel_size=3, stride=2, padding=1)
            self.downsample2 = nn.Conv2d(self.filters[1], self.filters[1], kernel_size=3, stride=2, padding=1)
            self.downsample3 = nn.Conv2d(self.filters[2], self.filters[2], kernel_size=3, stride=2, padding=1)
        else:
            assert 'downsample not in [maxpool|conv]'
        if leaky is True:
            relu = nn.LeakyReLU
            param = (0.2, True)
        else:
            relu = nn.ReLU
            param = [True]
        # self.Conv_input = nn.Sequential(nn.Conv2d(input_nc, self.filters[0], kernel_size=3, stride=1, padding=1),
        #                                 nn.BatchNorm2d(self.filters[0]),
        #                                 relu(*param))
        self.Conv_input = nn.Sequential(nn.Conv2d(input_nc, self.filters[0], kernel_size=7, stride=1, padding=3),
                                        norm_layer(self.filters[0]),
                                        relu(*param))
        self.Conv1 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, change_channel_nb=False)
        self.Conv2 = RB(self.filters[0], self.filters[1], norm_layer, leaky=leaky, change_channel_nb=True)
        self.Conv3 = RB(self.filters[1], self.filters[2], norm_layer, leaky=leaky, change_channel_nb=True)
        self.Conv4 = RB(self.filters[2], self.filters[3], norm_layer, leaky=leaky, change_channel_nb=True)
        self.Up4 = Up(self.filters[3], self.filters[2], norm_layer, leaky=leaky, upsample=upsample)
        self.Up_conv4 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, change_channel_nb=False)
        self.Up_conv4_2 = RB(self.filters[2], self.filters[2], norm_layer, leaky=leaky, change_channel_nb=False)
        self.Up3 = Up(self.filters[2], self.filters[1], norm_layer, leaky=leaky, upsample=upsample)
        self.Up_conv3 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, change_channel_nb=False)
        self.Up_conv3_2 = RB(self.filters[1], self.filters[1], norm_layer, leaky=leaky, change_channel_nb=False)
        self.Up2 = Up(self.filters[1], self.filters[0], norm_layer, leaky=leaky, upsample=upsample)
        self.Up_conv2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, change_channel_nb=False)
        self.Up_conv2_2 = RB(self.filters[0], self.filters[0], norm_layer, leaky=leaky, change_channel_nb=False)
        self.Conv = nn.Conv2d(self.filters[0], output_nc, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        x = input
        x_in = x
        x_in = self.Conv_input(x_in)
        e1 = self.Conv1(x_in)
        e2 = self.downsample1(e1)
        e2 = self.Conv2(e2)
        e3 = self.downsample2(e2)
        e3 = self.Conv3(e3)
        e4 = self.downsample3(e3)
        e4 = self.Conv4(e4)
        d4 = self.Up4(e4)
        d4 = self.Up_conv4(d4) + self.Up_conv4_2(e3)
        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3) + self.Up_conv3_2(e2)
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2) + self.Up_conv2_2(e1)
        out_res = self.Conv(d2)
        x = out_res
        return x










