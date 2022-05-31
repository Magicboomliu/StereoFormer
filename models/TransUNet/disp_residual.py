
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../..')
from utils.devtools import print_tensor_shape


def conv(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        # 使用sequential可以不用实现forward函数
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True)
    )

def conv3d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm3d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False)

def conv5x5(in_channels, out_channels, stride=2,
            dilation=1, use_bn=True):
    bias = False if use_bn else True
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride,
                     padding=2, dilation=dilation, bias=bias)
    relu = nn.ReLU(inplace=True)
    if use_bn:
        out = nn.Sequential(conv,
                            nn.BatchNorm2d(out_channels),
                            relu)
    else:
        out = nn.Sequential(conv, relu)
    return out

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def deconv_bn(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, True)
    )

def conv_Relu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False),
        nn.ReLU(True)
    )


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

# Baisc Layer
class BaseLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
        # self.relu = nn.GELU()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransformerConcated(nn.Module):
    def __init__(self,swin_feature_list):
        super().__init__()
        self.swin_feature_list = swin_feature_list
        # self.relu = nn.GELU()
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ch_list = list(reversed(self.swin_feature_list))
        
        self.layer_list = nn.ModuleList()
        for id in range(len(self.ch_list) - 1):
            self.layer_list.append(
                BaseLayer(
                    dim_in = self.ch_list[id] + self.ch_list[id+1],
                    dim_out = self.ch_list[id+1],
                )
            )
 
    def forward(self, x_list):

        out = x_list[0]
        
        for id in range(len(self.ch_list) - 1):
            out = self.up_sample(out)
            out = torch.cat([out, x_list[id+1]], dim=1)
            out = self.layer_list[id](out)
    
        return out








if __name__=='__main__':
    
    feature1 = torch.randn(1,128,40,80).cuda()
    feature2 = torch.randn(1,256,20,40).cuda()
    feature3 = torch.randn(1,512,10,20).cuda()
    
    
    transConcate = TransformerConcated(swin_feature_list=[128,256,512]).cuda()
    
    seg_logit = transConcate([feature3,feature2,feature1])
    print_tensor_shape(seg_logit)