import sys
import torch.nn as nn
import torch
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
sys.path.append("../..")
from models.TwoD.disp_residual import *
from models.residual.resnet import ResBlock
from models.backbone.swinformer import SwinTransformer,Swin_T


def print_tensor_shape(inputs):
    if isinstance(inputs,list) or isinstance(inputs,tuple):
        for value in inputs:
            print(value.shape)
    else:
        print(inputs.shape)

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class BaseLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransformerConcated(nn.Module):
    def __init__(self,swin_feature_list):
        super().__init__()
        self.swin_feature_list = swin_feature_list
        self.relu = nn.GELU()
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



class NiNet(nn.Module):
    def __init__(self, max_disp=192, input_channel=3, res_type='normal',
                 load_swin_pretrain=False,
                 swin_transformer_path=None,
                 squeezed_volume=False):
        super(NiNet, self).__init__()
        self.max_disp = max_disp
        self.squeezed_volume = squeezed_volume
        self.res_type = res_type
        self.load_swin_pretrain = load_swin_pretrain
        self.swin_transformer_path = swin_transformer_path
        
        # Transformer Branch
        self.swin_transformer_encoder = Swin_T(load_pretrain=self.load_swin_pretrain,
                                               pretrain_path=self.swin_transformer_path)
        
        # Fixed the transformer Branch
        for p in self.swin_transformer_encoder.parameters():
            p.requires_grad = False
        
        self.feature_fusion = TransformerConcated(swin_feature_list=[96,192,384,768])
        
        # Disparity Estimation Branch
        
        self.conv1 = conv(3, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8
        
        self.conv_redir = ResBlock(256, 32, stride=1)       # skip connection
        
        # squeezed 3d Convolution
        if squeezed_volume:
            self.conv3d = nn.Sequential(
                convbn_3d(32*2, 32, 3, 1, 1),
                nn.ReLU(True),
                convbn_3d(32, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.conv_compress = ResBlock(256, 32, stride=1)    # 1/8

            self.conv3_1 = ResBlock(80, 256)                # 192 / 8 = 24 -> correlation + squeezed volume -> 24 * 2 + 32
        else:
            self.conv3_1 = ResBlock(56, 256)
        
        # Downsample 
        self.conv4 = ResBlock(256, 512, stride=2)           # 1/16
        self.conv4_1 = ResBlock(512, 512)
        self.conv5 = ResBlock(512, 512, stride=2)           # 1/32
        self.conv5_1 = ResBlock(512, 512)
        self.conv6 = ResBlock(512, 1024, stride=2)          # 1/64
        self.conv6_1 = ResBlock(1024, 1024)
        
        
        # Upsample
        self.iconv5 = nn.ConvTranspose2d(1024, 512, 3, 1, 1) # Just change the channels, Size not change
        self.iconv4 = nn.ConvTranspose2d(768, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(384, 128, 3, 1, 1)

        
        # Deconvoluton : nn.ConvTranspose2d + Relu
        self.upconv5 = deconv(1024, 512) # Channel changes, size double
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256, 128)


        # disparity estimation
        self.disp3 = nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu3 = nn.ReLU(inplace=True)


        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_left, img_right,training=False):

        # Split left image and right image
        conv1_l = self.conv1(img_left)          # 64 1/2
        conv2_l = self.conv2(conv1_l)           # 128 1/4
        conv3_l = self.conv3(conv2_l)           # 256 1/8
        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)           # 1/8
            
        # 24 channels probility cost volume
        out_corr = build_corr(conv3_l,conv3_r, self.max_disp//8)
        # 32 channels cost volume
        out_conv3a_redir = self.conv_redir(conv3_l)
        # Concated Cost Volume
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), dim=1)         # 24+32=56
        # 3D cost volume
        if self.squeezed_volume:
            conv_compress_left = self.conv_compress(conv3_l)
            conv_compress_right = self.conv_compress(conv3_r)

            cost_volume = form_cost_volume(conv_compress_left, conv_compress_right, self.max_disp//8)
            cost_volume = self.conv3d(cost_volume)
            cost_volume = torch.squeeze(cost_volume, dim=1)
            in_conv3b = torch.cat((in_conv3b, cost_volume), dim=1)

        conv3b = self.conv3_1(in_conv3b)    # 56 ---> 256
        conv4a = self.conv4(conv3b)    #Downsample to 1/16     
        conv4b = self.conv4_1(conv4a)       # 512 1/16: Simple ResBlock
        conv5a = self.conv5(conv4b)    #Downsample to 1/32
        conv5b = self.conv5_1(conv5a)       # 512 1/32：Simple ResBlock
        conv6a = self.conv6(conv5b)    #Downsample to 1/64
        conv6b = self.conv6_1(conv6a)       # 1024 1/64:Simple ResBlock

        upconv5 = self.upconv5(conv6b)      # Upsample to 1/32 : 512 1/32
        concat5 = torch.cat((upconv5, conv5b), dim=1)   # 1024 1/32
        iconv5 = self.iconv5(concat5)       # 1024-->512

        upconv4 = self.upconv4(iconv5)      # Upsample to 1/16: 256 1/16
        concat4 = torch.cat((upconv4, conv4b), dim=1)   #256+512: 768 1/16
        iconv4 = self.iconv4(concat4)       # 768-->256 1/16

        upconv3 = self.upconv3(iconv4)      # Upsample to 1/8: 128 1/8
        concat3 = torch.cat((upconv3, conv3b), dim=1)    # 128+256=384 1/8
        iconv3 = self.iconv3(concat3)       # 128
        
        
        # Get 1/8 Disparity Here
        pr3 = self.disp3(iconv3)
        pr3 = self.relu3(pr3) # Use Simple CNN to do disparity regression

        # Swin-Former feature
        context_information_list = self.swin_transformer_encoder(img_left)
        '''[1,96,80,160] ----> 1/4 H
           [1,192,40,80] ----> 1/8 H
           [1,384,20,40] ----> 1/16 H
           [1,768,10,20] ----> 1/32 H'''
        #[1/4 Size Feature Size]
        context_feature = self.feature_fusion(context_information_list[::-1])
        # Use this as the context clue to do non-spatial propagation
        
        print(context_feature.shape)


if __name__ == '__main__':
    import time
    pretrained_path = "/home/zliu/Desktop/Codes/StereoFormer/pretrained/backbone/upernet_swin_tiny_patch4_window7_512x512.pth"
    model = NiNet(res_type='attention',squeezed_volume=True,load_swin_pretrain=True,
                  swin_transformer_path=pretrained_path).cuda()
    input = torch.randn(1, 3, 320, 640).cuda()
    
    
    output = model(input,input,True)
    
    # print_tensor_shape(output)

