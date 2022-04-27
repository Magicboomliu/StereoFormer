
import sys
sys.path.append("../..")
from utils.devtools import print_tensor_shape
import torch.nn as nn
import torch
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
from models.TwoD.disp_residual import *
from models.residual.resnet import ResBlock
from models.backbone.hrnet import hrnet18

# Feature Fusion
class Concated_Head(nn.Module):
    def __init__(self,select_res='1/8'):
        super(Concated_Head,self).__init__()
        self.select_res = select_res
        self.downsample = nn.Sequential(
            nn.Conv2d(270,256,3,1,1,bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
    def forward(self,feature_list):
        upsample_list = []
        if self.select_res=='1/8':
            upsample_list.append(F.interpolate(feature_list[0],scale_factor=1/2,mode='bilinear'))
            upsample_list.append(feature_list[1])
            upsample_list.append(F.interpolate(feature_list[2],scale_factor=2.0,mode='bilinear'))
            upsample_list.append(F.interpolate(feature_list[3],scale_factor=4.0,mode='bilinear'))
        elif self.select_res=='1/4':
            upsample_list.append(F.interpolate(feature_list[0]))
            upsample_list.append(F.interpolate(feature_list[1],scale_factor=2.0,mode='bilinear'))
            upsample_list.append(F.interpolate(feature_list[2],scale_factor=4.0,mode='bilinear'))
            upsample_list.append(F.interpolate(feature_list[3],scale_factor=8.0,mode='bilinear'))    
        else:
            raise NotImplementedError
        upsample_out = torch.cat(upsample_list,dim=1)
        upsample_out = self.downsample(upsample_out)
        
        return upsample_out

class Hrnet_EDNet(nn.Module):
    def __init__(self, batchNorm=False, max_disp=192, 
                        res_type='normal', 
                        squeezed_volume=False):
        super(Hrnet_EDNet, self).__init__()
        self.max_disp = max_disp
        self.squeezed_volume = squeezed_volume
        self.res_type = res_type
        
        # Left feature and Right Feature : 1/4, 1/8, 1/16, 1/32
        self.feature_extractor = hrnet18(False,False)
        self.feature_fusion = Concated_Head(select_res='1/8')
        
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

            self.conv3_1 = ResBlock(80, 256)    # 192 / 8 = 24 -> correlation + squeezed volume -> 24 * 2 + 32
        else:
            self.conv3_1 = ResBlock(56, 256)

        self.conv4 = ResBlock(256, 512, stride=2)           # 1/16
        self.conv4_1 = ResBlock(512, 512)
        self.conv5 = ResBlock(512, 512, stride=2)           # 1/32
        self.conv5_1 = ResBlock(512, 512)
        self.conv6 = ResBlock(512, 1024, stride=2)          # 1/64
        self.conv6_1 = ResBlock(1024, 1024)

        self.iconv5 = nn.ConvTranspose2d(1024, 512, 3, 1, 1) # Just change the channels, Size not change
        self.iconv4 = nn.ConvTranspose2d(768, 256, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(384, 128, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(82,64, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(48, 32, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(19, 32, 3, 1, 1)
        
        self.feature_deconv = deconv(18,16)
        # Deconvoluton : nn.ConvTranspose2d + Relu
        self.upconv5 = deconv(1024, 512) # Channel changes, size double
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256, 128)
        self.upconv2 = deconv(128, 64)
        self.upconv1 = deconv(64, 32)
        self.upconv0 = deconv(32, 16)

        # disparity estimation
        self.disp3 = nn.Conv2d(128,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu0 = nn.ReLU(inplace=True)
        
        
        # Disparity residual
        if self.res_type=='attention':
            residual = res_submodule_attention
        else:
            raise NotImplementedError

        self.res_submodule_2 = residual(scale=2, input_layer=64, out_planes=32)
        self.res_submodule_1 = residual(scale=1, input_layer=32, out_planes=32)
        self.res_submodule_0 = residual(scale=0, input_layer=32, out_planes=32)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_left, img_right, training=False):
        
        # Get 1/8 Disparity , 1/4 Disparity, 1/2 Disparity and Full Disparity
        left_feature_pyramid = self.feature_extractor(img_left)
        right_feature_pyramid = self.feature_extractor(img_right)
        
        left_feature_half = self.feature_deconv(left_feature_pyramid[0])
        
        #[1/4,1/8,1/16,1/32]
        #1/8 cost volume
        lf = self.feature_fusion(left_feature_pyramid)
        rf = self.feature_fusion(right_feature_pyramid)

        # build corr 
        # 24 channels probility cost volume
        out_corr = build_corr(lf,rf, self.max_disp//8)
        # 32 channels cost volume
        out_conv3a_redir = self.conv_redir(lf)
        # Concated Cost Volume
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), dim=1)         # 24+32=56
        # 3D cost volume
        if self.squeezed_volume:
            conv_compress_left = self.conv_compress(lf)
            conv_compress_right = self.conv_compress(rf)

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

    
        upconv2 = self.upconv2(iconv3)      # Upsample to 1/4 :64 1/4
        concat2 = torch.cat((upconv2, left_feature_pyramid[0]), dim=1)  # 64+128=192 1/4
        iconv2 = self.iconv2(concat2) #192-->64
        '''Here Beigin the Disparity Residual refinement'''
        # 1/4 Disparity Refinement
        # Upsample the 1/8 Disparity to coarse 1/4 Disparity
        pr2 = F.interpolate(pr3, size=(pr3.size()[2] * 2, pr3.size()[3] * 2), mode='bilinear')
        # Stacked Hourglass to do disparity residual 
        '''1/4 Feature Spatial Propagation Here'''
        res2 = self.res_submodule_2(img_left, img_right, pr2,iconv2)

        pr2 = pr2 + res2
        pr2 = self.relu2(pr2)

        # 1/2 Disparity Refinement
        upconv1 = self.upconv1(iconv2)      # Upsample to 1/2 :32 1/2
        concat1 = torch.cat((upconv1, left_feature_half), dim=1)  #32+64=96
        iconv1 = self.iconv1(concat1)       # 32 1/2
        # pr1 = self.upflow2to1(pr2)
        pr1 = F.interpolate(pr2, size=(pr2.size()[2] * 2, pr2.size()[3] * 2), mode='bilinear')
        '''1/2 Feature Spatial Propagation Here'''
        res1 = self.res_submodule_1(img_left, img_right, pr1,iconv1) #
        pr1 = pr1 + res1
        pr1 = self.relu1(pr1)

        # Full Scale Disparity refinemnts
        upconv1 = self.upconv0(iconv1)      # 16 1
        concat0 = torch.cat((upconv1, img_left), dim=1)     # 16+3=19 1
        iconv0 = self.iconv0(concat0)       # 16 1
        pr0 = F.interpolate(pr1, size=(pr1.size()[2] * 2, pr1.size()[3] * 2), mode='bilinear')
        '''Full Scale Feature Spatial Propagation Here'''
        res0 = self.res_submodule_0(img_left, img_right, pr0,iconv0)
        pr0 = pr0 + res0
        pr0 = self.relu0(pr0)
        
        
        pr1 = F.interpolate(pr1,scale_factor=2.0,mode='bilinear')
        pr2 = F.interpolate(pr2,scale_factor=4.0,mode='bilinear')
        pr3 = F.interpolate(pr3,scale_factor=8.0,mode='bilinear')
        
        if training:
            return [pr0, pr1, pr2, pr3]
        else: 
            return pr0
        

if __name__=="__main__":
    left_sample = torch.randn(1,3,320,640).cuda()
    right_sample = torch.randn(1,3,320,640).cuda()
    
    swin_t_ednet = Hrnet_EDNet(max_disp=192,res_type='attention',squeezed_volume=False).cuda()
    
    disparity_pyramid = swin_t_ednet(left_sample,right_sample,True)
    
    print_tensor_shape(disparity_pyramid)