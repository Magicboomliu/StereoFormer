import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
import sys
sys.path.append("../..")
from models.utils.upsample import ConvAffinityUpsample,upsample_convex8,upsample_simple8
from models.utils.build_cost_volume import CostVolume
from models.utils.estimation import DisparityEstimation
from models.utils.disp_residual import conv
from models.utils.feature_fusion import TransformerConcated
from models.BasicBlocks.resnet import ResBlock

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


# Write A UNet-based Baseline For testing
class Baseline(nn.Module):
    def __init__(self,max_disp=192,
                 cost_volume_type='correlation',
                 upsample_type='simple'):
        super().__init__()
        self.max_disp = max_disp
        self.cost_volume_type = cost_volume_type
        self.upsample_type = upsample_type
        
        # Upsample Disparity From Affinities
        if self.upsample_type:
            self.upsample_mask = ConvAffinityUpsample(input_channels=256,hidden_channels=128)
        
        
        self.conv1 = conv(3,64,7,2)               #1/2 
        self.conv2 = ResBlock(64,128,stride=2)    #1/4 
        self.conv3 = ResBlock(128,256,stride=2)   #1/8
        

        # Downsample
        self.conv4 = ResBlock(256, 512, stride=2)           # 1/16
        self.conv4_1 = ResBlock(512, 512)
        self.conv5 = ResBlock(512, 512, stride=2)           # 1/32
        self.conv5_1 = ResBlock(512, 512)
        self.conv6 = ResBlock(512, 1024, stride=2)          # 1/64
        self.conv6_1 = ResBlock(1024, 1024)
        
        #Upsample
        self.up4 = Up(in_ch=1024,out_ch=512,norm_layer=nn.BatchNorm2d,leaky=True,upsample='deconv')
        self.redir4 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=1,stride=1,padding=0,bias=False)
        )
        self.up3 = Up(in_ch=512,out_ch=512,norm_layer=nn.BatchNorm2d,leaky=True,upsample='deconv')
        self.redir3 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=1,stride=1,padding=0,bias=False)
        )
        self.up2 = Up(in_ch=512,out_ch=256,norm_layer=nn.BatchNorm2d,leaky=True,upsample='deconv')
        self.redir2 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0,bias=False)
        )
        
        
        match_similarity = True
        # 1/8 Scale Cost Volume
        if self.cost_volume_type in ['correlation','concated']:
            self.low_scale_cost_volume = CostVolume(max_disp=192//8,feature_similarity=self.cost_volume_type)

        self.initial_conv = nn.Sequential(
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1)
        )
        
        self.cost_aggregation = nn.Sequential(
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1),
            nn.Conv2d(in_channels=24,out_channels=24,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True)
        )
        
        if self.upsample_type:
            self.upsample_mask = ConvAffinityUpsample(input_channels=256,hidden_channels=128)



        # 1/8 Scale Disparity Estimation
        self.disp_estimation3 = DisparityEstimation(max_disp=192//8,match_similarity=match_similarity) 
        
        
        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    
    def forward(self,img_left,img_right,is_training=True):
        # Split left image and right image
        # Left Feature
        conv1_l = self.conv1(img_left)          # 64 1/2
        conv2_l = self.conv2(conv1_l)           # 128 1/4
        conv3_l = self.conv3(conv2_l)           # 256 1/8

        conv4_l = self.conv4(conv3_l)
        conv4_l_1 = self.conv4_1(conv4_l)   # 512 1/16
        conv5_l = self.conv5(conv4_l_1)     # 512 1/32
        conv5_l_1 = self.conv5_1(conv5_l) 
        conv6_l = self.conv6(conv5_l_1)       # 1024 1/64
        conv6_l_1 = self.conv6_1(conv6_l)     # 1024 1/64    
        conv7_l = self.up4(conv6_l_1)
        conv7_l_1 = F.relu(conv7_l+ self.redir4(conv5_l_1)) #512 1/32
        conv8_l = self.up3(conv7_l_1)
        conv8_l_1 = F.relu(conv8_l+ self.redir3(conv4_l_1)) #512 1/16
        conv9_l = self.up2(conv8_l_1)
        conv9_l_1 = F.relu(conv9_l+ self.redir2(conv3_l)) # 128 1/8
        
        # Right Feature
        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)           # 1/8
        conv4_r = self.conv4(conv3_r)
        conv4_r_1 = self.conv4_1(conv4_r)   # 512 1/16
        conv5_r = self.conv5(conv4_r_1)     # 512 1/32
        conv5_r_1 = self.conv5_1(conv5_r) 
        conv6_r = self.conv6(conv5_r_1)       # 1024 1/64
        conv6_r_1 = self.conv6_1(conv6_r)     # 1024 1/64
        
        conv7_r = self.up4(conv6_r_1)
        conv7_r_1 = F.relu(conv7_r+ self.redir4(conv5_r_1)) #512 1/32

        conv8_r = self.up3(conv7_r_1)
        conv8_r_1 = F.relu(conv8_r+ self.redir3(conv4_r_1)) #512 1/16


        conv9_r = self.up2(conv8_r_1)
        conv9_r_1 = F.relu(conv9_r+ self.redir2(conv3_r)) # 128 1/8
        
        # Build Cost Volume Here
        low_scale_cost_volume3 = self.low_scale_cost_volume(conv9_l_1,conv9_r_1)
        # Cost Volume Aggregation
        cost_volume = self.initial_conv(low_scale_cost_volume3)
        cost_volume = self.cost_aggregation(cost_volume)
        
        # Low Scale Disparity
        low_scale_disp3 = self.disp_estimation3(cost_volume)
        
        assert low_scale_disp3.min()>=0

        low_scale_disp3 = low_scale_disp3.unsqueeze(1)
        
        if self.upsample_type=='convex':
            pr3_mask = self.upsample_mask(conv9_l_1)
            pr0 = upsample_convex8(low_scale_disp3,pr3_mask)
        elif self.upsample_type=='simple':
            pr0 = upsample_simple8(low_scale_disp3)
        
        
        return pr0
        

if __name__=="__main__":
    
    # Left and right Input
    left = torch.randn(1,3,320,640).cuda()
    right = torch.randn(1,3,320,640).cuda()
    
    baseline = Baseline(max_disp=192,cost_volume_type='correlation',
                        upsample_type='simple').cuda()
    
    out = baseline(left,right)
    
    print(out.shape)