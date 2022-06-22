import numpy as np
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
from utils.devtools import print_tensor_shape
from models.BasicBlocks.resnet import ResBlock
from timm.models.layers import trunc_normal_
from models.LocalCostVolume.Attempts.updatev2 import DisparityUpdateDLCWithMask



def print_tensor_shape(inputs):
    if isinstance(inputs,list) or isinstance(inputs,tuple):
        for value in inputs:
            print(value.shape)
    else:
        print(inputs.shape)


class LowCNN(nn.Module):
    def __init__(self, max_disp=192,cost_volume_type='correlation',
                 upsample_type="simple",adaptive_refinement=False):
        super(LowCNN, self).__init__()
        self.max_disp = max_disp
        self.cost_volume_type = cost_volume_type
        self.upsample_type = upsample_type
        # whether using adaptive cost volume for refinement
        self.adaptive_refinement = adaptive_refinement
        
        if self.adaptive_refinement and self.upsample_type:
            self.local_cost_volume = DisparityUpdateDLCWithMask(input_channels=self.max_disp//8,hidden_dim=32,
                                                                feature_dim=64,
                                                        sample_points=20)

        
        self.conv1 = conv(3, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8
        
        self.downsample1 = ResBlock(256,256,stride=1) # 1/8
        self.downsample2 = ResBlock(256,512,stride=2) # 1/16
        self.downsample3 = ResBlock(512,512,stride=2) # 1/32
        
        
        self.feature_concated = TransformerConcated(swin_feature_list=[256,512,512])
        match_similarity = True
        
        self.correlation_aggreagtion = nn.Sequential(
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1)
        )
        # 1/8 Scale Cost Volume
        if self.cost_volume_type in ['correlation','concated']:
            self.low_scale_cost_volume = CostVolume(max_disp=192//8,feature_similarity=self.cost_volume_type)
    
        # 1/8 Scale Disparity Estimation
        self.disp_estimation3 = DisparityEstimation(max_disp=192//8,match_similarity=match_similarity) 
        

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv3d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    
    def forward(self,left,right,iters=12,is_training=True):
        '''CNN Blocks: Get 1/2,1/4,1/8 Level Left and Right Feature'''
        conv1_l = self.conv1(left)          # 64 1/2
        conv2_l = self.conv2(conv1_l)           # 128 1/4
        conv3_l = self.conv3(conv2_l)           # 256 1/8

        conv1_r = self.conv1(right)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)           # 1/8
        
        # Left Feature
        feature8_l = self.downsample1(conv3_l) # 1/8 L
        feature16_l = self.downsample2(feature8_l) # 1/16 L
        feature32_l = self.downsample3(feature16_l) # 1/32 L
        
        left_feature_list = [feature32_l,feature16_l,feature8_l]
        aggregated_feature_l = self.feature_concated(left_feature_list)
        
        # Right Feature
        feature8_r = self.downsample1(conv3_r) # 1/8 L
        feature16_r = self.downsample2(feature8_r) # 1/16 L
        feature32_r = self.downsample3(feature16_r) # 1/32 L
        
        right_feature_list = [feature32_r,feature16_r,feature8_r]
        aggregated_feature_r = self.feature_concated(right_feature_list)
        
        # Correlation Cost Volume Here 1/8 : Searching Range is 24
        
        low_scale_cost_volume3 = self.low_scale_cost_volume(aggregated_feature_l,aggregated_feature_r)
        low_scale_cost_volume3 = self.correlation_aggreagtion(low_scale_cost_volume3)
        final_cost = low_scale_cost_volume3
        
        low_scale_disp3 = self.disp_estimation3(final_cost)
        assert low_scale_disp3.min()>=0
        
        # Predict Disparity Here
        low_scale_disp3 = low_scale_disp3.unsqueeze(1)
        
        # Begin GRU-based Dynamic Cost Volume Regression
        
        disparity_predictions =[]
        
        for itr in range(iters):
            if itr ==0:
                disp,hidden_state,mask = self.local_cost_volume(final_cost,low_scale_disp3,left,right,None,aggregated_feature_l,True)
            else:
                disp,hidden_state,mask = self.local_cost_volume(final_cost,disp,left,right,hidden_state,aggregated_feature_l,True)
            
            if self.upsample_type=="convex":
                full_disp = upsample_convex8(disp,mask)
            else:
                full_disp = upsample_simple8(disp)
            
            disparity_predictions.append(full_disp)
        
        return disparity_predictions
                


if __name__=="__main__":
    left_image = torch.randn(1,3,320,640).cuda()
    right_image = torch.randn(1,3,320,640).cuda()
    
    use_adaptive_refinement = True
    lowCNN = LowCNN(cost_volume_type='correlation',upsample_type='convex',
                    adaptive_refinement=use_adaptive_refinement).cuda()
    
    output = lowCNN(left_image,right_image,12,True)
    
    print(len(output))
    
    for f in output:
        print(f.shape)
    
    
        
    
    
        