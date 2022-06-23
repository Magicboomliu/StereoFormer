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
from timm.models.layers import trunc_normal_
from models.CrossAttentionCostVolume.gwc_cost_volume import build_gwc_volume
from TransformerLZH.Transformer.CrossVit.crossvit_ape import CrossVit
from TransformerLZH.Transformer.SwinTransformer.MySwinBlocks import MySwinFormerBlocks

def print_tensor_shape(inputs):
    if isinstance(inputs,list) or isinstance(inputs,tuple):
        for value in inputs:
            print(value.shape)
    else:
        print(inputs.shape)

# Cascaded CostVolumeList
class CascadeGroupWiseCostVolumeAggregation(nn.Module):
    def __init__(self,cost_volume_length,
                 input_cost_volume_channels,
                 image_size,
                 include_old_cost_volume=False):
        super(CascadeGroupWiseCostVolumeAggregation,self).__init__()
        # Iterations Settings
        self.cost_volume_length = cost_volume_length
        self.input_cost_volume_channels = input_cost_volume_channels
        self.firstRound_cross_attention_nums = self.cost_volume_length//2
        self.secondRound_cross_attention_nums = self.firstRound_cross_attention_nums//2
        
        self.include_old_cost_volume = include_old_cost_volume
        
        if self.include_old_cost_volume:
            self.fixed_cost_volume = CostVolume(max_disp=192//8,feature_similarity='correlation')
        
        self.FirstRound_CrossAttention = nn.ModuleList()
        
        for _ in range(self.firstRound_cross_attention_nums):
            cross_vit = CrossVit(image_size=[image_size,image_size],
                                 embedd_dim=[input_cost_volume_channels,input_cost_volume_channels],
                                 input_dimension=[input_cost_volume_channels,input_cost_volume_channels],
                                 patch_size=((1,1),(1,1)),
                                 basic_depth=1,
                                 cross_attention_depth=1,
                                 cross_attention_dim_head=32,
                                 cross_attention_head=[4],
                                 enc_depths=[1,1],
                                 enc_heads=[[4],[4]],
                                 enc_head_dim=[32,32],
                                 enc_mlp_dims=[32,32],
                                 skiped_patch_embedding=False,
                                 dropout_rate=0.1,
                                 emb_dropout=0.1)
            
            self.FirstRound_CrossAttention.append(cross_vit)
        self.SecondRound_CrossAttention = nn.ModuleList()
        
        for _ in range(self.secondRound_cross_attention_nums):
            cross_vit = CrossVit(image_size=[image_size,image_size],
                        embedd_dim=[self.input_cost_volume_channels,self.input_cost_volume_channels],
                        input_dimension=(self.input_cost_volume_channels,self.input_cost_volume_channels),
                        patch_size=((1,1),(1,1)),
                        basic_depth=1,
                        cross_attention_dim_head=32,
                        cross_attention_depth=1,
                        cross_attention_head=[4],
                        enc_depths=[1,1],
                        enc_heads=[[4],[4]],
                        enc_head_dim=[32,32],
                        enc_mlp_dims=[32,32],
                        dropout_rate=0.1,
                        emb_dropout=0.1,
                        skiped_patch_embedding=False)
            self.SecondRound_CrossAttention.append(cross_vit)
        
    def forward(self,cost_volume:tuple,left_feature=None,right_feature=None):
        
        # Cost Volume Aggregation
        cost_volume0 = cost_volume[0]
        cost_volume1 = cost_volume[1]
        cost_volume2 = cost_volume[2]
        cost_volume3 = cost_volume[3]
        
        # First Round Aggregation : 0
        first_cost_volume0 = self.FirstRound_CrossAttention[0](cost_volume0,cost_volume2)
        first_cost_volume1 = self.FirstRound_CrossAttention[1](cost_volume1,cost_volume3)
        
        # Second Round Aggregation: 1
        second_cost_volume = self.SecondRound_CrossAttention[0](first_cost_volume0,first_cost_volume1)
        
        if self.include_old_cost_volume:
            old_cost_volume = self.fixed_cost_volume(left_feature,right_feature)
            final_cost_volume = second_cost_volume + old_cost_volume
        else:
            final_cost_volume = second_cost_volume
            
        return final_cost_volume
        

class Baseline_ca(nn.Module):
    def __init__(self, max_disp=192,cost_volume_type='group_wise_correlation',
                 upsample_type="simple"):
        super(Baseline_ca, self).__init__()
        self.max_disp = max_disp
        self.cost_volume_type = cost_volume_type
        self.upsample_type = upsample_type
        
        if self.upsample_type:
            self.upsample_mask = ConvAffinityUpsample(input_channels=256,hidden_channels=128)
        
        self.conv1 = conv(3, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8
        
        self.downsample1 = ResBlock(256,256,stride=1) # 1/8
        self.downsample2 = ResBlock(256,512,stride=2) # 1/16
        self.downsample3 = ResBlock(512,512,stride=2) # 1/32
        
        
        self.feature_concated = TransformerConcated(swin_feature_list=[256,512,512])
        match_similarity = True
    
        self.cost_volume_aggregation = MySwinFormerBlocks(input_feature_size=[320//8,640//8],
                                          input_feature_channels=192//8,
                                          skiped_patch_embed=False,
                                          block_depths=[2,2,2],
                                          out_indices=(0,1,2),
                                          nums_head=[4,8,4],
                                          patch_size=(1,1),
                                          downsample=False,
                                          embedd_dim=192//8,
                                          use_ape=False,
                                          frozen_stage=-1,
                                          use_prenorm=True,
                                          norm_layer=nn.LayerNorm)
        

        self.conv_aggreagtion = nn.Sequential(
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1)
        )
        
        # 1/8 Scale Cost Volume
        if self.cost_volume_type in ['correlation','concated']:
            self.low_scale_cost_volume = CostVolume(max_disp=192//8,feature_similarity=self.cost_volume_type)
        elif self.cost_volume_type in ['group_ca']:
            
            self.nums_groups = 4
            self.low_scale_cost_volume = CascadeGroupWiseCostVolumeAggregation(
                                                                            cost_volume_length=self.nums_groups,
                                                                            include_old_cost_volume=True,
                                                                            image_size=[320//8,640//8],
                                                                            input_cost_volume_channels=192//8)
            
    
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
        
    


        
    def forward(self,left,right,is_training=True):
        
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
        
        # # Correlation Cost Volume Here 1/8 : Searching Range is 24
        # low_scale_cost_volume3 = self.low_scale_cost_volume(aggregated_feature_l,aggregated_feature_r)
        
        # Group-Wise Cross Attention Cost Volume Aggergation.
        if self.cost_volume_type=='group_ca':
            groupwise_cost_volume = build_gwc_volume(aggregated_feature_l,aggregated_feature_r,maxdisp=192//8,num_groups=self.nums_groups)
            cost_volume_list = torch.chunk(groupwise_cost_volume,self.nums_groups,dim=1)
            cost_volume_list = [c.squeeze(1) for c in cost_volume_list]
            cost_volume = self.low_scale_cost_volume(cost_volume_list)
            aggregated_cost_volume = self.cost_volume_aggregation(cost_volume)
            final_cost_volume = aggregated_cost_volume[-1]
            
            final_cost_volume_conv = self.conv_aggreagtion(final_cost_volume)
        
        low_scale_disp3 = self.disp_estimation3(final_cost_volume_conv)
        
        assert low_scale_disp3.min()>=0
        
        low_scale_disp3 = low_scale_disp3.unsqueeze(1)
        
        if self.upsample_type=='convex':
            pr3_mask = self.upsample_mask(aggregated_feature_l)
            pr0 = upsample_convex8(low_scale_disp3,pr3_mask)
        elif self.upsample_type=='simple':
            pr0 = upsample_simple8(low_scale_disp3)
        
        
        return pr0


# Spatial Attention
class SA_Module(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=32):
        super(SA_Module, self).__init__()
        self.attention_value = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_value = self.attention_value(x)
        return attention_value




if __name__=="__main__":
    left_image = torch.randn(1,3,320,640).cuda()
    right_image = torch.randn(1,3,320,640).cuda()
    
    lowCNN = Baseline_ca(cost_volume_type='group_ca',upsample_type='convex').cuda()
    output = lowCNN(left_image,right_image,True)
    
    print(output.shape)
    