import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
import torch.nn.functional as F
import sys
sys.path.append("../..")
from models.TwoD.disp_residual import ConvAffinityUpsample
from models.TransUNet.disp_residual import *
from utils.devtools import print_tensor_shape
from models.residual.resnet import ResBlock
from models.TransUNet.build_cost_volume import CostVolume
from models.TransUNet.estimation import DisparityEstimation
from models.TransUNet.GWcCostVolume import build_concat_volume,build_gwc_volume,conv3d,StereoNetAggregation
from models.TwoD.disp_residual import upsample_convex8,upsample_simple8
from models.TwoD.vit import ViT,TransformerEncoder
from timm.models.layers import trunc_normal_
from models.backbone.swinformer import BasicLayer
from models.CostVolumeTrans.attention.SE_attention import CostWiseSEAttention

def print_tensor_shape(inputs):
    if isinstance(inputs,list) or isinstance(inputs,tuple):
        for value in inputs:
            print(value.shape)
    else:
        print(inputs.shape)

class LinearEmbedding(nn.Module):
    def __init__(self,input_dim,embedd_dim,norm_layer=None,use_proj=True):
        super(LinearEmbedding,self).__init__()
        self.input_dim = input_dim
        self.norm_layer = norm_layer
        self.use_proj = use_proj
        self.embedd_dim = embedd_dim   
        if self.norm_layer is not None:
            self.norm = norm_layer(embedd_dim)
        if self.use_proj:
            self.proj = nn.Conv2d(input_dim,embedd_dim,kernel_size=1,stride=1,padding=0,bias=False)
        
    def forward(self,x):
        if self.proj:
            x = self.proj(x)
    
        if self.norm_layer is not None:
            # Linear Norm
            Wh,Ww = x.size(2),x.size(3)
            x = x.flatten(2).transpose(1,2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embedd_dim, Wh, Ww)
            # Flatten to [B,N,C]: N= H x W
            x = x.flatten(2).transpose(1, 2)
            
        return x


class BaselineSE(nn.Module):
    def __init__(self, max_disp=192,cost_volume_type='group_wise_correlation',
                 upsample_type="simple"):
        super(BaselineSE, self).__init__()
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
        

        # Swin Former Blocks
        swin_former_depths=[4,4,4]
        drop_path_rate =0.2 
        
        mlp_ratio=1.
        qkv_bias=True
        qk_scale=None
        attn_drop_rate=0.
        drop_rate =0.
        self.num_layers = len(swin_former_depths)
        
        self.linear_embedding = LinearEmbedding(input_dim=192//8,embedd_dim=192//8,norm_layer=nn.LayerNorm)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(swin_former_depths))]  # stochastic depth decay rule
        num_features = [int(192//8) for _ in range(self.num_layers)]
        self.num_features = num_features
        
        # Transformer Layers
        self.window_size = 8
        numbers_of_head=[4,8,4]
        self.transformer_stages = nn.ModuleList()
        self.out_indices=(0, 1, 2)
        
        # Swin Transformers Stages
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(192//8),
                depth=swin_former_depths[i_layer],
                num_heads=numbers_of_head[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(swin_former_depths[:i_layer]):sum(swin_former_depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=None)
            self.transformer_stages.append(layer)

        # add a norm layer for each output
        for i_layer in self.out_indices:
            layer = nn.LayerNorm(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            
        
        self.feature_concated = TransformerConcated(swin_feature_list=[256,512,512])
        match_similarity = True
        
        
        self.correlation_aggregation = nn.Sequential(
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1),
            ResBlock(24,24,3,1)
        )
        
        # 1/8 Scale Cost Volume
        if self.cost_volume_type in ['correlation','concated']:
            self.low_scale_cost_volume = CostVolume(max_disp=192//8,feature_similarity=self.cost_volume_type)
        elif self.cost_volume_type in ['correlation_wo_mean']:
            self.low_scale_cost_volume = CostVolume(max_disp=192//8,feature_similarity=self.cost_volume_type)
            self.cost_channels_aggregation = CostWiseSEAttention(max_disp=192//8,channels=256,
                                                                 reduction=16,squeeze_type='conv3d')
    
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
        
    
    # Freeze Some Stages
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.linear_embedding.eval()
            for param in self.linear_embedding.parameters():
                param.requires_grad = False
        elif self.frozen_stages >= 1:
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        
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
        
        # Correlation Cost Volume Here 1/8 : Searching Range is 24
        low_scale_cost_volume3 = self.low_scale_cost_volume(aggregated_feature_l,aggregated_feature_r)
        
        if self.cost_volume_type=='correlation_wo_mean':
            low_scale_cost_volume3 = self.cost_channels_aggregation(low_scale_cost_volume3)

            
        # Transformer Based value
        linear_cost = self.linear_embedding(low_scale_cost_volume3)      
        cost_f = linear_cost
        Wh, Ww = low_scale_cost_volume3.size(2),low_scale_cost_volume3.size(3)
        # Transformer Aggregation
        for i in range(self.num_layers):
            swin_layer = self.transformer_stages[i]
            cost_out,H,W,cost_f,new_Wh,new_Ww = swin_layer(cost_f,Wh,Ww)
            Wh = new_Wh
            Ww = new_Ww

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                cost_out = norm_layer(cost_out)
                cost_out = cost_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
        
        final_cost = cost_out

        final_cost = self.correlation_aggregation(final_cost)
        
        low_scale_disp3 = self.disp_estimation3(final_cost)
        
        assert low_scale_disp3.min()>=0
        
        low_scale_disp3 = low_scale_disp3.unsqueeze(1)
        
        if self.upsample_type=='convex':
            pr3_mask = self.upsample_mask(aggregated_feature_l)
            pr0 = upsample_convex8(low_scale_disp3,pr3_mask)
        elif self.upsample_type=='simple':
            pr0 = upsample_simple8(low_scale_disp3)
        
        
        return pr0





if __name__=="__main__":
    left_image = torch.randn(1,3,320,640).cuda()
    right_image = torch.randn(1,3,320,640).cuda()
    
    lowCNN = BaselineSE(cost_volume_type='correlation_wo_mean',upsample_type='convex').cuda()
    output = lowCNN(left_image,right_image,True)
    
    print(output.shape)