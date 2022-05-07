from turtle import right
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from utils.devtools import print_tensor_shape
from models.residual.resnet import ResBlock
from models.TransUNet.disp_residual import *
from models.backbone.swinformer import BasicLayer,PatchMerging
from timm.models.layers import trunc_normal_
from torch.nn.init import kaiming_normal




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

class TransUNetStereo(nn.Module):
    def __init__(self,swin_former_depths=[4,6,2],
                        numbers_of_head=[4,8,8],
                        drop_path_rate=0.2,
                        embedd_dim = 128,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        attn_drop_rate=0.,
                        norm_layer=nn.LayerNorm,
                        drop_rate=0.,
                        out_indices=(0, 1, 2),
                        frozen_stages=-1,
                        use_checkpoint=False) -> None:
        super(TransUNetStereo,self).__init__()
        
        self.swin_former_depths = swin_former_depths
        self.numbers_of_head = numbers_of_head
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # How many Transformer Stages
        self.num_layers = len(self.swin_former_depths)
        
        # Basic Feature Extraction
        self.conv1 = conv(3, 64, 7, 2)                      # 1/2
        self.conv2 = ResBlock(64, 128, stride=2)            # 1/4
        self.conv3 = ResBlock(128, 256, stride=2)           # 1/8
        
        # Swin Former Blocks
        self.linear_embedding = LinearEmbedding(input_dim=256,embedd_dim=embedd_dim,norm_layer=nn.LayerNorm)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(swin_former_depths))]  # stochastic depth decay rule
        
        num_features = [int(embedd_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        
        # Transformer Layers
        self.window_size = 8
        self.transformer_stages = nn.ModuleList()
        
        # Swin Transformers Stages
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embedd_dim * 2 ** i_layer),
                depth=swin_former_depths[i_layer],
                num_heads=numbers_of_head[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(swin_former_depths[:i_layer]):sum(swin_former_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.transformer_stages.append(layer)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            
        self._freeze_stages()
        
        self.transformer_concated = TransformerConcated([embedd_dim,embedd_dim*2,embedd_dim*4])
        
        self.cnn_transformer_fusion_3 = ResBlock(n_in=128+256,n_out=128,kernel_size=3,stride=1)
        
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
    
         
    def forward(self,left_img,right_img,is_training=True):
        
        '''CNN Blocks: Get 1/2,1/4,1/8 Level Left and Right Feature'''
        conv1_l = self.conv1(left_img)          # 64 1/2
        conv2_l = self.conv2(conv1_l)           # 128 1/4
        conv3_l = self.conv3(conv2_l)           # 256 1/8

        conv1_r = self.conv1(right_image)
        conv2_r = self.conv2(conv1_r)
        conv3_r = self.conv3(conv2_r)           # 1/8
        
        # Linear Embedding
        conv3_l_flatten = self.linear_embedding(conv3_l) 
        conv3_r_flatten = self.linear_embedding(conv3_r) 
        
        # Transformer Feature Saved
        transformer_feature_out_left = []
        transformer_feature_out_right = []
        # Swin-Former based Feature Aggregation
        left_f = conv3_l_flatten
        right_f = conv3_r_flatten
        Wh, Ww = conv3_l.size(2),conv3_l.size(3)
        
        # Transformer Aggregation
        for i in range(self.num_layers):
            swin_layer = self.transformer_stages[i]
            left_out,H,W,left_f,new_Wh,new_Ww = swin_layer(left_f,Wh,Ww)
            right_out,H,W,right_f,new_Wh,new_Ww = swin_layer(right_f,Wh,Ww)
            Wh = new_Wh
            Ww = new_Ww

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                left_out = norm_layer(left_out)
                left_out = left_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                right_out = norm_layer(right_out)
                right_out = right_out.view(-1,H,W,self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                
                transformer_feature_out_left.append(left_out)
                transformer_feature_out_right.append(right_out)
        
        # Transformer Feature Fusion
        transformer_feature_out_left = list(reversed(transformer_feature_out_left))
        transformer_feature_out_right = list(reversed(transformer_feature_out_right))
        aggregated_left_18 = self.transformer_concated(transformer_feature_out_left)
        aggregated_right_18 = self.transformer_concated(transformer_feature_out_right)
        
        # Recover the resolution
        feature_1_8_l = torch.cat((conv3_l,aggregated_left_18),dim=1)
        feature_1_8_r = torch.cat((conv3_r,aggregated_right_18),dim=1)
        
        cnn_transformer_fusion3_l = self.cnn_transformer_fusion_3(feature_1_8_l)
        cnn_transformer_fusion3_r = self.cnn_transformer_fusion_3(feature_1_8_r)
        
        #TODO 
        '''Self-Attention Or Cross Attention Disparity Estimation'''
        
        
        

        


        


if __name__=="__main__":
    left_image = torch.randn(1,3,320,640).cuda()
    right_image = torch.randn(1,3,320,640).cuda()
    
    transUnet = TransUNetStereo().cuda()
    transUnet(left_image,right_image,True)
    