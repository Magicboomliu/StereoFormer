import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../..")
from timm.models.layers import trunc_normal_
from models.backbone.swinformer import BasicLayer
from torch.nn.init import kaiming_normal
from Transformers.Transformer.SwinTransformer.PatchMerging import PatchMerging

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

class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 swin_former_depths=[4,4,4],
                 input_dimension=256,
                 embedd_dim =256,
                 norm_layer =nn.LayerNorm,
                 window_size =8,
                 nums_head =[4,8,4],
                 out_indices =[0,1,2],
                 drop_path_rate =0.2,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate =0.,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 downsample=True):
        super(SwinTransformerBlock,self).__init__()
        self.downsample = downsample
        self.downsample_method = PatchMerging
        self.num_layers = len(swin_former_depths)
        
        self.linear_embedding = LinearEmbedding(input_dim=input_dimension,embedd_dim=embedd_dim,norm_layer=norm_layer)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(swin_former_depths))]  # stochastic depth decay rule
        num_features = [embedd_dim*(2**i) for i in range(self.num_layers)]
        self.num_features = num_features
        
        # Transformer Layers
        self.window_size = window_size
        self.numbers_of_head=nums_head
        self.transformer_stages = nn.ModuleList()
        self.out_indices=out_indices

        # Swin Transformers Stages
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=num_features[i_layer],
                depth=swin_former_depths[i_layer],
                num_heads=self.numbers_of_head[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(swin_former_depths[:i_layer]):sum(swin_former_depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=self.downsample_method if ((i_layer < self.num_layers - 1) and self.downsample) else None)
            self.transformer_stages.append(layer)

        # add a norm layer for each output
        for i_layer in self.out_indices:
            layer = nn.LayerNorm(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)


    
    def forward(self,feature):

        x = self.linear_embedding(feature)

        
        Wh,Ww = feature.size(2),feature.size(3)

        cost_f = x
        
        outs = []
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

                outs.append(cost_out)
        
        return outs
    

if __name__=="__main__":
    
    input_feature = torch.randn(2,256,40,80).cuda()
    
    
    swin_former = SwinTransformerBlock(swin_former_depths=[4,4,4],
                                       input_dimension=256,
                                       embedd_dim=256,
                                       norm_layer=nn.LayerNorm,
                                       window_size=8,
                                       nums_head=[4,8,4],
                                       out_indices=(0,1,2),
                                       downsample=True,
                                       mlp_ratio=4,
                                       qk_scale=None,
                                       qkv_bias=True,
                                       attn_drop_rate=0).cuda()
    outs = swin_former(input_feature)
    
    for out in outs:
        print(out.shape)
