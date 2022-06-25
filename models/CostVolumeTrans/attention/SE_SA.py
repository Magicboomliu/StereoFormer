import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../..")
from attentions.SEAttention import SEAttention
from models.CostVolumeTrans.attention.self_attention import Transformer

# Spatial Attention
class SA_Module(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=16):
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

# SE Attention For 4D Cost Volume [C,D,H,W]
class CostWiseSEAttention(nn.Module):
    def __init__(self,max_disp=24,channels=256,reduction=16,
                 squeeze_type='mean'):
        super(CostWiseSEAttention,self).__init__()
        self.squeeze_type = squeeze_type
        
        
        self.avg_pooling = nn.AdaptiveAvgPool3d((max_disp,1,1))

        self.transformer = Transformer(dim=24,
                                       depth=3,
                                       heads=[2,4,8],
                                       dim_head=32,
                                       mlp_dim=24,
                                       dropout=0.)
        #softmax function
        if self.squeeze_type=='mean':
            pass
        elif self.squeeze_type=='conv3d':
            self.final_conv = nn.Conv3d(channels,1,kernel_size=3,stride=1,padding=1,
                                    bias=False)
        else:
            raise NotImplementedError
    
        

    def forward(self,x):
        B,C,D,H,W = x.shape
        
        y = self.avg_pooling(x).view(B,C,D)
        y = self.transformer(y).view(B,C,D,1,1)

        # Depth Wise
        x= x +y
        if self.squeeze_type=='mean':
            x = x.mean(1)
        elif self.squeeze_type =='conv3d':
            x = self.final_conv(x).squeeze(1)
        else:
            raise NotImplementedError
        
        return x

    


if __name__=="__main__":
    cost_volume = torch.randn(2,256,24,40,80).cuda()
    
    cost_se_attention = CostWiseSEAttention(max_disp=24,channels=256,reduction=16,
                 squeeze_type='conv3d').cuda()
    
    x = cost_se_attention(cost_volume)
    
    print(x.shape)