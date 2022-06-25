import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../..")
from attentions.SEAttention import SEAttention


# SE Attention For 4D Cost Volume [C,D,H,W]
class CostWiseSEAttention(nn.Module):
    def __init__(self,max_disp=24,channels=256,reduction=16,
                 squeeze_type='mean'):
        super(CostWiseSEAttention,self).__init__()
        self.squeeze_type = squeeze_type
        self.avg_pooling = nn.AdaptiveAvgPool3d((max_disp,1,1))
        self.fc = nn.Sequential(
            nn.Linear(channels,channels//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction,channels,bias=False),
            nn.Sigmoid()
        )
        if self.squeeze_type=='mean':
            pass
        elif self.squeeze_type=='conv3d':
            self.final_conv = nn.Conv3d(channels,1,kernel_size=3,stride=1,padding=1,
                                    bias=False)
        else:
            raise NotImplementedError
    
    def forward(self,x):
        B,C,D,H,W = x.shape
        y = self.avg_pooling(x).view(B,C,D).transpose(1,2)
        y = self.fc(y).transpose(1,2).view(B,C,D,1,1)
        
        x= x*y.expand_as(x)
        
        if self.squeeze_type=='mean':
            x = x.mean(1)
        elif self.squeeze_type =='conv3d':
            x = self.final_conv(x).squeeze(1)
        else:
            raise NotImplementedError
        
        return x
        

    


if __name__=="__main__":
    cost_volume = torch.randn(1,256,24,40,80).cuda()
    
    cost_se_attention = CostWiseSEAttention(max_disp=24,channels=256,reduction=16,
                 squeeze_type='conv3d').cuda()
    
    cost_volume = cost_se_attention(cost_volume)
    
    print(cost_volume.shape)