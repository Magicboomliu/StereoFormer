import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.hrnet import hrnet18
from residual.resnet import ResBlock
from utils.devtools import print_tensor_shape


class HRNet_Stereo(nn.Module):
    def __init__(self,max_disp=192,res_type='normal',pretrain=False):
        super(HRNet_Stereo,self).__init__()
        self.max_disp = max_disp
        self.res_type = res_type
        
        # Feature Extraction
        self.encoder = hrnet18(pretrained=pretrain,progress=False)
        
        # Feature Fusion
        self.feature_fusion = ConcatHead(hrn_out_dim=270)
        
    def forward(self,left,right,is_training=False):
        
        left_feature = self.encoder(left) #[1/4,1/8,1/16,1/32]
        right_feature = self.encoder(right) #[1/4,1/8,1/16,1/32]
        
        left_feature = self.feature_fusion(left_feature)
        right_feature = self.feature_fusion(right_feature)
        
        # Cost Volume Building
        
        # Cost Volume Aggregation
        
        # 1/4 disp disparity
        
        
        
        
        return left_feature,right_feature



class ConcatHead(nn.Module):
    '''
    把 backbone 的多尺度输出合在一起
    '''
    def __init__(self,hrn_out_dim):
        super().__init__()
        self.feature_aggregation = ResBlock(n_in=hrn_out_dim,n_out=128,kernel_size=3,stride=1)

    def forward(self, x_list):
        upsample_list = []
        for id, x in enumerate(x_list):
            upsample_list.append(F.interpolate(x, scale_factor=int(1<<id), mode="bilinear") )
        upsample_out = torch.cat(upsample_list, dim=1)
        upsample_out = self.feature_aggregation(upsample_out)
        
        return upsample_out

if __name__=="__main__":
    
    # Test Input
    left_input = torch.randn(1,3,320,640).cuda()
    right_input = torch.randn(1,3,320,640).cuda()
    
    hrnet_stereo = HRNet_Stereo().cuda()
    
    left_feature,right_feature = hrnet_stereo(left_input,right_input)
    
    print_tensor_shape(left_feature)
    print("------------------------")
    print_tensor_shape(right_feature)