import sys
from xml.sax.handler import feature_external_ges
sys.path.append("../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.swinformer import SwinTransformer
from utils.devtools import print_tensor_shape
from models.cost import CostVolume,CostVolumePyramid
from models.residual.resnet import ResBlock


class Cost_Volume_Aggregation(nn.Module):
    def __init__(self,cost_volume_len=4,max_disp=192//4):
        super(Cost_Volume_Aggregation,self).__init__()
        
        # Intra-Scale Aggregation
        self.branches = nn.ModuleList()
        # Cross-Scale Aggregation
        self.fusion_branches = nn.ModuleList()
        
        self.cost_volume_len = cost_volume_len
        self.max_disp = max_disp
        
        # Intra Cost volume aggregation
        for idx in range(self.cost_volume_len):
            candidate = self.max_disp // pow(2,idx)
            self.branches.append(ResBlock(n_in=candidate,n_out=candidate,kernel_size=3,stride=1))

        # Cross Scale Cost Volume Aggregation
        '''All fuse to 1/4 Scale and 1/8 Scale'''
        for i in range(len(len(self.branches))):
            self.fusion_branches.append(nn.ModuleList())
            for j in range(self.cost_volume_len):
                if i==j:
                    self.fusion_branches[-1].append(nn.Identity())
                # incoming cost volume is smaller than current cost volume
                elif i<j:
                    self.fusion_branches[-1].append(
                    nn.Sequential(nn.Conv2d(self.max_disp // (2 ** j), self.max_disp // (2 ** i),kernel_size=1, bias=False),
                                      nn.BatchNorm2d(self.max_disp // (2 ** i))))
                # incoming cost volume is bigger than current cost volume
                elif i>j:
                    
                    pass
                        
    def forward(self,cost_volume_list):
        after_cost_volume_list =[]
        for idx, cost_volume in enumerate(cost_volume_list):
            intra_branch = self.branches[idx]
            cur_cost_volume = intra_branch(cost_volume)
            after_cost_volume_list.append(cur_cost_volume)
        
        return after_cost_volume_list
        



class Swin_Stereo(nn.Module):
    def __init__(self,max_disp=192,
                 feature_fusion=False):
        super(Swin_Stereo,self).__init__()
        self.max_disp = max_disp
        self.feature_fusion = feature_fusion
        self.encoder =SwinTransformer(pretrain_img_size=224,patch_size=4,in_chans=3,embed_dim=96,
                                 depths=[2,2,6,2],
                                 num_heads=[3,6,12,24],
                                 window_size=7,
                                 mlp_ratio=4,
                                 qkv_bias=True,
                                 qk_scale=None)
        # Cost Volume Here
        self.build_cost_volume = CostVolumePyramid(max_disp=self.max_disp//4,feature_similarity='correlation')
        
        # Cost Volume Aggregation
        self.cost_volume_aggregation = Cost_Volume_Aggregation(4,max_disp=192//4)
        
        
    def forward(self,left,right,is_training=False):
        
        left_feature = self.encoder(left) #[1/4,1/8,1/16,1/32]
        right_feature = self.encoder(right) #[1/4,1/8,1/16,1/32]
        
        cost_volume = self.build_cost_volume(left_feature,right_feature)
        
        cost_volume = self.cost_volume_aggregation(cost_volume)
        print_tensor_shape(cost_volume)
        
        return left_feature,right_feature
       


if __name__=="__main__":
    # Test Input
    left_input = torch.randn(1,3,320,640).cuda()
    right_input = torch.randn(1,3,320,640).cuda()
    
    swin_stereo = Swin_Stereo(max_disp=192,feature_fusion=False).cuda()
    
    L_feature, R_feature = swin_stereo(left_input,right_input,True)
    # print_tensor_shape(L_feature)
    # print("---------------------")
    # print_tensor_shape(R_feature)
    