import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
from models.GMA_Stereo.CostVolume.LocalCostVolumeSingle import build_local_cost_volume_fixed

class PyrmaidCostVolume(nn.Module):
    def __init__(self,radius,nums_levels,
                 sample_points):
        super(PyrmaidCostVolume,self).__init__()
        self.radius = radius
        self.nums_levels = nums_levels
        self.sample_points = sample_points
        
    
    def forward(self,cost_volume,radius,cur_disp):
        
        # Get the Cost Volume.
        cost_volume_pyramid = []
        cost_volume_pyramid.append(cost_volume)
        # from full searching range to 1/2 searching range.
        for i in range(self.nums_levels-1):
            B,D,H,W = cost_volume.shape
            cost_volume = cost_volume.view(B,D,-1).permute(0,2,1)
            cost_volume = F.avg_pool1d(cost_volume,2,stride=2)
            cost_volume = cost_volume.permute(0,2,1).contiguous().view(B,D//2,H,W)
            cost_volume_pyramid.append(cost_volume)
        
        # Index the Cost Volume.
        
        out_pyramid = []
        for i in range(self.nums_levels):
            corr = cost_volume_pyramid[i]
            ref_disp = cur_disp*1.0 /(2**i)
            local_cost_volume = build_local_cost_volume_fixed(corr,ref_disp,radius,self.sample_points)
            out_pyramid.append(local_cost_volume)
        
        out = torch.cat(out_pyramid,dim=1)
    
        return out
        
        
if __name__=="__main__":
    cost_volume = torch.randn(1,24,40,80).cuda()
    cur_disp = torch.abs(torch.randn(1,1,40,80)*10-8).cuda()
    
    pyramid_cost_volume = PyrmaidCostVolume(radius=2,nums_levels=3,sample_points=2*2).cuda()
    
    pyramid_cost_volume(cost_volume,2,cur_disp)