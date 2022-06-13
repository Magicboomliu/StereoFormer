import torch
import torch.nn as nn
import torch.nn.functional as F

def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x

def GetVarince(prob_volume,cur_disp):
    
    ''' (1) cur_disp: current disparity is the match expectation
        (2) cost volume: Cost Volume is the fixed-length Cost Volume'''
    # Find the variance of each position
    searching_range = prob_volume.shape[1]
    disparity_candidates = torch.arange(0, searching_range).type_as(prob_volume)
    disparity_candidates = disparity_candidates.view(1, prob_volume.size(1), 1, 1)
    error_via_average = (disparity_candidates - cur_disp) * (disparity_candidates - cur_disp)
    square_variance = torch.sum(prob_volume * error_via_average, 1, keepdim=False)  # [B, H, W]
    root_variance = torch.sqrt(square_variance).unsqueeze(1)
    return root_variance


# Variance Cost Volume with an learnable Offset
class VarianceCostVolumeWithOffset(nn.Module):
    def __init__(self):
        super(VarianceCostVolumeWithOffset,self).__init__()
        # how to learn an offset based on Variance?
        # probality Volume's reliables
        
        
        
    def forward(self,x):
        pass
    




if __name__=="__main__":
    
    # OLD Cost Volume
    old_cost_volume = torch.abs(torch.randn(1,3,320,640)).cuda()
    cur_disp = torch.abs(torch.randn(1,1,320,640)).cuda()
    variance_cost_volume = Variance_Based_Local_Cost_Volume(garma=1,sample_points=20).cuda()    
    disp = variance_cost_volume(old_cost_volume,cur_disp,True)

    print(disp.shape)
    