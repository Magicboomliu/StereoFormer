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


class Variance_Based_Local_Cost_Volume(nn.Module):
    def __init__(self,garma=1,sample_points=10):
        super(Variance_Based_Local_Cost_Volume,self).__init__()
        self.sample_points = sample_points
        self.garma = garma
    
    def forward(self,old_cost_volume,cur_disparity,consider_valid=False):
        
        B,D,H,W = old_cost_volume.shape
        #get the disparity variance along the baseline
        prob_volume = F.softmax(old_cost_volume, dim=1)  # [B, D, H, W]
        variance_map_cur_disp = GetVarince(prob_volume,cur_disparity)
        # identify the searching range
        lower_bound = cur_disparity - self.garma * variance_map_cur_disp
        upper_bound = cur_disparity + self.garma * variance_map_cur_disp
        if consider_valid:
            reference_image = torch.arange(W).view(1,1,1,W).repeat(1,1,H,1)
            lower_invalid_mask = (lower_bound<0).float()
            upper_invalid_mask = (upper_bound>=D-1).float()
            upper_invalid_mask2 = (upper_bound>reference_image).float()
            invalid_mask = lower_invalid_mask + upper_invalid_mask + upper_invalid_mask2
            invalid_mask = torch.clamp(invalid_mask,max=1.0)
        else:
            lower_bound = torch.clamp(lower_bound,min =0, max=old_cost_volume.size(1)-1)
            upper_bound = torch.clamp(upper_bound,min=0,max=old_cost_volume.size(1)-1)
            
        
        
        # sample intervals
        # New Sampling Points
        # Get Cost Volume Here
        # Get Score Map Here
        # Get Disparity Here
        
        
        
    


