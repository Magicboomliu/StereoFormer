import torch
import torch.nn as nn
import torch.nn.functional as F

def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x


# Local Cost Volume
class LocalCostVolume(nn.Module):
    def __init__(self,radius=4,sample_points=10) -> None:
        super().__init__()
        self.radius = radius
        self.sample_points = sample_points
    def forward(self,old_cost_volume,cur_disparity,consider_valid=False):
        B,D,H,W = old_cost_volume.shape
        lower_bound = cur_disparity - self.radius
        upper_bound = cur_disparity + self.radius
        
        if consider_valid:
          lower_invalid_mask = (lower_bound<0).float()
          upper_invalid_mask = (upper_bound>=D-1).float()
          invalid_mask = lower_invalid_mask + upper_invalid_mask
          invalid_mask = torch.clamp(invalid_mask,max=1.0)
        else:
          lower_bound = torch.clamp(lower_bound,min=0)
          upper_bound = torch.clamp(upper_bound,min=0,max=D)

        # sample intervals
        # [B,1,H,W]
        sample_intervals = (upper_bound - lower_bound)*1.0/ self.sample_points
        addition_summation = (torch.arange(self.sample_points+1)).type_as(old_cost_volume)
        #[B,Sample_N+1,H,W]
        addition_summation = addition_summation.view(1,self.sample_points+1,1,1)
        sampling_candiate_intervals = addition_summation * sample_intervals

        # New Sampling Points
        sampling_candidates = lower_bound + sampling_candiate_intervals

        # Get the New Sampling Points
        if consider_valid:
          sampling_candidates = sampling_candidates * (1-invalid_mask) + invalid_mask * cur_disparity.repeat(1,self.sample_points+1,1,1)
        # Get Cost Volume Here
        local_cost_volume = build_cost_volume_from_volume(old_cost_volume,sampling_candidates)

        # Get Score Map Here
        score_map = torch.softmax(local_cost_volume,dim=1)
        disp = torch.sum(score_map*sampling_candidates,dim=1)

        # Get Disparity Here
        disp = disp.unsqueeze(1)
            
        return disp
    

def build_cost_volume_from_volume(old_volume,sampling_candidates):
    '''Bilinear interplolation'''
    B,D,H,W = old_volume.shape
    # CEIL AND FLOOR: Ceil and Floor 
    sample_candidate_ceil = ste_ceil(sampling_candidates)
    sample_candidate_floor = ste_floor(sampling_candidates)
    
    sample_candidate_ceil = torch.clamp(sample_candidate_ceil,min=0,max=D-1).long()
    sample_candidate_floor = torch.clamp(sample_candidate_floor,min=0,max=D-1).long()
    
    # Floor Rate      
    floor_rate =(sample_candidate_ceil- sampling_candidates)
    ceil_rate = 1.0 - floor_rate
    
    
    floor_volume = torch.gather(old_volume,dim=1,index=sample_candidate_floor)
    ceil_volume = torch.gather(old_volume,dim=1,index=sample_candidate_ceil)
    new_volume = floor_volume * floor_rate + ceil_volume * ceil_rate
    
    return new_volume


