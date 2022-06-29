import torch
import torch.nn as nn
import torch.nn.functional as F

def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x


def build_local_cost_volume_fixed(cost_volume,cur_disp,searching_radius,sample_nums):
    
    # Cost Volume Shape
    B,D,H,W = cost_volume.shape
    
    # Get sample candidates
    lower_bound = cur_disp - searching_radius
    upper_bound = cur_disp + searching_radius
    sample_intervals = (upper_bound-lower_bound) *1.0/(sample_nums)    
    addition_summation = torch.arange(sample_nums+1).type_as(cur_disp)
    addition_summation=addition_summation.view(1,sample_nums+1,1,1)
    sampling_candiate_intervals = addition_summation * sample_intervals
    sampling_candidates =lower_bound + sampling_candiate_intervals
    
    # valid mask
    sample_candidate_ceil = ste_ceil(sampling_candidates)
    sample_candidate_floor = ste_floor(sampling_candidates)
    
    sample_candidate_ceil = torch.clamp(sample_candidate_ceil,min=0,max=D-1)
    sample_candidate_floor = torch.clamp(sample_candidate_floor,min=0,max=D-1)
    
    # Linear interplotation
    floor_rate =(sample_candidate_ceil- sampling_candidates)
    ceil_rate = 1.0 - floor_rate
    
    ceil_volume = torch.gather(cost_volume,dim=1,index=sample_candidate_ceil.long())
    floor_volume = torch.gather(cost_volume,dim=1,index=sample_candidate_floor.long())
    
    final_volume = ceil_volume*ceil_rate+ floor_volume*floor_rate
    
    return final_volume

if __name__=="__main__":
    cost_volume = torch.randn(1,24,40,80)
    cur_disp = torch.abs(torch.randn(1,1,40,80) *10-8)
    cur_disp = torch.clamp(cur_disp,max=24-1)
    
    final_cost_volume = build_local_cost_volume_fixed(cost_volume,cur_disp,2,4)
    
    print(final_cost_volume.shape)