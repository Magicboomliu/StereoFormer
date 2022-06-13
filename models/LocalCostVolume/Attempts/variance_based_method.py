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
            reference_image = torch.arange(W).view(1,1,1,W).repeat(1,1,H,1).type_as(cur_disparity)
            reference_image.requires_grad=False
            lower_invalid_mask = (lower_bound<0).float()
            upper_invalid_mask = (upper_bound>=D-1).float()
            upper_invalid_mask2 = (upper_bound>reference_image).float()
            invalid_mask = lower_invalid_mask + upper_invalid_mask + upper_invalid_mask2
            invalid_mask = torch.clamp(invalid_mask,max=1.0)
        else:
            lower_bound = torch.clamp(lower_bound,min =0, max=old_cost_volume.size(1)-1)
            upper_bound = torch.clamp(upper_bound,min=0,max=old_cost_volume.size(1)-1)
            
        
        # sample intervals
        sample_intervals = (upper_bound - lower_bound)*1.0/ self.sample_points
        addition_summation = (torch.arange(self.sample_points+1)).type_as(old_cost_volume)
        
        #[B,Sample_N+1,H,W]
        addition_summation = addition_summation.view(1,self.sample_points+1,1,1)
        sampling_candiate_intervals = addition_summation * sample_intervals
        # New Sampling Points
        sampling_candidates = lower_bound + sampling_candiate_intervals
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



if __name__=="__main__":
    
    old_cost_volume = torch.abs(torch.randn(1,3,320,640)).cuda()
    cur_disp = torch.abs(torch.randn(1,1,320,640)).cuda()
    
    variance_cost_volume = Variance_Based_Local_Cost_Volume(garma=1,sample_points=20).cuda()
    
    
    disp = variance_cost_volume(old_cost_volume,cur_disp,True)
    

    print(disp.shape)

    


