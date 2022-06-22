import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class DisparityEstimation(nn.Module):
    def __init__(self, max_disp, match_similarity=True):
        super(DisparityEstimation, self).__init__()

        self.max_disp = max_disp
        self.match_similarity = match_similarity

    def forward(self, cost_volume):
        assert cost_volume.dim() == 4

        # Matching similarity or matching cost
        cost_volume = cost_volume if self.match_similarity else -cost_volume

        prob_volume = F.softmax(cost_volume, dim=1)  # [B, D, H, W]
        # prob_volume_vis = prob_volume.squeeze(0).permute(1,2,0).cpu().numpy()
        # np.save("/home/zliu/Desktop/Codes/StereoFormer/prob_volume.npy",prob_volume_vis)
        
        if cost_volume.size(1) == self.max_disp:
            disp_candidates = torch.arange(0, self.max_disp).type_as(prob_volume)
        else:
            max_disp = prob_volume.size(1)  # current max disparity
            disp_candidates = torch.arange(0, max_disp).type_as(prob_volume)

        disp_candidates = disp_candidates.view(1, cost_volume.size(1), 1, 1)
        disp = torch.sum(prob_volume * disp_candidates, 1, keepdim=False)  # [B, H, W]

        return disp
    
    

class DisparityEstimationWithProb(nn.Module):
    def __init__(self, max_disp, match_similarity=True):
        super(DisparityEstimationWithProb, self).__init__()

        self.max_disp = max_disp
        self.match_similarity = match_similarity

    def forward(self, cost_volume):
        assert cost_volume.dim() == 4

        # Matching similarity or matching cost
        cost_volume = cost_volume if self.match_similarity else -cost_volume

        prob_volume = F.softmax(cost_volume, dim=1)  # [B, D, H, W]
        

        if cost_volume.size(1) == self.max_disp:
            disp_candidates = torch.arange(0, self.max_disp).type_as(prob_volume)
        else:
            max_disp = prob_volume.size(1)  # current max disparity
            disp_candidates = torch.arange(0, max_disp).type_as(prob_volume)

        disp_candidates = disp_candidates.view(1, cost_volume.size(1), 1, 1)
        disp = torch.sum(prob_volume * disp_candidates, 1, keepdim=False)  # [B, H, W]

        return disp,prob_volume