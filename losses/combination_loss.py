import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")

# End-Point-Error 
def EPE_Loss(disp_infer,disp_gt):
    mask = (disp_gt>0) & (disp_gt<192)
    disp_infer = disp_infer[mask]
    disp_gt = disp_gt[mask]
    return F.l1_loss(disp_infer,disp_gt,size_average=True)

# Single Scale Loss
class SingleScaleLoss(nn.Module):
    def __init__(self, 
                 loss='Smooth_l1'):
        super(SingleScaleLoss, self).__init__()        
        self.loss = loss
        self.smoothl1 = nn.SmoothL1Loss(size_average=True)
        if self.loss=='Smooth_l1':
            self.loss = self.smoothl1
    
    def forward(self, disp_infer, disp_gt):
        
        if disp_infer.size(-1)!=disp_gt.size(-1):
            scale = disp_gt.size(-1)//disp_infer.size(-1)
            disp_infer = F.interpolate(disp_infer,size=(disp_gt.size(-2),disp_gt.size(-1),
                                        ),mode='bilinear',align_corners=False) * scale
            target = disp_gt
            mask = (target<192) & (target>0)
            mask.detach_()
            
            disp_infer_ = disp_infer[mask]
            target_ = target[mask]
            
            loss = self.smoothl1(disp_infer_,target_)

        else:
            mask = (disp_gt<192) & (disp_gt>0)
            mask = mask.detach_()
            
            loss = self.loss(disp_infer[mask],disp_gt[mask])
        
        return loss

def singlescaleloss(loss='Smooth_l1'):
    return SingleScaleLoss(loss)


# Searching Range Loss
class Searching_Range_Loss(nn.Module):
    def __init__(self,alpha=0.9):
        super(Searching_Range_Loss,self).__init__()
        '''If alpha is bigger and bigger, the searching range will more contains GT
        If alpha is smaller and smaller, the searching range will smaller and smaller. '''
        self.alpha = alpha
        
    def forward(self,pred_disp,gt_disp,lower_map,upper_map):
        
        # Obtain the penity loss Here
        lower_threshold = pred_disp - lower_map
        upper_threshold = pred_disp + upper_map
        
        # Lower Loss Addition Term : GT< Lower
        diff_lower = lower_threshold - gt_disp
        lower_invalid_mask = diff_lower > 0
        lower_invalid_mask = lower_invalid_mask.float()
        
        
        # Upper Loss Addition Term: GT > Upper
        diff_upper = gt_disp - upper_threshold
        upper_invalid_mask = diff_upper > 0
        upper_invalid_mask = upper_invalid_mask.float()
        
        
        # soft lower loss
        loss_lower = torch.abs((lower_threshold - gt_disp) * lower_invalid_mask).sum()*1.0/(lower_invalid_mask.sum()+1e-8)
        # soft upper loss
        loss_upper = torch.abs((upper_threshold-gt_disp)* upper_invalid_mask).sum() *1.0/(upper_invalid_mask.sum()+1e-8)
        
        # EPE loss
        EPE_hard_loss = F.l1_loss(upper_threshold,lower_threshold)
        
        total_loss = self.alpha * (loss_lower + loss_upper) + (1.0 -self.alpha) * EPE_hard_loss
        
        return total_loss
    

# TOTAL LOSS COMBINATIONS
class TotalLoss(nn.Module):
    def __init__(self,disp_mode='Smooth_l1',alpha=0.9,disp_emphasis=3.0):
        super(TotalLoss,self).__init__()
        self.alpha = alpha
        self.disp_emphasis = disp_emphasis
        # Disp Loss
        self.disp_loss = SingleScaleLoss(loss='Smooth_l1')
        
        self.searching_range_loss = Searching_Range_Loss(alpha=self.alpha)
        
        
        
    def forward(self,pred_disp,gt_disp,lower_offset_map,upper_offset_map):
        
        disp_loss = self.disp_loss(pred_disp,gt_disp)
        
        searching_range_loss = self.searching_range_loss(pred_disp,gt_disp,lower_offset_map,upper_offset_map)
        
        total_loss = disp_loss * self.disp_emphasis + 1.0 * searching_range_loss
        
        return total_loss
        







