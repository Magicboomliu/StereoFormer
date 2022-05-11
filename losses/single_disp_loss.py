import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")


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