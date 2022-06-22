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

class Multiple_Equal_Loss(nn.Module):
    def __init__(self,weights=None,
                 loss='Smooth_l1'):
        super().__init__()
        self.weights = weights
        self.loss = loss
        self.smoothl1 = nn.SmoothL1Loss(size_average=True)
        if self.loss=='Smooth_l1':
            self.loss = self.smoothl1
    
    
    def forward(self,disp_infer,disp_gt):
        
        loss = 0
        
        if (type(disp_infer) is tuple) or (type(disp_infer) is list):
            for i, input_ in enumerate(disp_infer):
                assert disp_gt.size(-1)==input_.size(-1)
                target = disp_gt
                mask = (target <192) & (target>=0)
                mask.detach_()
                input_ = input_[mask]
                target_ = target[mask]
                
                loss+= self.smoothl1(input_,target_) * self.weights[i]
        
        else:
            mask = (target <192) & (target>=0)
            mask = mask.detach_()
            
            loss = self.loss(disp_infer[mask],disp_gt[mask])
        
        return loss
    

def multiequalloss(weight=None,loss='Smooth_l1'):
    if weight is None:
        weight =(0.8,1.2)
    
    return Multiple_Equal_Loss(weights=weight,loss=loss)


if __name__=="__main__":
    cur_disp1 = torch.abs(torch.randn(1,1,320,640)).cuda()
    cur_disp2 = torch.abs(torch.randn(1,1,320,640)).cuda()
    
    target_disp = torch.abs(torch.randn(1,1,320,640)).cuda()
    
    infer_list = [cur_disp1,cur_disp2]
    
    loss = multiequalloss(weight=[0.8,1.2]).cuda()
    
    ls = loss(infer_list,target_disp)
    
    print(ls)
    
    
    