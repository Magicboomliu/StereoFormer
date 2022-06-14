import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("../")


def EPE_Loss(disp_infer,disp_gt):
    mask = (disp_gt>0) & (disp_gt<192)
    disp_infer = disp_infer[mask]
    disp_gt = disp_gt[mask]
    return F.l1_loss(disp_infer,disp_gt,size_average=True)


class RangeLossAndDisparityLoss(nn.Module):
    def __init__(self,gamma,weight,loss):
        super(RangeLossAndDisparityLoss,self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.loss = loss
        # Disparity Loss
        self.equal_loss = Multiple_Equal_Loss(weights=self.weight,loss=self.loss)
    
    def forward(self,disp_infer,disp_gt,disp3,lower_bound,upper_bound):
        
        disparity_min = disp3 - lower_bound
        disparity_max = disp3 + upper_bound
        
        # Get Mask Here
        B,C,H,W = disp3.shape
        reference_image = torch.arange(W).view(1,1,1,W).repeat(1,1,H,1).type_as(disp_gt)
        lower_invalid_mask = (lower_bound<0).float()
        upper_invalid_mask = (upper_bound>=W-1).float()
        upper_invalid_mask2 = (upper_bound>reference_image).float()
        invalid_mask = lower_invalid_mask + upper_invalid_mask + upper_invalid_mask2
        invalid_mask = torch.clamp(invalid_mask,max=1.0)
        valid_mask = torch.ones_like(disp3).type_as(disp3) - invalid_mask
        
        disp_gt3 = F.interpolate(disp_gt,size=[disp3.size(-2),disp3.size(-1)],mode='bilinear',align_corners=False)/8.0
        
        # Range Loss: GT Disparity <= Min Disparity
        penty_lower_mask1 = ((disparity_min-disp_gt3)>0).float()
        penty_lower_mask1 = penty_lower_mask1 * valid_mask
        penty_lower_mask2 =valid_mask - penty_lower_mask1
        
        # GT Disparity >= Max Disparity
        penty_upper_mask1 = ((disp_gt3-disparity_max)>0).float()
        penty_upper_mask1 = penty_upper_mask1 * valid_mask
        penty_upper_mask2 =valid_mask - penty_upper_mask1
        lower_range_loss = (torch.abs(disp_gt3-disparity_min)* penty_lower_mask1 * self.gamma + torch.abs(disp_gt3-disparity_min)* penty_lower_mask2*(1-self.gamma)).sum()/(valid_mask.sum()+1e-8)
        upper_range_loss = (torch.abs(disp_gt3-disparity_max)* penty_upper_mask1 * self.gamma + torch.abs(disp_gt3-disparity_max)* penty_upper_mask2*(1-self.gamma)).sum()/(valid_mask.sum()+1e-8)
        range_loss = lower_range_loss + upper_range_loss
        
      
        disparity_loss = self.equal_loss(disp_infer,disp_gt)
        
        loss = range_loss *4.0  + disparity_loss

        return loss
        
        
def multiLossWithRangeLoss(gamma=0.9,weight=None,loss='Smooth_l1'):
    if weight is None:
        weight =(0.8,1.2)
    
    return RangeLossAndDisparityLoss(gamma=gamma,weight=weight,loss=loss) 




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


from models.LocalCostVolume.baseline_dynamic_supervised import LowCNN
if __name__=="__main__":
    left_image = torch.randn(1,3,320,640).cuda()
    right_image = torch.randn(1,3,320,640).cuda()
    
    disp_gt_sample =torch.randn(1,1,320,640).cuda()
    
    

    use_adaptive_refinement = True
    lowCNN = LowCNN(cost_volume_type='correlation',upsample_type='convex',
                    adaptive_refinement=use_adaptive_refinement).cuda()
    
    outputs,bounds,lower_disp = lowCNN(left_image,right_image,use_adaptive_refinement)
    
    loss_op = multiLossWithRangeLoss(gamma=0.9,weight=[0.8,1.2],loss='Smooth_l1')
    
    loss = loss_op(outputs,disp_gt_sample,lower_disp,bounds[0],bounds[1])


    
