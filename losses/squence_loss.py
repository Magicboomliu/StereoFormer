import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_loss(pred_disp_list,gt_disp,gamma=0.8,max_dispariy=192):
    n_predictions = len(pred_disp_list)    
    total_loss = 0.0
    
    mask1 = (gt_disp<192).float()
    mask2 = (gt_disp>0).float()
    mask = mask1 * mask2
    for i in range(n_predictions):
        i_weight =  gamma**(n_predictions - i - 1)
        i_loss = (pred_disp_list[i]*mask - gt_disp *mask).abs()
        total_loss += (i_weight * i_loss).mean()
    
    return total_loss


# End-Point-Error 
def EPE_Loss(disp_infer,disp_gt):
    mask = (disp_gt>0) & (disp_gt<192)
    disp_infer = disp_infer[mask]
    disp_gt = disp_gt[mask]
    return F.l1_loss(disp_infer,disp_gt,size_average=True)

        
    
