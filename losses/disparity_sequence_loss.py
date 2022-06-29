import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_loss(pred_disp,gt_disp,loss_gamma=0.9,max_disp=192):
    
    N_predictions = len(pred_disp)

    assert N_predictions >0
    
    disp_loss = 0.0
    
    valid_mask = (gt_disp>0)*(gt_disp<max_disp)
    valid_mask = valid_mask.type_as(gt_disp)
    
    pred_disp = [p * valid_mask for p in pred_disp]
    
    for i in range(N_predictions):
        assert not torch.isnan(pred_disp[i]).any() and not torch.isinf(pred_disp[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(N_predictions - 1))
        i_weight = adjusted_loss_gamma**(N_predictions - i - 1)
        i_loss = (pred_disp[i] - gt_disp).abs()
        disp_loss += i_weight * i_loss[valid_mask.bool()].mean()


    return disp_loss