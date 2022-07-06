import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GMA_Stereo.Models.GMA_Stereo.disparity_update import BasicUpdateBlock


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class GRURefinemnet(nn.Module):
    def __init__(self,hidden_dim,
                 cost_volume_dimension,
                 radius,
                 iters,
                 upsample_rate= 2.0,
                 output_list=False):
        super(GRURefinemnet,self).__init__()
        
        self.iters = iters
        self.radius = radius
        self.upsample_rate = upsample_rate
        self.output_list = output_list
        self.update_block = BasicUpdateBlock(hidden_dim=hidden_dim,
                                             cost_volume_dimension=cost_volume_dimension)

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = int(2**self.upsample_rate)
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)
    
    def forward(self,pyramid_cost_volume,cost_volume,
                cur_disp,inp,net):
        
        ref_coords = torch.zeros_like(cur_disp).type_as(cur_disp)
        
        if self.output_list:
            disparity_predictions =[]
        
        for itr in range(self.iters):
            cur_disp = cur_disp.detach()
            corr = pyramid_cost_volume(cost_volume,self.radius,cur_disp)
            disp= cur_disp - ref_coords
            with autocast(enabled=True):
                net, up_mask, delta_disp = self.update_block(net, inp, corr, disp)
            
            # Update the current disparity
            cur_disp = F.relu(cur_disp + delta_disp,True)
            
            # Upsample
            disp_up = self.upsample_flow(cur_disp,up_mask)
            
            if self.output_list:
                disparity_predictions.append(disp_up)
        
        if self.output_list:
            return disp_up,disparity_predictions
        else:
            return disp_up
        
        
        