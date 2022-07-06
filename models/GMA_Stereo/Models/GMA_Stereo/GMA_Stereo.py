import sys
from turtle import left
sys.path.append("..")
from core.extractor import BasicEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from CostVolume.build_cost_volume import CostVolume
from core.gma import Attention,Aggregate
from core.estimation import DisparityEstimation
from disparity_update import CMAUpdateBlock
from CostVolume.LocalCostVolume import PyrmaidCostVolume

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




class GMAStereo(nn.Module):
    def __init__(self,
                 radius = 2,
                 num_levels =3,
                 max_disp=192,
                 dropout=0.):
        super().__init__()
        
        # HyperParameters
        hdim = 128
        cdim = 128
        self.max_disp = max_disp
        self.hidden_dim  = 128
        self.context_dim = 128
        self.radius = radius
        self.num_levels = num_levels
        
        
        
        # Feature NetWork, Context Network, and update Blocks
        self.fnet = BasicEncoder(output_dim=256,norm_fn='instance',dropout=dropout)
        self.cnet = BasicEncoder(output_dim=256,norm_fn='batch',dropout=dropout)
        self.att = Attention(args='none', 
                             dim=cdim, 
                             heads=1, 
                             max_pos_size=160, dim_head=cdim)
        self.update_block = CMAUpdateBlock(hidden_dim=128,cost_volume_dimension=self.num_levels*(2*self.radius+1),
                                           num_heads=1)
        
        
        # inital Cost volume
        self.inital_correlation_cost_volume = CostVolume(max_disp=self.max_disp//8,feature_similarity='correlation')
        
        match_similarity = True
        
        self.disp_estimation = DisparityEstimation(max_disp=192//8,match_similarity=match_similarity) 
        
        self.pyramid_cost_volume = PyrmaidCostVolume(radius=self.radius,
                                                     nums_levels=self.num_levels,
                                                     sample_points=self.radius *2)
        
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 **3
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)
    
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    
    def forward(self,left_image,right_image,iters=12,disp_init=None,upsample=True,test_mode=False):
        
        left_image = left_image.contiguous()
        right_image = right_image.contiguous()
        
        hdim = self.hidden_dim
        cdim = self.context_dim
        
        # run the feature network
        with autocast(enabled=True):
            fmap1, fmap2 = self.fnet([left_image, right_image])
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        

        # run the context network
        with autocast(enabled=True):
            cnet = self.cnet(left_image)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            # Hidden State
            net = torch.tanh(net)
            # Context Feature
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        
        # Get a initial disparity using 4D correlation cost volume.
        correlation_cost_volume = self.inital_correlation_cost_volume(fmap1,fmap2)
        disp_initial = self.disp_estimation(correlation_cost_volume)
        disp_initial = disp_initial.unsqueeze(1)
        
        if disp_init is None:
            disp_init = disp_initial
        
        cur_disp = disp_init
        ref_coords = torch.zeros_like(disp_initial).type_as(correlation_cost_volume)
        
        disparity_predictions = []
        
        for itr in range(iters):
            cur_disp = cur_disp.detach()
            corr = self.pyramid_cost_volume(correlation_cost_volume,self.radius,cur_disp)
            disp= cur_disp - ref_coords
            with autocast(enabled=True):
                net, up_mask, delta_disp = self.update_block(net, inp, corr, disp, attention)
            
            # Update the current disparity
            cur_disp = F.relu(cur_disp + delta_disp,True)
            
            # Upsample
            disp_up = self.upsample_flow(cur_disp,up_mask)
            
            disparity_predictions.append(disp_up)
        
        if test_mode:
            return cur_disp,disp_up
        
        return disparity_predictions
            
            
        
if __name__=="__main__":
    
    # HyperParameters
    dropout_rate = 0.
        
    inputs = torch.randn(1,3,320,640).cuda()
    gma_stereo = GMAStereo(dropout=dropout_rate,max_disp=192,radius=2,num_levels=3).cuda()
    
    # Test the inputs
    disparity_predictions = gma_stereo(inputs,inputs)
    
    for d in disparity_predictions:
        print(d.shape)