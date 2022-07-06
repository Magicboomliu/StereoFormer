from http.client import ImproperConnectionState
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../../..")
from models.GMA_Stereo.Models.GMA_Stereo.CostVolume.build_cost_volume import CostVolume
from models.GMA_Stereo.Models.GMA_Stereo.core.estimation import DisparityEstimation
from models.GMA_Stereo.Models.GMA_Stereo.CostVolume.LocalCostVolume import PyrmaidCostVolume
from models.GMA_Stereo.Models.GMA_Stereo.core.extractor import BasicEncoder
from models.GMA_Stereo.Models.GMA_Stereo.recurrent_refinement import GRURefinemnet


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
        self.max_disp = max_disp
        self.hidden_dim  = 128
        self.context_dim = 128
        self.radius = radius
        self.num_levels = num_levels
        
        
        # Feature NetWork, Context Network, and update Blocks
        self.fnet = BasicEncoder(output_dim=256,norm_fn='instance',dropout=dropout)
        self.cnet = BasicEncoder(output_dim=256,norm_fn='batch',dropout=dropout)
        
        
        # inital Cost volume
        self.inital_correlation_cost_volume = CostVolume(max_disp=self.max_disp//8,feature_similarity='correlation')
        self.second_scale_cost_volume = CostVolume(max_disp=self.max_disp//4,feature_similarity='correlation')
        self.third_scale_cost_volume = CostVolume(max_disp=self.max_disp//2,feature_similarity='correlation')
        
        match_similarity = True
        self.disp_estimation = DisparityEstimation(max_disp=192//8,match_similarity=match_similarity) 
        
        self.pyramid_cost_volume = PyrmaidCostVolume(radius=self.radius,
                                                     nums_levels=self.num_levels,
                                                     sample_points=self.radius *2)
        
        self.refinement_part3 = GRURefinemnet(hidden_dim=128,cost_volume_dimension=self.num_levels*(2*self.radius+1),
                                              radius=2,iters=4,upsample_rate=1.0,
                                              output_list=False)
        
        self.refinement_part2 = GRURefinemnet(hidden_dim=96,cost_volume_dimension=self.num_levels*(2*self.radius+1),
                                              radius=2,iters=4,upsample_rate=1.0,
                                              output_list=False)

        self.refinement_part1 = GRURefinemnet(hidden_dim=64,cost_volume_dimension=self.num_levels*(2*self.radius+1),
                                              radius=2,iters=4,upsample_rate=1.0,
                                              output_list=False)
    
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    
    def forward(self,left_image,right_image,
                disp_init=None,upsample=True,test_mode=False):
        
        
        disparity_outputs =[]
        
        # Left and Right Input
        left_image = left_image.contiguous()
        right_image = right_image.contiguous()
        batch_size = left_image.shape[0]
        
        hdim = self.hidden_dim
        cdim = self.context_dim
        
        # run the feature network
        with autocast(enabled=True):
            cost_matching_feat,cost_matching_feat_list= self.fnet([left_image, right_image])
        
        # 1/8 Left Feature and Right Feature, 256 Dimension.
        fmap1,fmap2 = torch.split(cost_matching_feat,[batch_size,batch_size],dim=0)
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        matching_feature = [torch.split(feat,[batch_size,batch_size],dim=0) for feat in cost_matching_feat_list]
        matching_feat1 = [p[0] for p in matching_feature]
        matching_feat2 =[p[1] for p in matching_feature]

        
        #run the context network
        with autocast(enabled=True):
            cnet,context_feat1_list = self.cnet(left_image)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            # Hidden State
            net = torch.tanh(net)
            # Context Feature
            inp = torch.relu(inp)
        
        ############### 1/8 Scale Resolution #################
        # Get a initial disparity using 4D correlation cost volume.
        correlation_cost_volume = self.inital_correlation_cost_volume(fmap1,fmap2)
        disp_initial = self.disp_estimation(correlation_cost_volume)
        disp_initial = disp_initial.unsqueeze(1)
        
        disparity_outputs.append(disp_initial)
        
        # GRU Refinement(W/O GMA)
        ############### 1/4 Scale Resolution #################
        if disp_init is None:
            disp_init = disp_initial  
        cur_disp = disp_init
        

        # 1/8 -->1/4
        disp_up = self.refinement_part3(self.pyramid_cost_volume,correlation_cost_volume,cur_disp,
                                        inp,net)
        
        disparity_outputs.append(disp_up)
        
         ############### 1/2 Scale Resolution #################
        # Coarse cur_disparity
        cur_disp = disp_up
        # Get current Cost Volume
        correlation_cost_volume_one_fourth = self.second_scale_cost_volume(matching_feat1[1],matching_feat2[1])
        # 1/4 Resolution: 96 dimesion
        net1 = torch.tanh(context_feat1_list[1])
        inp1 = torch.relu(context_feat1_list[1])
        # 1/4 -->1/2
        disp_up = self.refinement_part2(self.pyramid_cost_volume,correlation_cost_volume_one_fourth,cur_disp,
                                        inp1,net1)

        disparity_outputs.append(disp_up)
        
        ################ full scale resolution ##################
        # Coarse cur_disparity
        cur_disp = disp_up
        # Get current Cost Volume
        correlation_cost_volume_one_twice = self.third_scale_cost_volume(matching_feat1[0],matching_feat2[0])
        # 1/2 Resolution: 64 dimesion
        net0 = torch.tanh(context_feat1_list[0])
        inp0 = torch.relu(context_feat1_list[0])
        # 1/2 -->1
        disp_up = self.refinement_part1(self.pyramid_cost_volume,correlation_cost_volume_one_twice,cur_disp,
                                        inp0,net0)
        
        disparity_outputs.append(disp_up)
        
        
        if test_mode:
            return disp_up
        else:
            return disparity_outputs[::-1]
        
        
        
        
if __name__=="__main__":
    
    # HyperParameters
    dropout_rate = 0.
        
    inputs = torch.randn(1,3,320,640).cuda()
    gma_stereo = GMAStereo(dropout=dropout_rate,max_disp=192,radius=2,num_levels=3).cuda()
    
    # Test the inputs
    outputs = gma_stereo(inputs,inputs)
    
    for out in outputs:
        print(out.shape)