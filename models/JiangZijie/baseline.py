import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from models.GMA_Stereo.Models.GMA_Stereo.CostVolume.build_cost_volume import CostVolume
from models.GMA_Stereo.Models.GMA_Stereo.core.estimation import DisparityEstimation
from models.GMA_Stereo.Models.GMA_Stereo.CostVolume.LocalCostVolume import PyrmaidCostVolume
from models.GMA_Stereo.Models.GMA_Stereo.core.extractor import BasicEncoder
from models.GMA_Stereo.Models.GMA_Stereo.recurrent_refinement import GRURefinemnet
from torch.nn.init import kaiming_normal

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
                 radius = 3,
                 num_levels =3,
                 max_disp=320,
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
        
        match_similarity = True
        self.disp_estimation = DisparityEstimation(max_disp=self.max_disp//8,match_similarity=match_similarity)
        
        self.pyramid_cost_volume = PyrmaidCostVolume(radius=self.radius,
                                                     nums_levels=self.num_levels,
                                                     sample_points=self.radius *2)

       # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv3d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        
        # Encoder Feature
        matching_feature = [torch.split(feat,[batch_size,batch_size],dim=0) for feat in cost_matching_feat_list]
        
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
        
        # Refinement With A Local Cost VolumeS
        print(disp_initial.shape)

    


if __name__ =="__main__":
        # HyperParameters
    dropout_rate = 0.
        
    inputs = torch.randn(1,3,320,640).cuda()
    gma_stereo = GMAStereo(dropout=dropout_rate,max_disp=192,radius=2,num_levels=3).cuda()
    
    # Test the inputs
    outputs = gma_stereo(inputs,inputs)
    
    
    
    pass