from turtle import left
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../..")
from models.BasicBlocks.resnet import ResBlock,DeformBlock
from utils.disparity_warper import disp_warp


def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x

def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


# Small Unet For offset prediction
class SmallUNet(nn.Module):
    def __init__(self,input_channels,hidden_layer=32):
        super(SmallUNet,self).__init__()
        self.input_channels = input_channels
        self.hidden_layers = hidden_layer
        self.uncertain_encoder = nn.Sequential(
            nn.Conv2d(self.input_channels,self.hidden_layers,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(self.hidden_layers),
            nn.ReLU(inplace=True)
        )
        self.disparity_error_encoder = nn.Sequential(
            nn.Conv2d(3,self.hidden_layers,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(self.hidden_layers),
            nn.ReLU(inplace=True)
        )
        
        self.feature_fusion1 = ResBlock(self.hidden_layers*2,self.hidden_layers,3,1)
        self.feature_fusion2 = DeformBlock(self.hidden_layers,self.hidden_layers//2,3,1)
        
        # Get Offset
        self.offset_prediction = nn.Sequential(
            nn.Conv2d(self.hidden_layers//2,2,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True)
        )
        
        
        
    def forward(self,disparity_error,uncertainity_volume):
        
        error_info = self.disparity_error_encoder(disparity_error)
        uncertainity_info = self.uncertain_encoder(uncertainity_volume)
        concatenation_info = torch.cat((error_info,uncertainity_info),dim=1)
        # Featue Fusion
        feature_fusion1 = self.feature_fusion1(concatenation_info)
        feature_fusion2 = self.feature_fusion2(feature_fusion1)
        
        offset = self.offset_prediction(feature_fusion2)
        
        upper_bound, offset_bound = torch.chunk(offset,2,dim=1)
        
        return upper_bound,offset_bound


# Pure Local Cost Volume
class PureLearningLocalCostVolume(nn.Module):
    def __init__(self,sample_points=10):
        super(PureLearningLocalCostVolume,self).__init__()
        
        self.sample_points = sample_points
        
        # Offset Prediction
        self.offset_prediction_network = SmallUNet(input_channels=24,hidden_layer=32)
        
    
    def forward(self,old_cost_volume,cur_disp,left_image,right_image,consider_valid=False):
        B,D,H,W = old_cost_volume.shape
        #get the disparity variance along the baseline
        prob_volume = F.softmax(old_cost_volume, dim=1)  # [B, D, H, W]
        disparity_candidates = torch.arange(D).view(1,D,1,1).repeat(1,1,H,W).type_as(cur_disp)
          
        if cur_disp.size(-1)!=left_image.size(-1):
            cur_left_image = F.interpolate(left_image,size=[cur_disp.size(-2),cur_disp.size(-1)],
                                           mode='bilinear',align_corners=False)
            cur_right_image = F.interpolate(right_image,size=[cur_disparity.size(-2),cur_disparity.size(-1)],
                                            mode='bilinear',align_corners=False)
        else:
            cur_left_image = left_image
            cur_right_image = right_image
        
        # Error Map
        warped_left,valid_mask = disp_warp(cur_right_image,cur_disp)
        error_map = warped_left - cur_left_image
        # get uncertainity Cost volume
        uncertainity_volume = prob_volume * disparity_candidates *(disparity_candidates-cur_disp)*(disparity_candidates-cur_disp)
        
        
        # OFFSET and UPPER Bound        
        lower_bound,upper_bound = self.offset_prediction_network(error_map,uncertainity_volume)
        # Get Sampling Intervals
        if consider_valid:
          lower_invalid_mask = (lower_bound<0).float()
          upper_invalid_mask = (upper_bound>=D-1).float()
          invalid_mask = lower_invalid_mask + upper_invalid_mask
          invalid_mask = torch.clamp(invalid_mask,max=1.0)
        else:
          lower_bound = torch.clamp(lower_bound,min=0)
          upper_bound = torch.clamp(upper_bound,min=0,max=D)
        # Get Sampling points
        sample_intervals = (upper_bound - lower_bound)*1.0/ self.sample_points
        addition_summation = (torch.arange(self.sample_points+1)).type_as(old_cost_volume)
        #[B,Sample_N+1,H,W]
        addition_summation = addition_summation.view(1,self.sample_points+1,1,1)
        
        sampling_candiate_intervals = addition_summation * sample_intervals
        # New Sampling Points
        sampling_candidates = lower_bound + sampling_candiate_intervals
        if consider_valid:
          sampling_candidates = sampling_candidates * (1-invalid_mask) + invalid_mask * cur_disparity.repeat(1,self.sample_points+1,1,1)        
        
        # Get Cost Volume Here
        local_cost_volume = build_cost_volume_from_volume(old_cost_volume,sampling_candidates)
        # Get Score Map Here
        score_map = torch.softmax(local_cost_volume,dim=1)
        disp = torch.sum(score_map*sampling_candidates,dim=1)

        # Get Disparity Here
        disp = disp.unsqueeze(1)
        
        return disp


def build_cost_volume_from_volume(old_volume,sampling_candidates):
    '''Bilinear interplolation'''
    B,D,H,W = old_volume.shape
    # CEIL AND FLOOR: Ceil and Floor 
    sample_candidate_ceil = ste_ceil(sampling_candidates)
    sample_candidate_floor = ste_floor(sampling_candidates)
    
    sample_candidate_ceil = torch.clamp(sample_candidate_ceil,min=0,max=D-1).long()
    sample_candidate_floor = torch.clamp(sample_candidate_floor,min=0,max=D-1).long()
    
    # Floor Rate      
    floor_rate =(sample_candidate_ceil- sampling_candidates)
    ceil_rate = 1.0 - floor_rate
    
    
    floor_volume = torch.gather(old_volume,dim=1,index=sample_candidate_floor)
    ceil_volume = torch.gather(old_volume,dim=1,index=sample_candidate_ceil)
    new_volume = floor_volume * floor_rate + ceil_volume * ceil_rate
    
    return new_volume
        
        
        
    


if __name__=="__main__":
    old_cost_volume = torch.abs(torch.randn(1,24,40,80)*3-2).cuda()
    cur_disparity = torch.abs(torch.randn(1,1,40,80)).cuda()
    left_image = torch.abs(torch.randn(1,3,320,640)).cuda()
    right_image = torch.abs(torch.randn(1,3,320,640)).cuda()
    
    pureLocalCostVolume = PureLearningLocalCostVolume().cuda()
    
    disp = pureLocalCostVolume(old_cost_volume,cur_disparity,left_image,right_image)    
 