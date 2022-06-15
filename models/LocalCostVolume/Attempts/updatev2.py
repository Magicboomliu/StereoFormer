import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import sys
sys.path.append("../..")
from utils.disparity_warper import disp_warp


def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x

class ConvGRU(nn.Module):
    def __init__(self, nb_channel, softsign):
        super(ConvGRU, self).__init__()
        self.conv_z = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        self.conv_b = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        self.conv_g = nn.Conv2d(nb_channel * 2, nb_channel, 3, 1, 1)
        init.orthogonal_(self.conv_z.weight)
        init.orthogonal_(self.conv_b.weight)
        init.orthogonal_(self.conv_g.weight)
        init.constant_(self.conv_z.bias, 0.)
        init.constant_(self.conv_b.bias, 0.)
        init.constant_(self.conv_g.bias, 0.)
        self.conv_zz = nn.Sequential(self.conv_z, nn.Sigmoid())
        self.conv_bb = nn.Sequential(self.conv_b, nn.Sigmoid())
        if not softsign:
            self.conv_gg = nn.Sequential(self.conv_g, nn.Tanh())
        else:
            self.conv_gg = nn.Sequential(self.conv_g, nn.Softsign())
        self.nb_channel = nb_channel
    def forward(self, input, prev_h):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        if prev_h is None:
            prev_h = torch.autograd.Variable(torch.zeros(batch_size, self.nb_channel, row, col)).cuda()
        x1 = torch.cat((input, prev_h), 1)
        # Z is the forget gate
        z = self.conv_zz(x1)
        # B is the remember gate
        b = self.conv_bb(x1) 
        s = b * prev_h
        s = torch.cat((s, input), 1)
        g = self.conv_gg(s)
        h = (1 - z) * prev_h + z * g
        return h

class BasicGuidanceNet(nn.Module):
    def __init__(self,input_channels,hidden_layer=32):
        super(BasicGuidanceNet,self).__init__()
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
    
    def forward(self,old_cost_volume,cur_disp,left_image,right_image):
        B,D,H,W = old_cost_volume.shape
        #get the disparity variance along the baseline
        prob_volume = F.softmax(old_cost_volume, dim=1)  # [B, D, H, W]
        disparity_candidates = torch.arange(D).view(1,D,1,1).repeat(1,1,H,W).type_as(cur_disp)
        if cur_disp.size(-1)!=left_image.size(-1):
            cur_left_image = F.interpolate(left_image,size=[cur_disp.size(-2),cur_disp.size(-1)],
                                           mode='bilinear',align_corners=False)
            cur_right_image = F.interpolate(right_image,size=[cur_disp.size(-2),cur_disp.size(-1)],
                                            mode='bilinear',align_corners=False)
        else:
            cur_left_image = left_image
            cur_right_image = right_image
        # Error Map
        warped_left,valid_mask = disp_warp(cur_right_image,cur_disp)
        error_map = warped_left - cur_left_image

        # get uncertainity Cost volume
        uncertainity_volume = prob_volume * disparity_candidates *(disparity_candidates-cur_disp)*(disparity_candidates-cur_disp)

        # Get Input Feature for GRU
        error_feature = self.disparity_error_encoder(error_map)
        uncertainity_feature = self.uncertain_encoder(uncertainity_volume)
        
        return torch.cat([error_feature,uncertainity_feature],dim=1)

class OffsetPredictionHead(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(OffsetPredictionHead,self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))

class DisparityUpdateDLC(nn.Module):
    def __init__(self,input_channels,
                 hidden_dim,
                 sample_points=10):
        super(DisparityUpdateDLC,self).__init__()
        self.sample_points = sample_points
        self.hidden_dim = hidden_dim
        
        self.encoder = BasicGuidanceNet(input_channels=input_channels,hidden_layer=self.hidden_dim)
        
        self.gru = ConvGRU(nb_channel=self.hidden_dim*2,softsign=False)
        
        self.offset = OffsetPredictionHead(input_dim=32,hidden_dim=64)
        
    def forward(self,old_cost_volume,cur_disp,
                left_image,right_image,hidden_state=None,
                consider_valid=True):
        
        B,D,H,W = old_cost_volume.shape
        # GRU feature
        offset_features = self.encoder(old_cost_volume,cur_disp,left_image,right_image)
        
        # GET GRU feature here
        hidden_state = self.gru(offset_features,hidden_state)
        
        bounds = self.offset(hidden_state)
        
        lower_bound,upper_bound = torch.chunk(bounds,2,dim=1)
        # Get the searching range
        lower_bound = cur_disp - lower_bound
        upper_bound = cur_disp + upper_bound
        
        #Get Sampling Intervals
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
          sampling_candidates = sampling_candidates * (1-invalid_mask) + invalid_mask * cur_disp.repeat(1,self.sample_points+1,1,1)        
        
        # Get Cost Volume Here
        local_cost_volume = build_cost_volume_from_volume(old_cost_volume,sampling_candidates)
        # Get Score Map Here
        score_map = torch.softmax(local_cost_volume,dim=1)
        disp = torch.sum(score_map*sampling_candidates,dim=1)

        # Get Disparity Here
        disp = disp.unsqueeze(1)
        
        
        return disp,hidden_state

class DisparityUpdateDLCWithMask(nn.Module):
    def __init__(self,input_channels,
                 feature_dim,
                 hidden_dim,
                 sample_points=10):
        super(DisparityUpdateDLCWithMask,self).__init__()
        self.sample_points = sample_points
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        self.encoder = BasicGuidanceNet(input_channels=input_channels,hidden_layer=self.hidden_dim)
        
        self.feature_encode =nn.Sequential(
            nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.gru = ConvGRU(nb_channel=self.hidden_dim*2+self.feature_dim,softsign=False)
        
        self.offset = OffsetPredictionHead(input_dim=self.hidden_dim*2+self.feature_dim,hidden_dim=64)
        
        self.mask = nn.Sequential(
            nn.Conv2d(self.hidden_dim*2+self.feature_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        
    def forward(self,old_cost_volume,cur_disp,
                left_image,right_image,hidden_state=None,
                left_feature=None,
                consider_valid=True):
        
        B,D,H,W = old_cost_volume.shape
        # GRU feature
        offset_features = self.encoder(old_cost_volume,cur_disp,left_image,right_image)
        left_feature = self.feature_encode(left_feature)
        input_features = torch.cat((offset_features,left_feature),dim=1)
        # GET GRU feature here
        hidden_state = self.gru(input_features,hidden_state)
        
        
        mask = .25 * self.mask(hidden_state)
        
        bounds = self.offset(hidden_state)
        
        lower_bound,upper_bound = torch.chunk(bounds,2,dim=1)
        # Get the searching range
        lower_bound = cur_disp - lower_bound
        upper_bound = cur_disp + upper_bound
        
        #Get Sampling Intervals
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
          sampling_candidates = sampling_candidates * (1-invalid_mask) + invalid_mask * cur_disp.repeat(1,self.sample_points+1,1,1)        
        
        # Get Cost Volume Here
        local_cost_volume = build_cost_volume_from_volume(old_cost_volume,sampling_candidates)
        # Get Score Map Here
        score_map = torch.softmax(local_cost_volume,dim=1)
        disp = torch.sum(score_map*sampling_candidates,dim=1)

        # Get Disparity Here
        disp = disp.unsqueeze(1)
        
        
        return disp,hidden_state,mask





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
    
    gru = ConvGRU(nb_channel=40,softsign=False).cuda()
    
    information_guidance = torch.randn(1,40,80,100).cuda()
    error_information = torch.randn(1,40,80,100).cuda()
    
    h = gru(information_guidance,error_information)
    