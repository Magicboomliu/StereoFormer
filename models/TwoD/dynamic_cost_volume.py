from math import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
from disparity_warper import disp_warp

# Differetial Round Operation
def ste_round(x):
    return torch.round(x) - x.detach() + x

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

class SmallUNet(nn.Module):
    def __init__(self,input_channels,hidden_layer=32):
        super(SmallUNet,self).__init__()
        self.input_channels = input_channels
        self.hidden_layers = hidden_layer
        self.conv1_a = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=hidden_layer,kernel_size=3,stride=2,
            padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer),
            nn.ReLU(True)
        )
        self.conv1_b = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer,out_channels=hidden_layer,kernel_size=3,stride=1,
            padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer),
            nn.ReLU(True)
        )
        # 1/4
        self.conv2_a = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer,out_channels=hidden_layer*2,kernel_size=3,stride=2,
            padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer*2),
            nn.ReLU(True)
        )
        self.conv2_b = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer*2,out_channels=hidden_layer*2,kernel_size=3,stride=1,
            padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer*2),
            nn.ReLU(True)
        )
        # Upsample Phase
        self.upconv_1 = convt_bn_relu(hidden_layer*2,hidden_layer,3,stride=2,padding=1,output_padding=1)
        self.upconv_2 = convt_bn_relu(hidden_layer*2,hidden_layer,3,stride=2,padding=1,output_padding=1)
        
    def forward(self,x):
        # 1/2
        feat1_a = self.conv1_a(x)
        feat1_b = self.conv1_b(feat1_a)
        #1/4
        feat2_a = self.conv2_a(feat1_b)
        feat2_b = self.conv2_b(feat2_a)
        # Upsampled to 1/2
        upconv2 = self.upconv_1(feat2_b) # 32
        # Upsampled to Full Size
        upconv1 = self.upconv_2(torch.cat((upconv2,feat1_b),dim=1))
        
        return upconv1

def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid

def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid

# get warped Image
def get_warped_image(img,disp,padding_mode='border'):
    
    assert disp.min()>=0
    # Get a grid of the current left image
    grid = meshgrid(img)
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode='nearest', padding_mode=padding_mode)
    
    # Valid mask is important
    original_coordinate_X,original_coordiante_Y = torch.chunk(grid,2,1)
    after_sample = original_coordinate_X  - disp
    valid_mask = after_sample >=0
    valid_mask = valid_mask.float()

    # mask = torch.ones_like(img)
    # valid_mask = F.grid_sample(mask, sample_grid, mode='nearest', padding_mode='zeros')
    # valid_mask[valid_mask < 0.9999] = 0
    # valid_mask[valid_mask > 0] = 1        
    return warped_img,valid_mask

def build_adapative_local_cost_volume(img_left,img_right,cur_disp,lowwer_bound=3,upper_bound=3):
    
    # Fix Length Cost Volume
    B,C,H,W = img_left.shape 
    round_cur_disp = ste_round(cur_disp)
    
    searching_range = lowwer_bound + upper_bound + 1
    searching_range = searching_range.int()
    # Cost Volume Here
    volume = img_left.new_zeros([B,searching_range,H,W])
    
    # Max Searching Range
    max_searching_area = img_left.new_zeros([B,searching_range,H,W])
    
    for i in range(searching_range):
        cur_candiate_disp_offset = i -lowwer_bound
        cur_disp_candidate = round_cur_disp + cur_candiate_disp_offset
        max_searching_area[:,i,:,:] = cur_disp_candidate.squeeze(1)
        ambigous_mask_inverse = sample_valid_mask(cur_disp_candidate)
        cur_disp_candidate = torch.clamp(cur_disp_candidate,min=0)
        warped_right_image,valid_mask = get_warped_image(img_right,cur_disp_candidate)
        volume[:,i,:,:] = (img_left * warped_right_image*valid_mask*ambigous_mask_inverse).mean(dim=1)
        
    volume =  volume.contiguous()
    
    return volume,max_searching_area

def sample_valid_mask(disp,min=0):
    valid_mask = disp>=0
    
    return valid_mask.float()

# Select the Suitable Cost Volume
def select_suitable_cost_volume(adaptive_cost_volume,lower_index_map,upper_index_map,
                                disp,max_searching_range,scale=1):
    B,C,H,W = disp.shape
    valid_lower_index = ste_round(lower_index_map)
    valid_upper_index = ste_round(upper_index_map)
    valid_lower_index = torch.clamp(valid_lower_index,min=0,max=W-1)
    valid_upper_index = torch.clamp(valid_upper_index,min=0,max=W-1)

    round_disp = ste_round(disp)
    
    # Max_searching_range [B,D,H,W]
    # lower threshold [B,1,H,W]
    # upper threshold [B,1,H,W]
  
    valid_mask1 = max_searching_range>=(round_disp-valid_lower_index)
    valid_mask1 = valid_mask1.float()
    
    valid_mask2 = max_searching_range<=(round_disp+valid_upper_index)
    valid_mask2 = valid_mask2.float()
    
    complete_valid_mask = valid_mask1 * valid_mask2
    adaptive_cost_volume = adaptive_cost_volume * complete_valid_mask + (-1000)*(1.0-complete_valid_mask)
   
    adaptive_cost_volume = adaptive_cost_volume.float()
    
    score = torch.softmax(adaptive_cost_volume/(scale*1.0),dim=1)


    return adaptive_cost_volume,score,complete_valid_mask

def disparity_residual_learning(score,max_lower_bound,max_upper_bound):
  searching_range = (max_lower_bound + max_upper_bound+1).item()
  residual_range = torch.arange(searching_range).type_as(max_lower_bound) - max_lower_bound
  residual_range = residual_range.view(1,int(searching_range),1,1)

  disp_residual = torch.sum((score * residual_range),dim=1).unsqueeze(1)
  
  return disp_residual


# DyamicCostVolumeBlocks
def DynamicCostVolumeBlock(img_left,img_right,cur_disp,lower_offset_map,upper_offset_map,scale=1):    
    # Make Sure the Searching Range
    max_lowwer_bound = lower_offset_map.max()
    max_lowwer_bound = ste_round(max_lowwer_bound)
    max_upper_bound = upper_offset_map.max()
    max_upper_bound = ste_round(max_upper_bound)
    
    # Get Max Cost Volume
    adaptive_cost_volume, max_searching_area = build_adapative_local_cost_volume(img_left,img_right,cur_disp,
                                                             max_lowwer_bound,max_upper_bound)
    
    adaptive_cost_volume_narrow,matching_score,complete_valid_mask = select_suitable_cost_volume(
    adaptive_cost_volume,lower_offset_map,upper_offset_map,cur_disp,max_searching_area,scale=scale)
    
    residual = disparity_residual_learning(matching_score,max_lowwer_bound,max_upper_bound)
    
    return residual, adaptive_cost_volume_narrow,complete_valid_mask
    

# RAFT based iterative refinement
class DynamicCostVolumeRefinement(nn.Module):
    def __init__(self,input_channels=132,output_channels=64,normalized_scale=1):
        super(DynamicCostVolumeRefinement,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.normalized_scale = normalized_scale
        self.search_range = SmallUNet(input_channels=self.input_channels,hidden_layer=self.output_channels)
        
        # Predict the lower bound and the upper bound
        self.offset = nn.Sequential(nn.Conv2d(self.output_channels,2,3,1,1),
                                        nn.ReLU(inplace=True))
        
        self.relu0 = nn.ReLU(inplace=True)
        
    
    def forward(self,left_feature,right_feature,disp,left_image,right_image):
        '''Local Cost Volume Refinement / RAFT/disp'''
        # Update the feature, Update the Local Cost Volume, Update the residual, Update the Disparity
        
        # Update the Searching Range
        left_clues = torch.cat((left_feature,left_image),dim=1)
        right_clus = torch.cat((right_feature,right_image),dim=1)
        warped_left = disp_warp(right_clus,disp)[0]
        error = left_clues - warped_left
        
        inputs = torch.cat((error,disp),dim=1)
        outputs = self.search_range(inputs)
        offset = self.offset(outputs)
        
        lower_bound, upper_bound = torch.chunk(offset,2,dim=1)
        
        assert disp.min()>=0
        
        residual, adaptive_cost_volume_narrow,complete_valid_mask = DynamicCostVolumeBlock(left_feature,right_feature,disp,
                                          lower_offset_map=lower_bound,
                                          upper_offset_map=upper_bound,
                                          scale=self.normalized_scale)
        
        # Update the disparity        
        disp = self.relu0(disp + residual)
        
        return { 'disp': residual,
                'adaptive_cost_volume': adaptive_cost_volume_narrow,
                'complete_valid_mask': complete_valid_mask,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound  }
        
        



if __name__=="__main__":
    
    # All the inputs
    left_image = torch.randn(1,3,40,80).cuda()
    right_image = torch.randn(1,3,40,80).cuda()
    left_feature = torch.randn(1,128,40,80).cuda()
    right_feature = torch.randn(1,128,40,80).cuda()
    disp = torch.abs(torch.randn(1,1,40,80)).cuda()
    
    dynamic_cost_volume = DynamicCostVolumeRefinement(input_channels=132,normalized_scale=1).cuda()
    
    results = dynamic_cost_volume(left_feature,right_feature,disp,left_image,right_image)
    
    print(results.keys())
    
    