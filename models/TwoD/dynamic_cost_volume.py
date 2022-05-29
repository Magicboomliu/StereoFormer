from math import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
from disparity_warper import disp_warp

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


# RAFT based iterative refinement
class DynamicCostVolumeRefinement(nn.Module):
    def __init__(self,input_channels=132,output_channels=64):
        super(DynamicCostVolumeRefinement,self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.search_range = SmallUNet(input_channels=self.input_channels,hidden_layer=self.output_channels)
        
        self.offset = nn.Sequential(nn.Conv2d(self.output_channels,2,3,1,1),
                                        nn.ReLU(inplace=True))
        
    
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
        valid_range_lower = lower_bound.max()
        valid_range_upper = upper_bound.max()

        # Searching Bound
        lower_index = disp - lower_bound
        upper_index = disp + upper_bound
        
        
        # Build A Local Cost Volime( Lower + Higher)
        maximum_searching_range = lower_bound + upper_bound
        



if __name__=="__main__":
    
    # All the inputs
    left_image = torch.randn(1,3,40,80).cuda()
    right_image = torch.randn(1,3,40,80).cuda()
    left_feature = torch.randn(1,128,40,80).cuda()
    right_feature = torch.randn(1,128,40,80).cuda()
    disp = torch.abs(torch.randn(1,1,40,80)).cuda()
    
    dynamic_cost_volume = DynamicCostVolumeRefinement().cuda()
    
    dynamic_cost_volume(left_feature,right_feature,disp,left_image,right_image)
    
    