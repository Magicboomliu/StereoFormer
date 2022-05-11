from distutils.command.build import build
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def conv3d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm3d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))

class StereoNetAggregation(nn.Module):
    def __init__(self, in_channels=32, hidden_layers=32):
        super(StereoNetAggregation, self).__init__()

        aggregation_modules = nn.ModuleList()

        aggregation_modules.append(conv3d(in_channels,hidden_layers))
        # StereoNet uses four 3d conv
        for _ in range(3):
            aggregation_modules.append(conv3d(hidden_layers, hidden_layers))
        self.aggregation_layer = nn.Sequential(*aggregation_modules)

        # Squeeze
        self.final_conv = nn.Conv3d(hidden_layers, 1, kernel_size=3, stride=1,
                                    padding=1, bias=True)

    def forward(self, cost_volume):
        assert cost_volume.dim() == 5  # [B, C, D, H, W]

        out = self.aggregation_layer(cost_volume)
        out = self.final_conv(out)  # [B, 1, D, H, W]
        out = out.squeeze(1)  # [B, D, H, W]

        return out

class GroupWiseCorrelationCostVolume(nn.Module):
    def __init__(self,max_disp,groups,is_concated=False):
        super(GroupWiseCorrelationCostVolume,self).__init__()
        self.max_disp = max_disp
        self.groups = groups
        self.is_concated = is_concated
        
        if self.is_concated:
            self.cost_volume_aggregation = StereoNetAggregation(in_channels=256+16,hidden_layers=32)
        else:
            self.cost_volume_aggregation = StereoNetAggregation(in_channels=16,hidden_layers=32)
        
        
    def forward(self,left_feature,right_feature):
        
        gwc_cost_volume = build_gwc_volume(left_feature,right_feature,maxdisp=self.max_disp,num_groups=16)
        concated_cost_volume = build_concat_volume(left_feature,right_feature,maxdisp=self.max_disp)
        if self.is_concated:
            volume = torch.cat((gwc_cost_volume,concated_cost_volume),dim=1)
        else:
            volume = gwc_cost_volume
        
        # Cost Volume Aggregation
        volume = self.cost_volume_aggregation(volume)
        
        return volume
        






# 1/8 Cost Volume 
if __name__=="__main__":
    left_feature = torch.randn(1,128,40,80)
    right_feature = torch.randn(1,128,40,80)
    
    
    # GroupWise Cost Volume
    gwc_cost_volume_op = GroupWiseCorrelationCostVolume(max_disp=192//8,groups=16,is_concated=True)
    
    gwc_cost_volume = gwc_cost_volume_op(left_feature,right_feature)
    
    print(gwc_cost_volume.shape)
    
