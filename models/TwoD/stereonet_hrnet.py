import sys
sys.path.append("../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.devtools import print_tensor_shape
from models.backbone.hrnet import hrnet18

# left -right
def make_cost_volume(left, right, max_disp):
    cost_volume = torch.ones(
        (left.size(0), left.size(1), max_disp, left.size(2), left.size(3)),
        dtype=left.dtype,
        device=left.device,
    )

    cost_volume[:, :, 0, :, :] = left - right
    for d in range(1, max_disp):
        cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]

    return cost_volume


def conv_3x3(in_c, out_c, s=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, s, d, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


def conv_1x1(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            conv_3x3(c0, c0, d=dilation),
            conv_3x3(c0, c0, d=dilation),
        )

    def forward(self, input):
        x = self.conv(input)
        return x + input


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        d = [1, 2, 4, 8, 1, 1]
        self.conv0 = nn.Sequential(
            conv_3x3(4, 32),
            *[ResBlock(32, d[i]) for i in range(6)],
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, disp, rgb):
        disp = (
            F.interpolate(disp, scale_factor=2, mode="bilinear", align_corners=False)
            * 2
        )
        rgb = F.interpolate(
            rgb, (disp.size(2), disp.size(3)), mode="bilinear", align_corners=False
        )
        x = torch.cat((disp, rgb), dim=1)
        x = self.conv0(x)
        return F.relu(disp + x)


class Concated_Head(nn.Module):
    def __init__(self,select_res='1/8'):
        super(Concated_Head,self).__init__()
        self.select_res = select_res
        self.downsample = nn.Sequential(
            nn.Conv2d(270,128,3,1,1,bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
    def forward(self,feature_list):
        upsample_list = []
        if self.select_res=='1/8':
            upsample_list.append(F.interpolate(feature_list[0],scale_factor=1/2,mode='bilinear'))
            upsample_list.append(feature_list[1])
            upsample_list.append(F.interpolate(feature_list[2],scale_factor=2.0,mode='bilinear'))
            upsample_list.append(F.interpolate(feature_list[3],scale_factor=4.0,mode='bilinear'))
        elif self.select_res=='1/4':
            upsample_list.append(F.interpolate(feature_list[0]))
            upsample_list.append(F.interpolate(feature_list[1],scale_factor=2.0,mode='bilinear'))
            upsample_list.append(F.interpolate(feature_list[2],scale_factor=4.0,mode='bilinear'))
            upsample_list.append(F.interpolate(feature_list[3],scale_factor=8.0,mode='bilinear'))    
        else:
            raise NotImplementedError
        upsample_out = torch.cat(upsample_list,dim=1)
        upsample_out = self.downsample(upsample_out)
        
        return upsample_out
        


class StereoNet_HR(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 3
        self.align = 2 ** self.k
        self.max_disp = (192 + 1) // (2 ** self.k)
        # Swin-Former Tiny
        self.feature_extractor = hrnet18(False,False)
        self.feature_fusion = Concated_Head(select_res='1/8')
        
        self.cost_filter = nn.Sequential(
            nn.Conv3d(128, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 1, 3, 1, 1),
        )
        self.refine_layer = nn.ModuleList([RefineNet() for _ in range(self.k)])

    def forward(self, left_img, right_img,is_training=False):
        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))

        # 1/4 Scale Feature
        lf_list = self.feature_extractor(left_img)
        rf_list = self.feature_extractor(right_img)
        
        lf = self.feature_fusion(lf_list)
        rf = self.feature_fusion(rf_list)
        
        
        # 1/4 Scale Feature
        cost_volume = make_cost_volume(lf, rf, self.max_disp)
        cost_volume = self.cost_filter(cost_volume).squeeze(1)
        

        x = F.softmax(cost_volume, dim=1)
        d = torch.arange(0, self.max_disp, device=x.device, dtype=x.dtype)
        # 1/8 Disparity 
        x = torch.sum(x * d.view(1, -1, 1, 1), dim=1, keepdim=True)
        
        # add 1/8 disparity into disparity list
        multi_scale = []
        scale = left_img.size(3)//x.size(3)
        full_res = F.interpolate(x * scale, left_img.shape[2:])[:, :, :h, :w]
        multi_scale.append(full_res)
        
        # Refinement and add the refinement result into disparity result
        for refine in self.refine_layer:
            x = refine(x, left_img)
            scale = left_img.size(3) / x.size(3)
            full_res = F.interpolate(x * scale, left_img.shape[2:])[:, :, :h, :w]
            multi_scale.append(full_res)

        if is_training:
            return multi_scale
        else:
            return multi_scale[-1]


if __name__ == "__main__":

    left = torch.rand(1, 3, 320, 640)
    right = torch.rand(1, 3, 320, 640)
    model = StereoNet_HR()
    
    output = model(left,right,False)
    # print(output['disp'].shape)
    # print("---------------")
    print_tensor_shape(output)
