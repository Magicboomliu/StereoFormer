import imp
import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.hrnet import hrnet18
from models.residual.resnet import ResBlock
from utils.devtools import print_tensor_shape
from models.submodule import convbn,hourglass3d,build_concat_volume,build_gwc_volume,convbn_3d,disparity_regression
import math

class HRNet_Stereo(nn.Module):
    def __init__(self,max_disp=192,
                 pretrain=False,
                 use_feature_fusion=False,
                 use_concated_volume=False):
        super(HRNet_Stereo,self).__init__()
        
        self.max_disp = max_disp
        self.use_feature_fusion = use_feature_fusion
        self.use_concated_volume = use_concated_volume
        
        # Feature Extraction
        self.encoder = hrnet18(pretrained=pretrain,progress=False)
        
        if self.use_feature_fusion:
            # Feature Fusion
            out_dim = 12
            self.feature_fusion = ConcatHead(hrn_out_dim=270,out_dim=12)
        
        # Group-wise correlation
        self.nums_groups = 30
        
        
        # cost volume aggregation, based on 3d Convolution
        self.dres0 = nn.Sequential(convbn_3d(self.nums_groups + out_dim * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
        
        self.dres2 = hourglass3d(32)

        self.dres3 = hourglass3d(32)

        self.dres4 = hourglass3d(32)
        
        # Classification
        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            
    
        
    def forward(self,left,right,is_training=False):
        
        left_feature = self.encoder(left) #[1/4,1/8,1/16,1/32]
        right_feature = self.encoder(right) #[1/4,1/8,1/16,1/32]
        
        # concated feature
        if self.use_feature_fusion:
            left_feature_gwc,concated_left_feature = self.feature_fusion(left_feature)
            right_feature_gwc,concated_right_feature = self.feature_fusion(right_feature)
        
        # Cost Volume Building
        # GWC cost volume :[B,G,D//4,H,W]
        gwc_volume = build_gwc_volume(left_feature_gwc,right_feature_gwc,self.max_disp//4,self.nums_groups)
        if self.use_concated_volume:
            concated_volume = build_concat_volume(concated_left_feature,concated_right_feature,self.max_disp//4)
            
            cost_volume = torch.cat((gwc_volume,concated_volume),dim=1)
        else:
            cost_volume = gwc_volume
        
        # Cost volume shape [B,2Concated_C+G,H//4,W//4]
        cost0 = self.dres0(cost_volume)
        cost0 = self.dres1(cost0) + cost0
        # Cost volume aggregation
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)


        if is_training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.upsample(cost0, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.max_disp)

            cost1 = F.upsample(cost1, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.max_disp)

            cost2 = F.upsample(cost2, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.max_disp)

            cost3 = F.upsample(cost3, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.max_disp)
            return [pred3, pred2, pred1, pred0]
        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.max_disp)
            return [pred3]
            


class ConcatHead(nn.Module):
    '''
    把 backbone 的多尺度输出合在一起
    '''
    def __init__(self,hrn_out_dim,out_dim=12):
        super().__init__()
        self.agg_conv = nn.Sequential(convbn(hrn_out_dim, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, out_channels=out_dim, kernel_size=1, padding=0, stride=1,
                                                    bias=False))
        # self.feature_aggregation = ResBlock(n_in=hrn_out_dim,n_out=out_dim,kernel_size=3,stride=1)

    def forward(self, x_list):
        upsample_list = []
        for id, x in enumerate(x_list):
            upsample_list.append(F.interpolate(x, scale_factor=int(1<<id), mode="bilinear") )
        upsample_out_gwc = torch.cat(upsample_list, dim=1)
        # upsample_out = self.feature_aggregation(upsample_out)
        upsample_out = self.agg_conv(upsample_out_gwc)
        
        return upsample_out_gwc,upsample_out

if __name__=="__main__":
    
    # Test Input
    left_input = torch.randn(1,3,320,640).cuda()
    right_input = torch.randn(1,3,320,640).cuda()
    
    hrnet_stereo = HRNet_Stereo(pretrain=False,max_disp=192,use_feature_fusion=True,use_concated_volume=True).cuda()
    
    disparity_pyramid = hrnet_stereo(left_input,right_input,True)
    
    print_tensor_shape(disparity_pyramid)
    # print("------------------------")
    # print_tensor_shape(right_feature)