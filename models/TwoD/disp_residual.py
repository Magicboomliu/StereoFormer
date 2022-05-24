
import sys
sys.path.append("../..")
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from utils.disparity_warper import disp_warp
from deform.modules.modulated_deform_conv import ModulatedDeformConvPack

def build_corr(img_left, img_right, max_disp=40):
    B, C, H, W = img_left.shape
    volume = img_left.new_zeros([B, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :-i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

    volume = volume.contiguous()
    return volume


def form_cost_volume(ref_feature, tar_feature, disp):
    B, C, H, W = ref_feature.shape
    cost = Variable(torch.FloatTensor(B, C*2, disp, H, W).zero_()).cuda()
    for i in range(disp):
        if i > 0:
            cost[:, :C, i, :, i:] = ref_feature[:, :, :, i:]
            cost[:, C:, i, :, i:] = tar_feature[:, :, :, :-i]
        else:
            cost[:, :C, i, :, :] = ref_feature
            cost[:, C:, i, :, :] = tar_feature
    cost = cost.contiguous()
    return cost

# conv3x3 + BN + relu
def conv(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )
# conv3x3 + BN
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
        nn.BatchNorm2d(out_planes)
    )

# simple conv3x3 only    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False)

# deconv : upsample to double
def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )
# conv + relu
def conv_Relu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False),
        nn.ReLU(True)
    )
# conv3d + BatchNorm
def convbn_3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))
# Correlation Cost Volume 
def build_corr(img_left, img_right, max_disp=40):
    B, C, H, W = img_left.shape
    volume = img_left.new_zeros([B, max_disp, H, W])
    for i in range(max_disp):
        if i > 0:
            volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :-i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

    volume = volume.contiguous()
    return volume
# Concated Cost Volume
def form_cost_volume(ref_feature, tar_feature, disp):
    B, C, H, W = ref_feature.shape
    cost = Variable(torch.FloatTensor(B, C*2, disp, H, W).zero_()).cuda()
    for i in range(disp):
        if i > 0:
            cost[:, :C, i, :, i:] = ref_feature[:, :, :, i:]
            cost[:, C:, i, :, i:] = tar_feature[:, :, :, :-i]
        else:
            cost[:, :C, i, :, :] = ref_feature
            cost[:, C:, i, :, :] = tar_feature
    cost = cost.contiguous()
    return cost



class res_submodule_attention(nn.Module):

    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_attention, self).__init__()

        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        
        #Spatial Attention Module here
        self.attention = SA_Module(input_nc=10)

        # input convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, feature):
        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        # dummy_flow = torch.autograd.Variable(torch.zeros(disp.data.shape).cuda())
        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
  
        # call new warp function
        left_rec ,mask= disp_warp(right,disp_)
    
        error_map = left_rec - left

        # This is the attention's input
        query = torch.cat((left, right, error_map, disp_), dim=1)
        # Attention Here
        attention_map = self.attention(query)
        # attention feature
        attented_feature = attention_map * torch.cat((feature,query), dim=1)

        # ResBlocks
        conv1 = self.conv1(attented_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(attented_feature), inplace=True)

        res = self.res(conv6) * scale
        return res

# Residual Prediction with Surface Normal
class res_submodule_with_trans_deform(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule_with_trans_deform, self).__init__()
        
        
        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        if scale ==1 or scale ==0:
            self.attention = SA_Module(input_nc=10+32)
        else:
            self.attention = SA_Module(input_nc=10+64)
            

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True),

        )
        
        self.conv2 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*2,out_planes*2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True),
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )
        
        self.conv4 = nn.Sequential(
            ModulatedDeformConvPack(out_planes*4,out_planes*4,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True),
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        self.conv6 = nn.Sequential(
            
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, transformer_feature,feature):
     
        scale = left.size()[2] / disp.size()[2]
        left = self.pool(left)
        right = self.pool(right)

        # dummy_flow = torch.autograd.Variable(torch.zeros(disp.data.shape).cuda())
        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
        left_rec,mask = disp_warp(right,disp_)
        error_map = left_rec -left

        query = torch.cat((left, right, disp_,error_map,transformer_feature), dim=1)
        attention = self.attention(query)
        attention_feature = attention * torch.cat((left, right, disp_, error_map, transformer_feature,feature), dim=1)
        conv1 = self.conv1(attention_feature)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp_, error_map, transformer_feature,feature), dim=1)))
        res = self.res(conv6) * scale
        return res




# Simple ConvGRU
class ConvGRU(nn.Module):
    def __init__(self,hidden_dimension=64,input_dimension=128):
        super(ConvGRU,self).__init__()
        self.convz = nn.Conv2d(hidden_dimension+input_dimension,hidden_dimension,3,padding=1)
        self.convr = nn.Conv2d(hidden_dimension+input_dimension,hidden_dimension,3,padding=1)
        self.convq = nn.Conv2d(hidden_dimension+input_dimension,hidden_dimension,3,padding=1)
        
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

# Seprate X Y GRU
class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        # Kept Gate
        z = torch.sigmoid(self.convz1(hx))
        # Forget Gate
        r = torch.sigmoid(self.convr1(hx))
        
        # H detlta
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h



# Residual Update UpSampling
class ResidualUpdateModule(nn.Module):
    def __init__(self):
        super(ResidualUpdateModule,self).__init__()
        
    def forward(self,x):
        ''' 
        Left feature CNN 
        Left feature Transformer
        Corase Disparity
        Left image / Right Image
        local Lost volume?
        
        '''
        pass


# Basic upsample
class ConvAffinityUpsample(nn.Module):
    def __init__(self,input_channels,hidden_channels=128):
        super(ConvAffinityUpsample,self).__init__()
        self.upsample_mask = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=hidden_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels,8*8*9,1,padding=0)
        )
    
    def forward(self,feature):
        
        mask = .25 * self.upsample_mask(feature)
        
        return mask


def upsample_convex8(disp, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = disp.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)
        

    up_disp = F.unfold(8 * disp, [3,3], padding=1)
        # up_flow: [B,C*kW*kH,L] here with padding the L should be H*W
        
    up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

    up_disp = torch.sum(mask * up_disp, dim=2)
       
    up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        
    return up_disp.reshape(N, 1, 8*H, 8*W)


def upsample_simple8(disp, mode='bilinear'):
    new_size = (8 * disp.shape[2], 8 * disp.shape[3])
    return  8 * F.interpolate(disp, size=new_size, mode=mode, align_corners=True)






# Spatial Attention
class SA_Module(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=16):
        super(SA_Module, self).__init__()
        self.attention_value = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            nn.Conv2d(ndf, output_nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_value = self.attention_value(x)
        return attention_value