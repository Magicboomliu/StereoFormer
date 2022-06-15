import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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



class PureLearningDynamicLocalCostVolume(nn.Module):
    def __init__(self,sample_points=10):
        super(PureLearningDynamicLocalCostVolume,self).__init__()
        
        self.sample_points = sample_points
        
    
    def forward(self,x):
        pass