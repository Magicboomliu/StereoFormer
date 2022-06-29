import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GMA_Stereo.core.gma import Aggregate



# Disparity residual prediction head.
class DisparityHead(nn.Module):
    def __init__(self,input_dim=128,hidden_dim=256):
        super(DisparityHead,self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim,hidden_dim,3,padding=1)
        self.conv2 = nn.Conv2d(hidden_dim,1,3,padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        return self.conv2(self.relu(self.conv1(x)))
        
# ConvGRU
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        # hidden state.
        hx = torch.cat([h, x], dim=1)
        # remember gate.
        z = torch.sigmoid(self.convz(hx))
        # input gate
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

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
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


# Encode the Cost Volume and the disparity at current estimation state.
class BasicDepthEncoder(nn.Module):
    def __init__(self,cost_volume_dimension):
        super(BasicDepthEncoder,self).__init__()
        self.cost_volume_dimension = cost_volume_dimension
        self.convc1 = nn.Conv2d(cost_volume_dimension, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-1, 3, padding=1)
        
    
    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_feat = F.relu(self.convf1(disp))
        disp_feat = F.relu(self.convf2(disp_feat))

        cor_disp_feat = torch.cat([cor, disp_feat], dim=1)
        out = F.relu(self.conv(cor_disp_feat))
        return torch.cat([out, disp], dim=1)



class BasicUpdateBlock(nn.Module):
    def __init__(self,hidden_dim=128,input_dim=128,
                 cost_volume_dimension=128):
        super(BasicUpdateBlock,self).__init__()
        self.encoder = BasicDepthEncoder(cost_volume_dimension=cost_volume_dimension)
        self.gru = SepConvGRU(hidden_dim=hidden_dim,input_dim=128+hidden_dim)
        self.disp_head = DisparityHead(hidden_dim,hidden_dim=256)
        
        self.mask = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,64*9,1,padding=0))
        
    def forward(self,net,inp,corr,disp,upsample=True):
        
        motion_features = self.encoder(disp, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_disp = self.disp_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_disp


class CMAUpdateBlock(nn.Module):
    def __init__(self,hidden_dim=128,
                 cost_volume_dimension=128,
                 num_heads=1):
        super(CMAUpdateBlock,self).__init__()
        self.encoder = BasicDepthEncoder(cost_volume_dimension=cost_volume_dimension)
        self.gru = SepConvGRU(hidden_dim=hidden_dim,input_dim=128+hidden_dim+hidden_dim)
        self.disp_head = DisparityHead(hidden_dim,hidden_dim=256)

        
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = Aggregate(dim=128, dim_head=128, heads=num_heads)
        

    def forward(self,net,inp,corr,disp,attention):
        # Disparity Encoding
        disp_corr_feat = self.encoder(disp,corr)
        
        disp_corr_feat_global = self.aggregator(attention,disp_corr_feat)
        
        inp_cat = torch.cat([inp,disp_corr_feat,disp_corr_feat_global],dim=1)

        # Attention Update
        net = self.gru(net,inp_cat)
        
        delta_disp = self.disp_head(net)
        
        # Scale Mask to balance the gradients
        
        mask = .25 * self.mask(net)
        
        return net,mask,delta_disp



if __name__=="__main__":
    hidden_state = torch.randn(1,128,40,80).cuda()
    input_feature = torch.randn(1,128,40,80).cuda()
    corr = torch.randn(1,24,40,80).cuda()
    disp = torch.randn(1,1,40,80).cuda()
    attention = torch.randn(1,1,3200,3200).cuda()
    
    cma = CMAUpdateBlock(hidden_dim=128,cost_volume_dimension=24,num_heads=1).cuda()
    
    net,mask,delta_disp = cma(hidden_state,input_feature,corr,disp,attention)
    