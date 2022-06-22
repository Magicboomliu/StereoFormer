import torch
import torch.nn as nn
import torch.nn.functional as F

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