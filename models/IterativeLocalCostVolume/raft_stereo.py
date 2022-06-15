from turtle import left, right
from sklearn.feature_selection import SelectKBest
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
sys.path.append("../..")
from models.IterativeLocalCostVolume.extractor.extractor import MultiBasicEncoder,BasicEncoder,ResidualBlock
from models.IterativeLocalCostVolume.update.update import BasicMultiUpdateBlock_LZH
from models.IterativeLocalCostVolume.corr.corr import CorrBlock1D
from models.IterativeLocalCostVolume.utils.utils import coords_grid,upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='raft-stereo', help="name your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

# Training parameters
parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

# Validation parameters
parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

# Architecure choices
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

# Data augmentation
parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
args = parser.parse_args()


class RAFT_Stereo(nn.Module):
    def __init__(self,
                 hidden_dim =[128,128,128],
                 downsample =2,
                 corr_levels =4,
                 corr_radius =4,
                 n_gru_layers=3,
                 mix_precision=False):
        super(RAFT_Stereo,self).__init__()
        
        self.context_dims = hidden_dim
        self.hidden_dims = hidden_dim
        self.downsample = downsample
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.n_gru_layers = n_gru_layers
        self.mix_precision = mix_precision
        
        self.cnet = MultiBasicEncoder(output_dim=[self.hidden_dims,self.context_dims],
                                      norm_fn="batch",
                                      downsample= self.downsample
                                      )
        self.update_block =  BasicMultiUpdateBlock_LZH(corr_levels=self.corr_levels,
                                                       corr_radius=self.corr_radius,
                                                       n_gru_layers=self.n_gru_layers,
                                                       n_downsample=self.downsample,
                                                       hidden_dims=self.hidden_dims)
        
        #Three GRU Layers
        #[128 --> 128]
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(self.context_dims[i], self.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.n_gru_layers)])

        
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=self.downsample)
    
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape
        
        # Left image coordinate
        coords0 = coords_grid(N, H, W).to(img.device)
        
        # Right Image coordiante
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1
        
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self,left_image,right_image,iters=12,flow_init=None,test_mode=False):
        
        # ImageData Normalization
        left_image = (2 * (left_image / 255.0) - 1.0).contiguous()
        right_image = (2 * (right_image / 255.0) - 1.0).contiguous()
        
        # Run the context network: Different Context for different GRU Layers to provide context
        # Get this at once
        cnet_list = self.cnet(left_image, num_layers=self.n_gru_layers)
        
        # Get 1/4 Level Left and Right Feature
        fmap1, fmap2 = self.fnet([left_image,right_image])
        
        # Hidden State : 1/4,1/8,1/16
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        # Context Input : 1/4,1/8,1/16
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        
        # Context used for GRU's qkr, and each resolution
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
        
        # print(inp_list[0][0].shape) # [B,128,H//4,W//4]
        # print(inp_list[1][0].shape) #[B,128,H//8,W//8]
        # print(inp_list[2][0].shape) #[B,128,H//16,W//16]
        
        corr_block = CorrBlock1D
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        corr_fn = corr_block(fmap1=fmap1,fmap2=fmap2,radius=self.corr_radius,num_levels=self.corr_levels)
        
        # From 1/4 Hidden Feature Size tp build A Zero diff left and right coordinate maps
        coords0, coords1 = self.initialize_flow(net_list[0]) #[B,2,H,W], [B,2,H,W]
        
        if flow_init is not None:
            coords1 = coords1 + flow_init
        
        
        # Gru begin here
        disparity_predictions = []
        for itr in range(iters):
            # Right Image coordinate
            coords1 = coords1.detach()
            # Invoke the call function Here
            corr = corr_fn(coords1) # Index correlation volume
            print(corr.shape)
            flow = coords1 - coords0
            with autocast(enabled=self.mix_precision):
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, 
                                                                  iter32=self.n_gru_layers==3, 
                                                                  iter16=self.n_gru_layers>=2)    
            delta_flow[:,1] = 0.0
            
            coords1 = coords1 + delta_flow
            
            if test_mode and itr<iters-1:
                continue
            
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            disparity_predictions.append(flow_up)
        
        if test_mode:
            return coords1 - coords0, flow_up

        return disparity_predictions



if __name__=="__main__":
    left_image = torch.randn(1,3,320,640).cuda()
    right_image = torch.randn(1,3,320,640).cuda()
    
    raft_stereo = RAFT_Stereo().cuda()
    
    disparity_predictions = raft_stereo(left_image,right_image,12,None,False)
    
    # for d in disparity_predictions:
    #     print(d.shape)
