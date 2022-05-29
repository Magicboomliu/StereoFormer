from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from models.RAFT_Stereo.update import BasicMultiUpdateBlock
from models.RAFT_Stereo.extractor import BasicEncoder,MultiBasicEncoder,ResidualBlock
from models.RAFT_Stereo.corr import CorrBlock1D,PytorchAlternateCorrBlock1D,CorrBlockFast1D,AlternateCorrBlock
from models.RAFT_Stereo.utils import coords_grid,upflow8

from losses.squence_loss import sequence_loss

def print_tensor_shape(inputs):
    if isinstance(inputs,list) or isinstance(inputs,tuple):
        for value in inputs:
            print(value.shape)
    else:
        print(inputs.shape)
        
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

class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims #[128,128,128]

        # n downsample is 3
        
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
                
            else:
                
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                
                # 1/4 Left feature and 1/4 Right feature
                fmap1, fmap2 = self.fnet([image1, image2])

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
    


class Args(object):
    def __init__(self,name='raft-stereo',corr_implementation='reg',corr_levels=4,n_downsample=2,
                 corr_radius=4,
                 mixed_precision=True,
                 slow_fast_gru=False,
                 shared_backbone = False,
                 n_gru_layers=3,hidden_dims=[128]*3) -> None:
        self.name = name
        self.corr_radius = corr_radius
        self.corr_implementation = corr_implementation
        self.corr_levels = corr_levels
        self.n_downsample = n_downsample
        self.n_gru_layers = n_gru_layers
        self.hidden_dims = hidden_dims
        self.shared_backbone = shared_backbone
        self.mixed_precision = mixed_precision
        self.slow_fast_gru = slow_fast_gru
        
if __name__=="__main__":
    # import argparse
    import numpy as np
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    # parser.add_argument('--restore_ckpt', help="restore checkpoint")
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # # Training parameters
    # parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    # parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    # parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    # parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    # parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    # parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    # parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # # Validation parameters
    # parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # # Architecure choices
    # parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    # parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    # parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    # parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    # parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    # parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    # parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    # parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # # Data augmentation
    # parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    # parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    # parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    # parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    # parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    # args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    left_image = torch.randn(1,3,320,640).cuda()
    right_image = torch.randn(1,3,320,640).cuda()
    
    args = Args()
    raft_stereo = RAFTStereo(args=args).cuda()
    
    
    disp = raft_stereo(left_image,right_image,test_mode=False)
    
    gt_disp = torch.abs(torch.randn(1,1,320,640)).cuda()
    
    print(disp[0].shape)
    print(disp[1].shape)
    
    # loss = sequence_loss(disp,gt_disp)
    
    # print(loss.item())
