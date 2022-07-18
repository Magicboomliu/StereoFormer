from matplotlib.pyplot import sca
import torch
import torch.nn as nn
import math

class PositionEncodingSine1DRelative(nn.Module):
    def __init__(self,num_pos_feats=64,
                 temperature=10000,
                 normalize = False,
                 scale=None):
        super().__init__()
        
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    @torch.no_grad()
    def forward(self, left_feat):
        """
        :param inputs: NestedTensor
        :return: pos encoding [N,C,H,2W-1]
        """
        x = left_feat
        # update h and w if downsampling
        bs, _, h, w = x.size()
        
        
        # Index
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)
        
        if self.normalize:
            x_embed = x_embed * self.scale
        
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # 2W-1xC
        # interleave cos and sin instead of concatenate
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC

        return pos