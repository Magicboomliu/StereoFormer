import torch
from torch import nn, Tensor
from attention import MultiheadAttentionRelative
from torch.utils.checkpoint import checkpoint
from utils.misc import get_clones
from Regression_Head import DisparityOccRegression
from ContextAdjustmentLayer import build_context_adjustment_layer

# Self-Attention.
class  TransformerSelfAttnLayer(nn.Module):
    def __init__(self,hidden_dim:int,nhead:int):
        super().__init__()
        self.self_attn = MultiheadAttentionRelative(hidden_dim,nhead)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
    def forward(self, feat: Tensor,
                pos = None,
                pos_indexes = None):
        """
        :param feat: image feature [W,2HN,C]
        :param pos: pos encoding [2W-1,HN,C]
        :param pos_indexes: indexes to slice pos encoding [W,W]
        :return: updated image feature
        """
        feat2 = self.norm1(feat)

        feat2, attn_weight, _ = self.self_attn(query=feat2, key=feat2, value=feat2, pos_enc=pos,
                                               pos_indexes=pos_indexes)
        # Residual Addition
        feat = feat + feat2

        return feat


# Cross-Attention.
class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self,feat_left,feat_right,
                pos,pos_indexes,last_layer):

        """
        :param feat_left: left image feature, [W,HN,C]
        :param feat_right: right image feature, [W,HN,C]
        :param pos: pos encoding, [2W-1,HN,C]
        :param pos_indexes: indexes to slicer pos encoding [W,W]
        :param last_layer: Boolean indicating if the current layer is the last layer
        :return: update image feature and attention weight
        """
        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)
        
        # Postional Encoding.
        if pos is not None:
            pos_flipped = torch.flip(pos,[0])
        else:
            pos_flipped = pos
        
        # value, attn, attn_feat
        feat_right_2 = self.cross_attn(query = feat_right_2,
                                       key = feat_left_2,
                                       value = feat_left_2,
                                       pos_enc = pos_flipped,
                                       pos_indexes = pos_indexes)[0]
        # update left features
        # use attn mask for last layer
        if last_layer:
            w = feat_left_2.size(0)
            attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)  # generate attn mask
        else:
            attn_mask = None
        # normalize again the updated right features
        feat_right_2 = self.norm2(feat_right)
        feat_left_2, attn_weight, raw_attn = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2,
                                                             attn_mask=attn_mask, pos_enc=pos,
                                                             pos_indexes=pos_indexes)        
        # torch.save(attn_weight, 'cross_attn_' + str(layer_idx) + '.dat')

        feat_left = feat_left + feat_left_2

        # concat features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        return feat, raw_attn
        

    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a mask which is upper triangular
        :param sz: square matrix size
        :return: diagonal binary mask [sz,sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask




class TransformeCostVolume(nn.Module):
    def __init__(self,
                 hidden_dim = 128,
                 n_head = 8,
                 num_attn_layers =6):
        super(TransformeCostVolume,self).__init__()
        
        # Before the cross-attention: first self-attention
        self_attn_layer = TransformerSelfAttnLayer(hidden_dim, n_head)
        # six self-attention layers
        self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)
        
        # Then cross attention
        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim,n_head)
        self.cross_attn_layers = get_clones(cross_attn_layer,num_attn_layers)
        
        # Layer Normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Settings
        self.hidden_dim = hidden_dim
        self.nhead = n_head
        self.num_attn_layers = num_attn_layers
        
        # Disparity and Occlusion Corase Prediction
        self.regression_head = DisparityOccRegression()
        
        # ContextAdjustment Layer
        self.context_adjust_layer = build_context_adjustment_layer('cal')
    
    
    def _alternating_attn(self,feat,pos_enc,pos_indexes,hn):
        '''
        Atternate self and cross attention with gradient checkpoint to save memory
        :param feat: image feature concatenated from left and right, [W,2HN,C]
        :param pos_enc: positional encoding, [W,HN,C]
        :param pos_indexes: indexes to slice positional encoding, [W,HN,C]
        :param hn: size of HN
        :return: attention weight [N,H,W,W]
        '''
        pass
        global layer_idx
        # alternating
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            # Current Layers
            layer_idx = idx
            
            # checkpoint self attn
            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):
                    return module(*inputs)

                return custom_self_attn
            
            # Feature Self-Attention
            feat = checkpoint(create_custom_self_attn(self_attn), feat, pos_enc, pos_indexes)

            # add a flag for last layer of cross attention
            if idx == self.num_attn_layers - 1:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, True)

                    return custom_cross_attn
            else:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, False)

                    return custom_cross_attn

            feat, attn_weight = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:], pos_enc,
                                           pos_indexes)
        layer_idx = 0
        return attn_weight

    def forward(self,feat_left,feat_right, pos_enc: None):
        """
        :param feat_left: feature descriptor of left image, [N,C,H,W]
        :param feat_right: feature descriptor of right image, [N,C,H,W]
        :param pos_enc: relative positional encoding, [N,C,H,2W-1]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image
        """
        # flatten NxCxHxW to WxHNxC
        old_feature = feat_left
        bs,c,hn,w = feat_left.shape
        
        feat_left = feat_left.permute(1,3,2,0).flatten(2).permute(1,2,0) # CxWxHxN -> CxWxHN -> WxHNxC
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)

        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes = None
        
       
        # concatenate left and right features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        # compute attention
        attn_weight = self._alternating_attn(feat, pos_enc, pos_indexes, hn)
        attn_weight = attn_weight.view(hn, bs, w, w).permute(1, 0, 2, 3)  # NxHxWxW, dim=2 left image, dim=3 right image

        disp_low_res,occlusion_low_res=self.regression_head(attn_weight,old_feature)
        
        # Maybe can Try Upsampleing
        final_disp, final_occ = self.context_adjust_layer(disp_low_res.unsqueeze(1),occlusion_low_res.unsqueeze(1),
                                                          img_left)
        return final_disp,final_occ



from positional_encoding import PositionEncodingSine1DRelative
if __name__=="__main__":
    img_left = torch.randn(1,3,8,16).cuda()
    feature_left = torch.randn(1,128,8,16).cuda()
    feature_right = torch.randn(1,128,8,16).cuda()
    
    transformer_cost_volume = TransformeCostVolume(
                hidden_dim = 128,
                 n_head = 8,
                 num_attn_layers =6).cuda()
    
    
    positional_encoding_op = PositionEncodingSine1DRelative(num_pos_feats=128,temperature=10000,
                                                         normalize=False,scale=None).cuda()
    
    # 2W-1,C
    pos_enc = positional_encoding_op(feature_left)
    
    # with positional encoding.
    final_disp,final_occ = transformer_cost_volume(feature_left,feature_right,pos_enc)
    
    print(final_disp.shape)
    