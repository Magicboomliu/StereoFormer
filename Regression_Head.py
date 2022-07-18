import torch
import torch.nn as nn
import torch.nn.functional as F



class DisparityOccRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost
        
        
    
    def _compute_unscaled_pos_shift(self, w: int, device: torch.device):
        """
        Compute relative difference between each pixel location from left image to right image, to be used to calculate
        disparity
        :param w: image width
        :param device: torch device
        :return: relative pos shifts
        """
        pos_r = torch.linspace(0, w - 1, w)[None, None, None, :].to(device)  # 1 x 1 x 1 x W_right
        pos_l = torch.linspace(0, w - 1, w)[None, None, :, None].to(device)  # 1 x 1 x W_left x1
        pos = pos_l - pos_r
        pos[pos < 0] = 0
        return pos
    
    def _compute_low_res_disp(self, pos_shift, attn_weight, occ_mask):
        """
        Compute low res disparity using the attention weight by finding the most attended pixel and regress within the 3px window
        :param pos_shift: relative pos shift (computed from _compute_unscaled_pos_shift), [1,1,W,W]
        :param attn_weight: attention (computed from _optimal_transport), [N,H,W,W]
        :param occ_mask: ground truth occlusion mask, [N,H,W]
        :return: low res disparity, [N,H,W] and attended similarity sum, [N,H,W]
        """

        # find high response area
        high_response = torch.argmax(attn_weight, dim=-1)  # NxHxW

        # build 3 px local window
        response_range = torch.stack([high_response - 1, high_response, high_response + 1], dim=-1)  # NxHxWx3

        # attention with re-weighting
        attn_weight_pad = F.pad(attn_weight, [1, 1], value=0.0)  # N x Hx W_left x (W_right+2)
        attn_weight_rw = torch.gather(attn_weight_pad, -1, response_range + 1)
        
        # offset range by 1, N x H x W_left x 3

        # compute sum of attention
        norm = attn_weight_rw.sum(-1, keepdim=True)
        if occ_mask is None:
            norm[norm < 0.1] = 1.0
        else:
            norm[occ_mask, :] = 1.0  # set occluded region norm to be 1.0 to avoid division by 0

        # re-normalize to 1
        attn_weight_rw = attn_weight_rw / norm  # re-sum to 1
        pos_pad = F.pad(pos_shift, [1, 1]).expand_as(attn_weight_pad)
        pos_rw = torch.gather(pos_pad, -1, response_range + 1)

        # compute low res disparity
        disp_pred_low_res = (attn_weight_rw * pos_rw)  # NxHxW
        
        return disp_pred_low_res.sum(-1), norm

    def _compute_low_res_occ(self, matched_attn):
        """
        Compute low res occlusion by using inverse of the matched values
        :param matched_attn: updated attention weight without dustbins, [N,H,W,W]
        :return: low res occlusion map, [N,H,W]
        """
        occ_pred = 1.0 - matched_attn
        return occ_pred.squeeze(-1)
    
    def _softmax(self, attn):
        """
        Alternative to optimal transport
        :param attn: raw attention weight, [N,H,W,W]
        :return: updated attention weight, [N,H,W+1,W+1]
        """
        bs, h, w, _ = attn.shape

        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)

        attn_softmax = F.softmax(similarity_matrix, dim=-1)

        return attn_softmax
    
    
    def forward(self,attn_weight,x):
        '''
        :param attn_weight: raw attention weight, [N,H,W,W]
        :param x: input data
        :return: dictionary of predicted values
        '''
        bs, _, h, w = x.size()
        
        scale = 0
        
        attn_ot = self._softmax(attn_weight)

        # Get initial Disparity
        pos_shift = self._compute_unscaled_pos_shift(attn_weight.shape[2], attn_weight.device)  # NxHxW_leftxW_right        
        disp_pred_low_res, matched_attn = self._compute_low_res_disp(pos_shift, attn_ot[..., :-1, :-1], None)
        
        # Get Initail Occlusion Mask
        occ_pred_low_res = self._compute_low_res_occ(matched_attn)
        
        return disp_pred_low_res,occ_pred_low_res
        
        
        