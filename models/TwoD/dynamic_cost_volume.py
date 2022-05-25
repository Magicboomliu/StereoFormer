import torch
import torch.nn as nn
import torch.nn.functional as F

# RAFT Stereo Version

class DynamicCostVolumeRefinement(nn.Module):
    def __init__(self):
        super(DynamicCostVolumeRefinement,self).__init__()
    
    def forward(self,left_feature,right_feature,disp,left_image,right_image):
        '''Local Cost Volume Refinement / RAFT/disp'''
        
        
        pass




if __name__=="__main__":
    pass