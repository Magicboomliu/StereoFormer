import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from models.GMA_Stereo.Models.GMA_Stereo.CostVolume.build_cost_volume import CostVolume


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


class GlobalAggregationSM(nn.Module):
    def __init__(self,
                 max_disp=192):
        super().__init__()
        
        hdim = 128
        cdim = 128
        self.max_disp = max_disp
        self.hidden_state = 128
        self.context_state = 128
            
    
    def forward(self,x):
        pass