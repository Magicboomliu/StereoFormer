import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# 6 
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])