from __future__ import print_function
import sys
sys.path.append("../")
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.AverageMeter import AverageMeter
from utils.common import logger, check_path, write_pfm
from dataloader.preprocess import scale_disp
from dataloader.SceneflowLoader import StereoDataset
from dataloader import transforms

# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



