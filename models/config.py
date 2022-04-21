from ast import parse
import imp
import parser
import argparse
from models.swin_former import SwinTransformer

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--type",type=str, help='Select a Swin-Former')
parser.add_argument("--in_channels",type=int,help='Input image channels')
parser.add_argument("--patch_size",type=int,help='Image patch size')
parser.add_argument("--window_size",type=int,help='Window attention size')
parser.add_argument("--embedding_dim",type=int,help='Embedding dimension')
parser.add_argument("--depth",type=tuple,default=(2,2,6,2),help='Depth blocks')
parser.add_argument("--num_heads",type=tuple,default=(3, 6, 12, 24),help='Multi-head attention')
parser.add_argument("--num_classes",type=int,default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
# 是否冻结权重
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

parser.add_argument('--data_path', type=str,
                        default="/data/flower_photos")
# operations
opt = parser.parse_args()

