
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys
sys.path.append("../../..")
from Transformers.positional_encoding.absolute_sincos_embedding import positionalencoding2d,positionalencoding1d

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
#FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # FLC output
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Transformer Blocks
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads[_], dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self,
                 image_size=(224,224),
                 patch_size=16,
                 embedd_dim = 512,
                 mlp_dim = 256,
                 depths = 3,
                 heads =[2,4,8],
                 input_channels=128,
                 dim_head = 64,
                 dropout_rate=0.,
                 emb_dropout =0.,
                 skiped_patch_embedding=False,
                 ape='learn'):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embedd_dim = embedd_dim
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.dropout_rate = dropout_rate
        self.depths  = depths
        self.heads = heads
        self.dim_head = dim_head
        self.skiped_patched_embedding = skiped_patch_embedding
        self.ape = ape

        H,W = self.image_size
        patch_H, patch_W = self.patch_size
        
        assert H % patch_H == 0 and W % patch_W == 0, 'Image dimensions must be divisible by the patch size.'

        self.dropout = nn.Dropout(self.emb_dropout)
        
        self.transformer = Transformer(self.embedd_dim, 
                                       self.depths, 
                                       self.heads, 
                                       self.dim_head, 
                                       self.mlp_dim, 
                                       self.dropout_rate)        
    def forward(self,img):
        B,C,D,H,W = img.shape
        
        # Transformer Aggregation.
        x = img.permute(0,3,4,1,2).view(-1,C,D)
        
        
        x= self.dropout(x) #[B,N,C]
        
        x = self.transformer(x)
        x = x.view(B,H,W,C,D).permute(0,3,4,1,2)
        
        return x



# Apply Cost Channel Transformer 
class CostChannelTransformer(nn.Module):
    def __init__(self,
              feature_channels = 128,
              image_size=(40,80),patch_size=(1,1),heads=(2,4,4),dim_head=24,depths=3,
              embedd_dim=24,mlp_dim=24,input_channels=24,dropout_rate=0.,emb_dropout=0.,
              ape='sincos1d'):
        super(CostChannelTransformer,self).__init__()
        self.image_size = image_size
        self.feature_channels = feature_channels
        self.patch_size = patch_size
        self.heads = heads
        self.dim_head = dim_head
        self.depths = depths
        self.embedd_dim = embedd_dim
        self.mlp_dim = mlp_dim
        self.input_channels = input_channels
        self.dropout = dropout_rate
        self.emb_dropout = emb_dropout
        
        self.first_conv = nn.Conv3d(self.feature_channels,self.feature_channels//2,kernel_size=3,stride=1,padding=1,
                                    bias=True)
        self.cross_channels_aggregation = ViT(image_size=image_size,patch_size=patch_size,heads=heads,
                                              dim_head=dim_head,
                                              depths=depths,
                                            embedd_dim=embedd_dim,mlp_dim=mlp_dim,
                                            input_channels=input_channels,dropout_rate=dropout_rate,
                                            emb_dropout=emb_dropout,ape='sincos1d')
        self.final_conv = nn.Conv3d(self.feature_channels//2,1,kernel_size=3,stride=1,padding=1,
                                    bias=True)
        

    def forward(self,x):
        x = F.relu(self.first_conv(x),True)
        
        x = self.cross_channels_aggregation(x)
        
        out = self.final_conv(x)
        
        out = out.squeeze(1)
        
        return out

if __name__=="__main__":
    
    image = torch.randn(1,256,24,40,80).cuda()
    
    # vit = ViT(image_size=(40,80),patch_size=(1,1),heads=(2,4,4),dim_head=24,depths=3,
    #           embedd_dim=24,mlp_dim=24,input_channels=24,dropout_rate=0.,emb_dropout=0.,
    #           ape='sincos1d').cuda()
    
    
    cross_channels_transformer = CostChannelTransformer(feature_channels=256,
                                                        image_size=(40,80),
                                                        patch_size=(1,1),
                                                        heads=(2,4,4),
                                                        dim_head=192//8,
                                                        depths=3,
                                                        embedd_dim=24,
                                                        mlp_dim=24,
                                                        input_channels=24,
                                                        dropout_rate=0.,
                                                        emb_dropout=0.,
                                                        ape='sincos1d').cuda()
    
    output = cross_channels_transformer(image)
    
    print(output.shape)