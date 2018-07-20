import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=1024, c_dim=5, image_size=256, op_channels=3):
        super(Generator, self).__init__()

        self.spatial = nn.Linear(c_dim,conv_dim,bias=False) #Use Conv2d here? Wont it be the same?
        
        layers = []
        curr_dim = conv_dim
        for i in range(int(math.log2(image_size)-2)):
            layers.append(nn.ConvTranspose2d(curr_dim,curr_dim//2,kernel_size=4,stride=2,padding=1,bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2,affine=True,track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim=curr_dim//2

        layers.append(nn.Conv2d(curr_dim,op_channels,kernel_size=3,stride=1,padding=1,bias=False))
        layers.append(nn.Tanh())
        
        self.upsample = nn.Sequential(*layers)

    def forward(self, c):
        # Replicate spatially and concatenate domain information.
        c_h = self.spatial(c)
        c_h = c_h.view(c_h.size(0), c_h.size(1), 1, 1)
        c_h = c_h.repeat(1, 1, 4, 4)
        x = self.upsample(c_h)
        return x
        
class Discriminator(nn.Module):
    """Intermediate layer to obtain latent space network with PatchGAN."""
    def __init__(self, image_size=256, conv_dim=64, inp_channels=3,c_dim=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inp_channels, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, int(math.log2(image_size)-1)):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        
        self.real_conv = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.cls_conv  = nn.Conv2d(curr_dim, c_dim, kernel_size=2, stride=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.real_conv(h)
        out_cls = self.cls_conv(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

# class Discriminator(nn.Module):
#     """ Outputs attributes and real/fake"""
#     def __init__(self, image_size=256, conv_dim=64, c_dim=5):
#         super(Discriminator,self).__init__()
        
#         curr_dim=conv_dim
#         for _ in range(1,int(math.log2(image_size)-1)):
#             curr_dim = curr_dim*2
        
#         kernel_size=image_size//np.power(2,7)
        
#         self.real_conv = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.cls_conv  = nn.Conv2d(curr_dim, c_dim, kernel_size=2, stride=1, bias=False)

#     def forward(self,x):
#         out_src = self.real_conv(x)
#         out_cls = self.cls_conv(x)
#         return out_src,out_cls.view(out_cls.size(0),out_cls.size(1))

# class Q(nn.Module):
#     """ Outputs logits and stats for G(x,c)"""
#     def __init__(self,image_size=256,conv_dim=64,con_dim=2):
#         super(Q,self).__init__()

#         curr_dim=conv_dim
#         for _ in range(1,int(math.log2(image_size)-1)):
#             curr_dim=curr_dim*2

#         # Remove this and see!?
#         self.conv=nn.Sequential(nn.Conv2d(curr_dim, 128,  kernel_size=1,bias=False),
#                                 nn.LeakyReLU(0.01,inplace=True),
#                                 nn.Conv2d(128,    64, kernel_size=1,bias=False),
#                                 nn.LeakyReLU(0.01,inplace=True))

#         self.conv_mu =nn.Conv2d(curr_dim,con_dim,kernel_size=2,stride=1,padding=0)
#         self.conv_var=nn.Conv2d(curr_dim,con_dim,kernel_size=2,stride=1,padding=0)

#     def forward(self,h):
#         # out=self.conv(h)
        
#         mu_out=self.conv_mu(h).squeeze()
#         var_out=self.conv_var(h).squeeze().exp()
        
#         return mu_out,var_out

