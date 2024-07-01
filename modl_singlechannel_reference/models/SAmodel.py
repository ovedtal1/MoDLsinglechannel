import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.complex_utils as cplx
from utils.transforms import SenseModel,SenseModel_single
from utils.flare_utils import ConjGrad

class Operator(torch.nn.Module):
    def __init__(self, A):
        super(Operator, self).__init__()
        self.operator = A

    def forward(self, x):
        return self.operator(x)

    def adjoint(self, x):
        return self.operator(x, adjoint=True)

    def normal(self, x):
        out = self.adjoint(self.forward(x))
        return out
    

class LeakyReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LeakyReLUConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        return self.lrelu(self.conv(x))

class mRCAB(nn.Module):
    def __init__(self, in_channels):
        super(mRCAB, self).__init__()
        self.conv1 = LeakyReLUConv(in_channels, in_channels)
        self.conv2 = LeakyReLUConv(in_channels, in_channels)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        ca = self.ca(res)
        return x + res * ca

class ResidualGroup(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(*[mRCAB(in_channels) for _ in range(num_blocks)])
        
    def forward(self, x):
        return x + self.blocks(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SubPixelConv(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=1):
        super(SubPixelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

class MyNetwork(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_residual_groups=3, num_blocks=5):
        super(MyNetwork, self).__init__()
        self.initial_conv = LeakyReLUConv(in_channels, 64)
        self.residual_groups1 = ResidualGroup(64, num_blocks)
        self.residual_groups2 = ResidualGroup(64, num_blocks)
        self.residual_groups3 = ResidualGroup(64, num_blocks)
        self.mrcabs = mRCAB(64)
        self.spatial_attention = SpatialAttention()
        self.sub_pixel_conv = SubPixelConv(64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, kspace, reference_image,init_image=None, mask=None):
        #if mask is None:
        #    mask = cplx.get_mask(kspace)
        #kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())
        """
        # Declare signal model
        A = SenseModel_single(weights=mask)
        Sense = Operator(A)
        # Compute zero-filled image reconstruction
        x = Sense.adjoint(kspace)
        x = x.permute(0, 3, 1, 2)
        """
        x = kspace
        #print(f'After adjoint: {x.shape}')
        #trilnear = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.initial_conv(x)
        #print(f'After initial_conv: {x.shape}')

        res1 = self.residual_groups1(x)
        #print(f'After res1: {res1.shape}')
        res2 = self.residual_groups2(res1)
        #print(f'After res2: {res2.shape}')
        res3 = self.residual_groups3(res2)
        #print(f'After res3: {res3.shape}')
        x = self.mrcabs(res3)
        x = self.spatial_attention(x) * x
        #print(f'After spatial_attention: {x.shape}')
        x = self.sub_pixel_conv(x)
        #print(f'After sub_pixel conv: {x.shape}')
        x = self.final_conv(x)
        #print(f'After final conv: {x.shape}')
        #x = x.permute(0,2,3, 1)
        return x 
