import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from fastmri.models import Unet
from recon_net import ReconNet


class LeakyReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LeakyReLUConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        return self.lrelu(self.conv(x))

class mRCAB(nn.Module):
    def __init__(self, in_channels):
        super(mRCAB, self).__init__()
        self.conv1 = LeakyReLUConv(in_channels, in_channels)
        self.conv2 = LeakyReLUConv(in_channels, in_channels)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels * 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels * 16, in_channels, 1, padding=0),
            nn.Sigmoid()
        )
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        ca = self.ca(res)
        return self.gn(x + res * ca)

class ResidualGroup(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(ResidualGroup, self).__init__()
        self.blocks = nn.Sequential(*[mRCAB(in_channels) for _ in range(num_blocks)])
        
    def forward(self, x):
        return x + self.blocks(x)
    

class ViTfuser(nn.Module):
    def __init__(self, net, epsilon=1e-6):
        super().__init__()
        self.device = 'cuda:0'

        # ViT layer
        self.recon_net = ReconNet(net).to(self.device)#.requires_grad_(False)
        # Load weights
        cp = torch.load('./lsdir-2x+hq50k_vit_epoch_60.pt', map_location=self.device)
        self.recon_net.load_state_dict(cp['model_state_dict'])

        # Fusion layers
        self.epsilon = epsilon
        self.param1 = nn.Parameter(torch.normal(1, 0.01, size=(416,)))
        self.param2 = nn.Parameter(torch.normal(0, 0.01, size=(416,)))
        """
        self.rg_image  = ResidualGroup(416, 5)
        self.rg_ref  = ResidualGroup(416, 5)
        self.rg_fusion  = ResidualGroup(416, 5)
        """
    def printer(self, x):
        print("Current value of param1 during forward:", self.param1)
        return

    def forward(self, img,ref): #,ref
        # Norm
        #print("Current value of param1 during forward:", self.param2)
        in_pad, wpad, hpad = self.recon_net.pad(img)
        ref_pad, wpad, hpad = self.recon_net.pad(ref)
        input_norm,mean,std = self.recon_net.norm(in_pad.float())
        ref_norm,mean_ref,std_ref = self.recon_net.norm(ref_pad.float())
        
        # Feature extract
        features = self.recon_net.net.forward_features(input_norm)
        #features = self.rg_image(features)

        features_ref = self.recon_net.net.forward_features(ref_norm)
        #features_ref = self.rg_ref(features_ref)
        #features = (features + features_ref)/2 
        #print(f'fetures shape: {features.shape}')
        # Fusion
        
        batch_size, num_channels, height = features.shape
        features_flat = features.reshape(batch_size, num_channels, -1)
        features_ref_flat = features_ref.reshape(batch_size, num_channels, -1)       
        # Reshape params to match the dimensions
        param1_expanded = self.param1.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        param2_expanded = self.param2.reshape(1, -1, 1)  # Shape: [1, 416, 1]
        # Expand params to match the flattened tensor dimensions
        param1_expanded = param1_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        param2_expanded = param2_expanded.expand(batch_size, -1, height)  # Shape: [batch_size, 416, height*width]
        # Calculate weighted sum
        weighted_sum = (param1_expanded * features_flat + param2_expanded * features_ref_flat)
        
        # Calculate normalization factor
        normalization_factor = param1_expanded + param2_expanded + self.epsilon
        
        # Normalize
        features_comb = weighted_sum / normalization_factor
        #features_comb = self.rg_fusion(features_comb.squeeze(0)).unsqueeze(0)     
     
        # Reshape back to [1, 416, 1024]
        features_comb = features_comb.reshape(features_flat.shape[0], 416, 1024)
        
        """
        # Recon Head
        head_out = self.recon_net.net.head(features)#0.5*features+0.5*features_ref 
        head_out_img = self.recon_net.net.seq2img(head_out, (260, 160))

        # un-norm
        merged = self.recon_net.unnorm(head_out_img, mean, std) 

        # un-pad 
        im_out = self.recon_net.unpad(merged,wpad,hpad)
        """
        return features_comb