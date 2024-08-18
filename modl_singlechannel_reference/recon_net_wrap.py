import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from fastmri.models import Unet
from recon_net import ReconNet

def initialize_conv1d_as_delta(conv1,channels):
    # Ensure that this operation is not tracked by autograd
    torch.nn.init.zeros_(conv1.weight)
    #torch.nn.init.zeros_(conv1.bias)
    # set identity kernel
    #print (conv1.weight.data.shape)
    for i in range(channels):
        conv1.weight.data[i, i, 1] = torch.ones((1,1))



def initialize_conv2d_as_delta(conv2,channels):
    # Get the weight tensor of the convolutional layer
    with torch.no_grad():
        conv2.weight[:, :, :, :] = 0.0
        conv2.bias[:] = 0.0
        for i in range(channels):
            conv2.weight[i, i, 2, 2] = 1.0 # our equivalent delta-dirac

def initialize_conv2d_as_delta_noBias(conv2,channels):
    # Get the weight tensor of the convolutional layer
    with torch.no_grad():
        conv2.weight[:, :, :, :] = 0.0
        for i in range(channels):
            conv2.weight[i, i, 3, 3] = 1.0 # our equivalent delta-dirac


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionBlock, self).__init__()
        
        # Convolutions with kernel size 1 to initialize close to delta function
        self.conv1_acq1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1_acq2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)


        # Initializing convolutions close to identity (delta-like)
        initialize_conv1d_as_delta(self.conv1_acq1,in_channels)
        initialize_conv1d_as_delta(self.conv1_acq2,in_channels)

        # Activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        # mRCAB and similarity weightings (these will be implemented as small sub-modules)
        self.mrcab = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)  # Placeholder for actual mRCAB
        initialize_conv1d_as_delta(self.mrcab,in_channels)

    def forward(self, acq1, acq2):
        # Pass through first set of convolutions and activations
        acq1 = self.lrelu(self.conv1_acq1(acq1))
        acq2 = self.lrelu(self.conv1_acq2(acq2))

        # Compute similarity weightings
        similarity = torch.sigmoid(acq1 - acq2)
        
        # Element-wise operations (fusion process)
        fused_features = acq1 * similarity + acq2 * (1 - similarity)

        # mRCAB processing
        fused_features = self.mrcab(fused_features)
        
        return fused_features
class BilinearPoolingFusionNet(nn.Module):
    def __init__(self):
        super(BilinearPoolingFusionNet, self).__init__()
        # Apply a convolution to project back to the original feature space
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5,padding=2)
        self.param1 = nn.Parameter(torch.normal(1, 0.01, size=(198,)))
        self.param2 = nn.Parameter(torch.normal(0, 0.01, size=(198,)))  
        self.epsilon = 1e-6
        initialize_conv2d_as_delta(self.conv1,1)
        initialize_conv2d_as_delta(self.conv2,1)

    def forward(self, x1, x2):
        #x1_conv = self.conv1(x1.transpose(1, 2)).transpose(1, 2)
        #print(f'x1 before: {x1.shape}')
        x2 = self.conv2(x2.unsqueeze(1)).squeeze(1)
        x1 = self.conv1(x1.unsqueeze(1)).squeeze(1)
        #print(f'x1 after: {x1.shape}')
        #print("Current value of param2 during forward:", self.param2)
        # fuse
        batch_size, num_channels, height = x1.shape
        features_flat = x1.reshape(batch_size, num_channels, -1)
        features_ref_flat = x2.reshape(batch_size, num_channels, -1)       
        
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
        features_comb = features_comb.reshape(features_flat.shape[0], 198, 1024)        
        
        return features_comb

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        initialize_conv2d_as_delta_noBias(self.conv1,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ViTfuser(nn.Module):
    def __init__(self, net, epsilon=1e-6):
        super().__init__()
        self.device = 'cuda:0'

        # ViT layer
        self.recon_net = ReconNet(net).to(self.device)#.requires_grad_(False)
        self.recon_net_ref = ReconNet(net).to(self.device)
        # Load weights
        cp = torch.load('./lsdir-2x+hq50k_vit_epoch_60.pt', map_location=self.device)
        self.recon_net.load_state_dict(cp['model_state_dict'])
        self.recon_net_ref.load_state_dict(cp['model_state_dict'])

        # Fusion layers
        self.epsilon = epsilon
        # High Resolution
        #self.param1 = nn.Parameter(torch.normal(1, 0.01, size=(416,)))
        #self.param2 = nn.Parameter(torch.normal(0, 0.01, size=(416,)))
        # Low Resolution
        self.param1 = nn.Parameter(torch.normal(1, 0.01, size=(198,)))
        self.param2 = nn.Parameter(torch.normal(0, 0.01, size=(198,)))
        #self.fuser = FeatureFusionBlock(in_channels=198,out_channels=198)
        """
        self.conv1_acq1 = nn.Conv1d(198, 198, kernel_size=3,padding=1, bias=False)
        self.conv1_acq2 = nn.Conv1d(198, 198, kernel_size=3,padding=1, bias=False)


        # Initializing convolutions close to identity (delta-like)
        initialize_conv1d_as_delta(self.conv1_acq1,198)
        initialize_conv1d_as_delta(self.conv1_acq2,198)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        # mRCAB and similarity weightings (these will be implemented as small sub-modules)
        self.mrcab = nn.Conv1d(198, 198, kernel_size=3,padding=1, bias=False)  # Placeholder for actual mRCAB
        initialize_conv1d_as_delta(self.mrcab,198)
        """

    def printer(self, x):
        print("Current value of param1 during forward:", self.param1)
        return

    def forward(self, img,ref): #,ref
        # Norm
        #print("Current value of param1 during forward:", self.param1)
        #print("Current value of param2 during forward:", self.param2)
        in_pad, wpad, hpad = self.recon_net.pad(img)
        ref_pad, wpad, hpad = self.recon_net.pad(ref)
        input_norm,mean,std = self.recon_net.norm(in_pad.float())
        ref_norm,mean_ref,std_ref = self.recon_net.norm(ref_pad.float())
        #print("Weights of the Conv1 layer:")
        #print(self.conv1.weight)        
        # Feature extract
        features = self.recon_net.net.forward_features(input_norm)#.permute(0,2,1)
        #features = self.conv1_acq1(features)

        
        features_ref = self.recon_net_ref.net.forward_features(ref_norm)#.permute(0,2,1)
        #features_ref = self.conv1_acq1(features_ref)
        """
        acq1 = self.lrelu(self.conv1_acq1(features))
        acq2 = self.lrelu(self.conv1_acq2(features_ref))

        # Compute similarity weightings
        similarity = torch.sigmoid(acq1 - acq2)
        
        # Element-wise operations (fusion process)
        features_ref = features * similarity + features_ref * (1 - similarity)

        # mRCAB processing
        features_ref = self.mrcab(features_ref)
        """



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
        
        # Reshape back to [1, 416, 1024] - High Resolution
        #features_comb = features_comb.reshape(features_flat.shape[0], 416, 1024)
        # Low Resolution
        features_comb = features_comb.reshape(features_flat.shape[0], 198, 1024)
        
        #features_comb = self.fuser(features,features_ref)
        #print(f'features_comb: {features_comb.shape}')
        
        # Recon Head
        head_out = self.recon_net.net.head(features_comb)
        
        # High Resolution
        #head_out_img = self.recon_net.net.seq2img(head_out, (260, 160))
        # Low Resolution 
        head_out_img = self.recon_net.net.seq2img(head_out, (180, 110))


        # un-norm
        merged = self.recon_net.unnorm(head_out_img, mean, std) 

        # un-pad 
        im_out = self.recon_net.unpad(merged,wpad,hpad)
        
        return im_out