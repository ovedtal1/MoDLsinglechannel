import os, sys
import torch
from torch import nn
import sigpy.plot as pl
import utils.complex_utils as cplx
from utils.transforms import SenseModel,SenseModel_single
from utils.layers3D import ResNet
from unet.unet_model import UNet
from unet.unet_model import UNetPrior
from utils.flare_utils import ConjGrad
import matplotlib
from models.SAmodel import MyNetwork
#from image_fusion import fuse
from ImageFusionBlock import ImageFusionBlock
import numpy as np
from IFCNN import myIFCNN
## New Try
from ImageFusion_Dualbranch_Fusion.densefuse_net import DenseFuseNet
from PIL import Image
import torchvision.transforms as transforms
import time
import os
from recon_net_wrap import ViTfuser
from vision_transformer import VisionTransformer

#### End new try
# matplotlib.use('TkAgg')

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

class UnrolledViT(nn.Module):
    """
    PyTorch implementation of Unrolled Compressed Sensing.

    Implementation is based on:
        CM Sandino, et al. "DL-ESPIRiT: Accelerating 2D cardiac cine 
        beyond compressed sensing" arXiv:1911.05845 [eess.SP]
    """

    def __init__(self, params):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()

        # Extract network parameters
        self.num_grad_steps = params.num_grad_steps 
        self.num_cg_steps = params.num_cg_steps
        self.share_weights = params.share_weights
        self.modl_lamda = params.modl_lamda
        self.reference_mode = params.reference_mode
        self.reference_lambda = params.reference_lambda
        self.device = 'cuda:0'

        net = VisionTransformer(
        avrg_img_size=320,
        patch_size = (10,10),
        in_chans=1,
        embed_dim=64,
        depth=10,
        num_heads=16
        )
        if self.share_weights:
            print("shared weights")
            self.resnets = nn.ModuleList([MyNetwork(2,2)] * self.num_grad_steps)
            self.similaritynets = nn.ModuleList([ViTfuser(net)] * self.num_grad_steps)
        else:
            print("No shared weights")
            self.resnets = nn.ModuleList([MyNetwork(2,2) for i in range(self.num_grad_steps)])
            self.similaritynets = nn.ModuleList([ViTfuser(net) for i in range(self.num_grad_steps)])

        # intialize for training
        #checkpoint_file = "./lsdir-2x+hq50k_vit_epoch_60.pt"
        #checkpoint = torch.load(checkpoint_file,map_location=self.device)
        #for net in self.similaritynets:
        #    net.recon_net.load_state_dict(checkpoint['model'])
    def freezer():
        for net in self.similaritynets:
            net.recon_net.net.forward_features.requires_grad_(False)
            net.recon_net.net.head.requires_grad_(False)
        

    def forward(self, kspace, reference_image,init_image=None, mask=None):
        """
        Args:
            kspace (torch.Tensor): Input tensor of shape [batch_size, height, width, time, num_coils, 2]
            maps (torch.Tensor): Input tensor of shape   [batch_size, height, width,    1, num_coils, num_emaps, 2]
            mask (torch.Tensor): Input tensor of shape   [batch_size, height, width, time, 1, 1]

        Returns:
            (torch.Tensor): Output tensor of shape       [batch_size, height, width, time, num_emaps, 2]
        """
        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())

        # Declare signal model
        A = SenseModel_single(weights=mask)
        Sense = Operator(A)
        # Compute zero-filled image reconstruction
        zf_image = Sense.adjoint(kspace)

        
        # Reference dealing
        reference_image = reference_image.permute(0,3,1,2) 
        #print(f'ref size: {reference_image.shape}')
        real_part_ref = reference_image[:,0,:,:].unsqueeze(1)
        imag_part_ref = reference_image[:,1,:,:].unsqueeze(1)
        mag_ref = torch.sqrt(real_part_ref**2 + imag_part_ref**2)


        image = zf_image 

        #reference_image = reference_image.permute(0,3,1,2) 
        # Begin unrolled proximal gradient descent
        iter = 1
        for resnet, similaritynet in zip(self.resnets, self.similaritynets):
            #print(image.shape)
            # Combine the output of ResNet with the reference image
            if (self.reference_mode == 1 ): # and iter != self.num_grad_steps
                #print(f'image shape: {image.shape}')
                #print(f'reference_image shape: {reference_image.shape}')
                image = image.permute(0,3,1,2)
                #print(f'Image size: {image.shape}')
                real_part = image[:,0,:,:].unsqueeze(1)
                imag_part = image[:,1,:,:].unsqueeze(1)
                #print(f'imag_part_ref shape: {imag_part_ref.shape}')

                phase = torch.atan2(real_part,imag_part)
                mag_image = torch.sqrt(real_part**2 + imag_part**2)
                #concat = torch.cat((mag_image,mag_ref),dim =1)
                #print(f'mag_image shape: {mag_image.shape}')
                #print(f'mag_ref shape: {mag_ref.shape}')
                #print(combined_input.shape)
                #mag_image = mag_image.permute(0, 3, 1, 2)
                #mag_ref = mag_ref.permute(0, 3, 1, 2)  # Permute to [batch_size, channels, height, width]
                #refined_image_real = similaritynet(real_part,real_part_ref) # mag_image,mag_ref
                #refined_image_imag = similaritynet(imag_part,imag_part_ref) # mag_image,mag_ref
                refined_image = similaritynet(mag_image,mag_ref)
                #refined_image_imag = similaritynet(imag_part,imag_part_ref)
                #print(f'Vit out shape: {refined_image.shape}')
                image = torch.cat((refined_image*torch.cos(phase),refined_image*torch.sin(phase)),dim=1)
                #image = torch.cat((refined_image_real,refined_image_imag),dim=1)
                #image = torch.sqrt(refined_image_real**2 + refined_image_imag**2)
                image = refined_image
                #image = torch.cat((refined_image,torch.zeros(refined_image.shape).to('cuda:0')),dim=1)
                image = image.permute(0, 2, 3, 1) # Permute back to original shape




            # classical cnn denoiser
            #image = image.permute(0,3,1,2) 

            #image = resnet(kspace=image,reference_image=reference_image,iter=iter)

            #image = image.permute(0,2,3,1)
            
            ## data consistency
            iter = iter +1
            """
            #if iter < self.num_grad_steps:
            rhs = zf_image + self.modl_lamda * image
            CG_alg = ConjGrad(Aop_fun=Sense.normal,b=rhs,verbose=False,l2lam=self.modl_lamda,max_iter=self.num_cg_steps)
            image = CG_alg.forward(rhs)
            """
        ## Transfer to magnitude
        """
        image = image.permute(0,3,1,2)
        #print(f'Image size: {image.shape}')
        real_part = image[:,0,:,:].unsqueeze(1)
        imag_part = image[:,1,:,:].unsqueeze(1)
        mag_image = torch.sqrt(real_part**2 + imag_part**2)           
        image = mag_image.permute(0, 2, 3, 1)
        """
        return image