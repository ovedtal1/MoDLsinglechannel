"""
MoDL for single channel MRI 
by Ke Wang (kewang@berkeley.edu), 2020.

"""

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
from ImageFusion_Dualbranch_Fusion.channel_fusion import channel_f as channel_fusion
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

class Unrolled(nn.Module):
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
        #self.similarity = ImageFusionBlock().to(self.device)
        self.similarity = DenseFuseNet().to(self.device)
        checkpoint = torch.load('./ImageFusion_Dualbranch_Fusion/weights/best.pkl')
        # checkpoint = torch.load('./train_result/model_weight_new.pkl')
        self.similarity.load_state_dict(checkpoint['weight'])
        # Declare ResNets and RNNs for each unrolled iteration
        if self.share_weights:
            print("shared weights")
            self.resnets = nn.ModuleList([MyNetwork(2,2)] * self.num_grad_steps)
            self.similaritynets = nn.ModuleList([myIFCNN(fuse_scheme=0)] * self.num_grad_steps)
        else:
            print("No shared weights")
            self.resnets = nn.ModuleList([MyNetwork(2,2) for i in range(self.num_grad_steps)])
            self.similaritynets = nn.ModuleList([UNetPrior(2,2) for i in range(self.num_grad_steps)])

        # Declare step sizes for each iteration
#         init_step_size = torch.tensor([-2.0], dtype=torch.float32).to(params.device)
#         if fix_step_size:
#             self.step_sizes = [init_step_size] * num_grad_steps
#         else:
#             self.step_sizes = [torch.nn.Parameter(init_step_size) for i in range(num_grad_steps)] 


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
#         CG_alg = ConjGrad(Aop_fun=Sense.normal,b=zf_image,verbose=False,l2lam=0.05,max_iter=self.c)
#         cg_image = CG_alg.forward(zf_image)
#         pl.ImagePlot(zf_image.detach().cpu())
        reference_image = reference_image.permute(0,3,1,2) 
#         sys.exit()
        image = zf_image 
        #reference_image = reference_image.permute(0,3,1,2) 
        # Begin unrolled proximal gradient descent
        iter = 1
        for resnet, similaritynet in zip(self.resnets, self.similaritynets):
            # ResNet Denoiser
            #image = torch.cat([image, reference_image], dim=3)
            #image = image.permute(0,3,1,2) 
            #image = resnet(image)
            #image = image.permute(0,2,3,1)

            #print(image.shape)
            # Combine the output of ResNet with the reference image
            if (self.reference_mode == 1):
                combined_input = torch.cat([image, reference_image], dim=2)  # Concatenate along the channel dimension
                #print(combined_input.shape)
                combined_input = combined_input.permute(0, 3, 1, 2)  # Permute to [batch_size, channels, height, width]
                refined_image = similaritynet(combined_input)
                refined_image = refined_image.permute(0, 2, 3, 1) # Permute back to original shape
                image = refined_image
                #image = refined_image.permute(0, 2, 3, 1)
            #image = torch.cat([image, reference_image], dim=2) # for transformer
            image = image.permute(0,3,1,2) 
            #image = resnet(kspace=image,reference_image=reference_image,iter=iter)
            #image = resnet(image)
            
            if iter < 10:
                #with torch.no_grad():    
                real_part = image[:,0,:, :]
                imag_part = image[:,1,:, :]
                real_part_ref = reference_image[:,0,:, :]
                imag_part_ref = reference_image[:,1,:, :]
                phase = torch.atan2(imag_part, real_part)
                image_in = torch.sqrt(real_part**2 + imag_part**2).unsqueeze(1)
                reference_in = torch.sqrt(real_part_ref**2 + imag_part_ref**2).unsqueeze(1)
                #print(f'in similarity shape: {image_in.shape}')
                feature1a, feature2a = self.similarity.encoder(image_in,isTest=True),self.similarity.encoder(reference_in,isTest=True)
                featuresa = channel_fusion(feature1a, feature2a,is_test=True)
                image_real = self.similarity.decoder(featuresa)
                #feature1b, feature2b = self.similarity.encoder(imag_part.unsqueeze(1),isTest=True),self.similarity.encoder(imag_part_ref.unsqueeze(1),isTest=True)
                #featuresb = channel_fusion(feature1b, feature2b,is_test=True)
                #image_imag = self.similarity.decoder(featuresb)

                    #print(f'out similarity shape: {image.shape}')
                real_part = image_real[:,0,:,:] * torch.cos(phase)
                imag_part = image_real[:,0,:,:] * torch.sin(phase)
                    #print(f'real_part shape: {real_part.shape}')
                image = torch.stack((real_part, imag_part), dim=1)
                #print(f'real part shape: {real_part.shape}')
                #print(f'image shape: {image.shape}')
            #print(f'image shape after concat {image.shape}')
            """
            ## Scans fusion
            if iter < 2:
                #ssprint(image.shape)
                with torch.no_grad():
                    real_part = image[0,0,:, :]
                    imag_part = image[0,0,:, :]
                    phase = torch.atan2(imag_part, real_part)
                    image_in = image[0,...].permute( 1, 2, 0)
                    reference_image_in = reference_image[0,...].permute( 1, 2, 0)
                    #print(f'phase shape {phase.shape}')
                    #print(f'image size pre similarity: {image_in.shape}')
                    #print(f'reference size pre similarity: {reference_image_in.shape}')
                    image = self.similarity(image_in, reference_image_in)
                    #print(f'image shape after sim {image.shape}')
                    real_part = image[...,0] * torch.cos(phase)
                    imag_part = image[...,0] * torch.sin(phase)
                    image = torch.stack((real_part, imag_part), dim=2)
                    image = image.permute( 2, 0, 1)
                    image = image.unsqueeze(0)
                    #print(f'image shape after concat {image.shape}')
            """
            image = resnet(kspace=image,reference_image=reference_image,iter=iter)

            image = image.permute(0,2,3,1)
            
            ## data consistency
            iter = iter +1
            #if iter < self.num_grad_steps:
            rhs = zf_image + self.modl_lamda * image
            CG_alg = ConjGrad(Aop_fun=Sense.normal,b=rhs,verbose=False,l2lam=self.modl_lamda,max_iter=self.num_cg_steps)
            image = CG_alg.forward(rhs)
        
        return image