
import os, sys
import logging
import random
import h5py
import shutil
import time
import argparse
import numpy as np
import sigpy.plot as pl
import torch
import sigpy as sp
import torchvision
from torch import optim
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
# import custom libraries
from utils import transforms as T
from utils import subsample as ss
from utils import complex_utils as cplx
from utils.resnet2p1d import generate_model
from utils.flare_utils import roll
# import custom classes
from utils.datasets import SliceData
from subsample_fastmri import MaskFunc
from MoDL_single import UnrolledModel
import argparse
from models.SAmodel import MyNetwork
from models.Unrolled import Unrolled
from models.UnrolledRef import UnrolledRef
from models.UnrolledTransformer import UnrolledTrans
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

## Data 
class DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.
    """

    def __init__(self, mask_func, args, use_seed=False):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def __call__(self, kspace, target, reference, reference_kspace,slice):
       
        im_lowres = abs(sp.ifft(sp.resize(sp.resize(kspace,(172,24)),(172,108))))
        magnitude_vals = im_lowres.reshape(-1)
        k = int(round(0.05 * magnitude_vals.shape[0]))
        scale = magnitude_vals[magnitude_vals.argsort()[::-1][k]]
        kspace = kspace/scale

        # Convert everything from numpy arrays to tensors
        kspace_torch = cplx.to_tensor(kspace).float()   
        target_torch = cplx.to_tensor(target).float() / scale
        
        # Use poisson mask instead
        mask2 = sp.mri.poisson((172,108), 2, calib=(18, 14), dtype=float, crop_corner=False, return_density=True, seed=0, max_attempts=6, tol=0.01)
        mask_torch = torch.stack([torch.tensor(mask2).float(),torch.tensor(mask2).float()],dim=2)
    
        #kspace_torch = T.kspace_cut(mask_torch,0.5)
        kspace_torch = T.awgn_torch(kspace_torch,15,L=1)
        kspace_torch = kspace_torch*mask_torch

    
        ### Reference addition ###
        im_lowres_ref = abs(sp.ifft(sp.resize(sp.resize(reference_kspace,(172,24)),(172,108))))
        magnitude_vals_ref = im_lowres_ref.reshape(-1)
        k_ref = int(round(0.05 * magnitude_vals_ref.shape[0]))
        scale_ref = magnitude_vals_ref[magnitude_vals_ref.argsort()[::-1][k_ref]]
        reference_torch = cplx.to_tensor(reference).float()/ scale_ref
        # Resolution degrading
       
        return kspace_torch, target_torch,mask_torch, reference_torch

def create_datasets(args):
    # Generate k-t undersampling masks
    train_mask = MaskFunc([0.08],[4])
    train_data = SliceData(
        root=str(args.data_path),
        transform=DataTransform(train_mask, args),
        sample_rate=1
    )
    return train_data
def create_data_loaders(args):
    train_data = create_datasets(args)
#     print(train_data[0])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
    )
    return train_loader
def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer
    

## Hyper parameters
params = Namespace()
params.data_path = "./registered_data/"
params.batch_size = 16
params.num_grad_steps = 4 #4
params.num_cg_steps = 8
params.share_weights = True
params.modl_lamda = 0.05
params.lr = 0.001
params.weight_decay = 0
params.lr_step_size = 5
params.lr_gamma = 0.5
params.epoch = 101
params.reference_mode = 0
params.reference_lambda = 0.1

train_loader = create_data_loaders(params)

## Load model and Loss selection
single_MoDL = UnrolledModel(params).to(device)
optimizer = build_optim(params, single_MoDL.parameters())
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.lr_step_size, params.lr_gamma)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params.lr_gamma, patience=10, verbose=True)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()

epochs_plot = []
losses_plot = []

for epoch in range(params.epoch):
    single_MoDL.train()
    avg_loss = 0.

    for iter, data in enumerate(train_loader):
        input,target,mask,reference = data
        input = input.to(device)
        target = target.to(device)
        mask = mask.to(device)
        reference = reference.to(device)

        im_out = single_MoDL(input.float(),reference_image=reference,mask=mask)
        loss = criterion(im_out,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        #if iter % 125 == 0:
            #logging.info(
            #    f'Epoch = [{epoch:3d}/{params.epoch:3d}] '
            #    f'Iter = [{iter:4d}/{len(train_loader):4d}] '
            #    f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g}'
            #)
    #Saving the model
    exp_dir = "L2_checkpoints_poisson_x2_MoDL/"
    if epoch % 10 == 0:
        torch.save(
            {
                'epoch': epoch,
                'params': params,
                'model': single_MoDL.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir
            },
            f=os.path.join(exp_dir, 'model_%d.pt'%(epoch))
        )

    scheduler.step(loss.item())
    # Append epoch and average loss to plot lists
    epochs_plot.append(epoch)
    losses_plot.append(loss.item())

# Plotting the loss curve
plt.figure()
plt.plot(epochs_plot, losses_plot, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MoDL L2 train Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(exp_dir, 'loss_plot_plato.png'))  # Save plot as an image

# Save all_losses to a file for later comparison
losses_file = os.path.join(exp_dir, 'all_losses_plato.txt')
with open(losses_file, 'w') as f:
    for loss in losses_plot:
        f.write(f'{loss}\n')
