"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from utils import complex_utils as cplx

import torch
from torch import nn
class SenseModel_single(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.
    """
    def __init__(self, coord=None, weights=None):
        super().__init__()
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def _adjoint_op(self, kspace):
        image = ifft2(self.weights * kspace)
        return image

    def _forward_op(self, image):
        kspace = self.weights * fft2(image)
        return kspace

    def forward(self, input, adjoint=False):
        if adjoint:
            output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output

class SenseModel(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.
    """
    def __init__(self, maps, coord=None, weights=None):
        super().__init__()

        self.maps = maps

        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def _adjoint_op(self, kspace):
        image = ifft2(self.weights * kspace)
        image = cplx.mul(image.unsqueeze(-2), cplx.conj(self.maps))
        return image.sum(-3)

    def _forward_op(self, image):
        kspace = cplx.mul(image.unsqueeze(-3), self.maps)
        kspace = self.weights * fft2(kspace.sum(-2))
        return kspace

    def forward(self, input, adjoint=False):
        if adjoint:
            output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output


class ArrayToBlocks(nn.Module):
    def __init__(self, block_size, image_shape, overlapping=False):
        """
        A module that extracts spatial patches from a 6D array with size [1, x, y, t, e, 2].
        Output is also a 6D array with size [N, block_size, block_size, t, e, 2].
        """
        super().__init__()

        # Get image / block dimensions
        self.block_size = block_size
        self.image_shape = image_shape
        _, self.nx, self.ny, self.nt, self.ne, _ = image_shape

        # Overlapping vs. non-overlapping block settings
        if overlapping:
            block_stride = self.block_size // 2
            # Use Hanning window to reduce blocking artifacts
            win1d = torch.hann_window(block_size, dtype=torch.float32) ** 0.5
            self.win = win1d[None,:,None,None,None,None] * win1d[None,None,:,None,None,None]
        else:
            block_stride = self.block_size
            self.win = torch.tensor([1.0], dtype=torch.float32)
            
            
        # Figure out padsize (to avoid black bars)
        num_blocks_x = (self.nx // self.block_size) + 2
        num_blocks_y = (self.ny // self.block_size) + 2
        self.pad_x = (self.block_size*num_blocks_x - self.nx) // 2
        self.pad_y = (self.block_size*num_blocks_y - self.ny) // 2
        nx_pad = self.nx + 2*self.pad_x
        ny_pad = self.ny + 2*self.pad_y

        # Compute total number of blocks
        num_blocks_x = (self.nx-self.block_size+2*self.pad_x) / block_stride + 1
        num_blocks_y = (self.ny-self.block_size+2*self.pad_y) / block_stride + 1
        self.num_blocks = int(num_blocks_x * num_blocks_y)

        # Set fold params
        self.fold_params = dict(kernel_size=2*(block_size,), stride=block_stride)
        self.unfold_op = nn.Unfold(**self.fold_params)
        self.fold_op = nn.Fold(output_size=(ny_pad, nx_pad), **self.fold_params)

    def extract(self, images):
        # Re-shape into a 4D array because nn.Unfold requires it >:(
        images = images.reshape([1, self.nx, self.ny, self.nt*self.ne*2]).permute(0,3,2,1)

        # Pad array
        images = nn.functional.pad(images, 2*(self.pad_x,) + 2*(self.pad_y,), mode='constant')

        # Unfold array into vectorized blocks
        blocks = self.unfold_op(images) # [1, nt*ne*2*bx*by, n]

        # Reshape into 2D blocks
        shape_out = (self.nt, self.ne, 2, self.block_size, self.block_size, self.num_blocks)
        blocks = blocks.reshape(shape_out).permute(5,4,3,0,1,2)

        # Apply window
        blocks *= self.win.to(images.device)

        return blocks

    def combine(self, blocks):
        # Apply window 
        blocks *= self.win.to(blocks.device)

        # Reshape back into nn.Fold format
        blocks = blocks.permute(3,4,5,2,1,0)
        blocks = blocks.reshape((1, self.nt*self.ne*2*self.block_size**2, self.num_blocks))

        # Fold blocks back into array
        images = self.fold_op(blocks) # [1, nt*ne*2, ny, nx]

        # Crop zero-padded images
        images = center_crop(images.permute(0,3,2,1), [1, self.nx, self.ny, self.nt*self.ne*2])
        images = images.reshape(self.image_shape)

        return images

    def forward(self, input, adjoint=False):
        if adjoint:
            output = self.combine(input)
        else:
            output = self.extract(input)
        return output


def decompose_LR(images, num_basis, block_size=16, overlapping=False, block_op=None):
    """
    Decomposes spatio-temporal data into spatial and temporal basis functions (L, R)
    """
    # Get image dimensions
    _, nx, ny, nt, ne, _ = images.shape
    nb = num_basis

    # Initialize ArrayToBlocks op if it hasn't been initialized already
    if block_op is None:
        block_op = ArrayToBlocks(block_size, images.shape, overlapping=overlapping)

    # Extract spatial blocks across images
    blocks = block_op(images)
    nblks = blocks.shape[0] # number of blocks
    blk_size = blocks.shape[1] # block shape [blk_size, blk_size]

    # Reshape into batch of 2D matrices
    blocks = blocks.permute(0,1,2,4,3,5)
    blocks = blocks.reshape((nblks, blk_size*blk_size*ne, nt, 2))

    # Perform SVD to get left and right singular vectors for each patch
    U, S, V = cplx.svd(blocks, compute_uv=True)

    # Truncate singular values and vectors
    U = U[:, :, :nb, :] # [N, Px*Py*E, B, 2]
    S = S[:, :nb]       # [N, B]
    V = V[:, :, :nb, :] # [N, T, B, 2]

    # Combine and reshape matrices
    S_sqrt = S.reshape((nblks, 1, 1, 1, 1, nb, 1)).sqrt()
    L = U.reshape((nblks, blk_size, blk_size,  1, ne, nb, 2)) * S_sqrt
    R = V.reshape((nblks,        1,        1, nt,  1, nb, 2)) * S_sqrt

    return L, R


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension containing
            real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    ndims = len(list(data.size()))

    if ndims == 5:
        data = data.permute(0,3,1,2,4)
    elif ndims == 6:
        data = data.permute(0,3,4,1,2,5)
    elif ndims > 6:
        raise ValueError('fft2: ndims > 6 not supported!')

    data = ifftshift(data, dim=(-3, -2))
    data_complex = torch.complex(data[...,0], data[...,1])
    data_fft = torch.fft.fftn(data_complex,dim=(-2, -1), norm="ortho")
    data = torch.stack((data_fft.real, data_fft.imag), dim=-1)
    data = fftshift(data, dim=(-3, -2))

    if ndims == 5:
        data = data.permute(0,2,3,1,4)
    elif ndims == 6:
        data = data.permute(0,3,4,1,2,5)
    elif ndims > 6:
        raise ValueError('fft2: ndims > 6 not supported!')

    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data with the last dimension containing
            real and imaginary components.
        dims (2-tuple): Containing spatial dimension indices.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    ndims = len(list(data.size()))

    if ndims == 5:
        data = data.permute(0,3,1,2,4)
    elif ndims == 6:
        data = data.permute(0,3,4,1,2,5)
    elif ndims > 6:
        raise ValueError('ifft2: ndims > 6 not supported!')

    data = ifftshift(data, dim=(-3, -2))
    data_complex = torch.complex(data[...,0], data[...,1])
    data_ifft = torch.fft.ifftn(data_complex,dim=(-2, -1), norm="ortho")
    data = torch.stack((data_ifft.real, data_ifft.imag), dim=-1)
    data = fftshift(data, dim=(-3, -2))

    
    if ndims == 5:
        data = data.permute(0,2,3,1,4)
    elif ndims == 6:
        data = data.permute(0,3,4,1,2,5)
    elif ndims > 6:
        raise ValueError('ifft2: ndims > 6 not supported!')

    return data


def root_sum_of_squares(x, dim=0):
    """
    Compute the root sum-of-squares (RSS) transform along a given dimension of a complex-valued tensor.
    """
    assert x.size(-1) == 2
    return torch.sqrt((x ** 2).sum(dim=-1).sum(dim))


def time_average(data, dim, eps=1e-6, keepdim=True):
    """
    Computes time average across a specified axis.
    """
    mask = cplx.get_mask(data)
    return data.sum(dim, keepdim=keepdim) / (mask.sum(dim, keepdim=keepdim) + eps)


def sliding_window(data, dim, window_size):
    """
    Computes sliding window with circular boundary conditions across a specified axis.
    """
    assert 0 < window_size <= data.shape[dim]

    windows = [None] * data.shape[dim]
    for i in range(data.shape[dim]):
        data_slide = roll(data, int(window_size/2)-i, dim)
        window = data_slide.narrow(dim, 0, window_size)
        windows[i] = time_average(window, dim)

    return torch.cat(windows, dim=dim)


def center_crop(data, shape):
    """
    Apply a center crop to a batch of images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. 
        shape (list of ints): The output shape. If shape[dim] = -1, then no crop 
            will be applied in that dimension.
    """
    for i in range(len(shape)):
        if (shape[i] == data.shape[i]) or (shape[i] == -1):
            continue
        assert 0 < shape[i] <= data.shape[i]
        idx_start = (data.shape[i] - shape[i]) // 2
        data = data.narrow(i, idx_start, shape[i])

    return data


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def PSNR(input, target):
    eps = 1e-8
    return -10*torch.log10(torch.mean((input - target) ** 2, dim=[1, 2, 3])+eps)

