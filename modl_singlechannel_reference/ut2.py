import os
import numpy as np
import matplotlib

#matplotlib.use('Agg')
import cv2
import pywt
import sys
import bm3d

import cfl
import sigpy as sp
import sigpy.plot as pl
import sigpy.mri as mr
from SSIM_PIL import compare_ssim
from PIL import Image
############################

def fft2c(input):
    input_copy = input.copy()
    output = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(input_copy)))
    return output

def ifft2c(input):
    input_copy = input.copy()
    output = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(input_copy)))
    return output
    
def awgn(s,SNRdB,L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
        """
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/np.size(s)#np.size(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if np.isrealobj(s):# check if input is real/complex object type
        n = np.sqrt(N0)*np.random.randn(s.shape[0],s.shape[1]) # computed noise
    else:
        n = np.sqrt(N0/2)*(np.random.randn(s.shape[0],s.shape[1])+1j*np.random.randn(s.shape[0],s.shape[1]))
    r = s + n # received signal
    return r
    
    
def resolutionDegrade(input,factor_x,factor_y):
    """
    Get input in image domain
    """
    kspace_in = fft2c(input)
    kspace_out = kspace_in
    len_x, len_y = kspace_in.shape
    kspace_out[1:int(len_x*(1-1/(factor_x))/2),:] = 0
    kspace_out[-1-int(len_x*(1-1/(factor_x))/2):-1,:] = 0
    kspace_out[:,1:int(len_y*(1-1/(factor_y))/2)] = 0
    kspace_out[:,-1-int(len_y*(1-1/(factor_y))/2):-1] = 0  
    output_img = ifft2c(kspace_out)
    
    return output_img, kspace_out

  
def kspaceCrop(input,factor_x,factor_y):
    kspace_in = fft2c(input)
    len_x, len_y = kspace_in.shape
    crop_x = int(len_x * (1 - 1 / factor_x) / 2)
    crop_y = int(len_y * (1 - 1 / factor_y) / 2)

    kspace_out = kspace_in[crop_x:-crop_x, crop_y:-crop_y]

    output_img = ifft2c(kspace_out)

    return output_img, kspace_out
    

def ImPlot(vector, vector_type,number_of_data, title, save_path):
    """
    Plot and save a 2D vector with appropriate axes and title.

    Parameters:
    - vector: 2D NumPy array representing the vector.
    - vector_type: Type of the vector ("kspace" or "image").
    - title: Title for the plot.
    - save_path: File path to save the plot.

    Returns:
    None
    """
    fig, ax = matplotlib.pyplot.subplots(figsize=(6*number_of_data,6))
    if vector_type == "kspace":
        ax.imshow(np.log(1+np.abs(vector)), cmap='gray',origin='lower')  # Assuming the vector contains complex values; adjust cmap if needed
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
    elif vector_type == "image":
        ax.imshow(np.abs(vector), cmap='gray',origin='lower')  # Assuming the vector contains complex values; adjust cmap if needed
        ax.set_axis_off()
    else:
        raise ValueError("Invalid vector_type. Choose 'kspace' or 'image'.")

    ax.set_title(title)
    matplotlib.pyplot.savefig(save_path)
    
    
def ImPlotDual(vector1, vector2, vector_type, title1, title2, top_title, save_path):
    """
    Plot and save two 2D vectors side by side with appropriate axes and titles.

    Parameters:
    - vector1: 2D NumPy array representing the first vector.
    - vector2: 2D NumPy array representing the second vector.
    - vector_type: Type of the vectors ("kspace" or "image").
    - title1: Title for the first plot.
    - title2: Title for the second plot.
    - top_title: Title for both images.
    - save_path: File path to save the plot.

    Returns:
    None
    """
    fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(12, 6))

    if vector_type == "kspace":
        axes[0].imshow(np.log(1 + np.abs(vector1)), cmap='gray', origin='lower')
        axes[0].set_xlabel("kx")
        axes[0].set_ylabel("ky")

        axes[1].imshow(np.log(1 + np.abs(vector2)), cmap='gray', origin='lower')
        axes[1].set_xlabel("kx")
        axes[1].set_ylabel("ky")

    elif vector_type == "image":
        axes[0].imshow(np.abs(vector1), cmap='gray', origin='lower')
        axes[0].set_axis_off()

        axes[1].imshow(np.abs(vector2), cmap='gray', origin='lower')
        axes[1].set_axis_off()

    else:
        raise ValueError("Invalid vector_type. Choose 'kspace' or 'image'.")

    axes[0].set_title(title1)
    axes[1].set_title(title2)
    matplotlib.pyplot.suptitle(top_title)
    matplotlib.pyplot.savefig(save_path)


def normalize(data):
    # Calculate the threshold (98% strongest sample)
    percentile_98 = np.percentile(np.abs(data), 98)
    # Normalize the data
    #normalized_data = data / percentile_97
    normalized_data= (data - np.min(data)) / percentile_98 - np.min(data)
    """
    percentile_98 = np.percentile(np.abs(data), 98)
    normalized_data= data / percentile_98 
    clipped_image = np.clip(normalized_data, -1, 1)
    """
    clipped_image = np.clip(normalized_data, 0, 1)
    return clipped_image

def normalizeComplex(data):
    # Calculate the threshold (96% strongest sample)
    percentile_98 = np.percentile(np.abs(data), 96)
    # Normalize the data
    #normalized_data = data / percentile_97
    normalized_data= data / percentile_98 
    clipped_image = np.clip(normalized_data, 0, 1)
    return normalized_data


def SoftThreshComplex(y,lam):
    # y - the data we want to threshold
    # lam - lambda, the threshold
    # Applying soft threshold
    mask1 = np.sqrt(np.dot(y, np.conjugate(y).T)) <= lam
    mask2 = np.sqrt(np.dot(y, np.conjugate(y).T)) > lam
    y[mask1] = 0
    # Apply soft threshold to the real part
    y.real[mask2] = (np.abs(y.real[mask2]) - lam) / np.abs(y.real[mask2]) * y.real[mask2]
    # Apply soft threshold to the imaginary part
    y.imag[mask2] = (np.abs(y.imag[mask2]) - lam) / np.abs(y.imag[mask2]) * y.imag[mask2]

    return y


def generate_variable_density_mask(shape, acceleration_factor,center):
    # Set x,y grid for funciton
    x, y = np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0]))

    # Set density function
    density_function =  np.exp(-((x - shape[1] / 2) ** 2 + (y - shape[0] / 2) ** 2) / (2 * (shape[1] / 4) ** 2))
    mask = np.zeros(shape, dtype=np.float32)

    # Set higher probability for the middle of the k-space
    middle = shape[1] // 2, shape[0] // 2
    density_function[middle[1]-center:middle[1]+center, middle[0]-center:middle[0]+center] = 1


    # Generate random samples based on the density function
    num_samples = int(np.prod(shape) / acceleration_factor)

    flat_density_function = density_function.flatten() / np.sum(density_function)

    samples = np.random.choice(np.arange(np.prod(shape)), size=num_samples, replace=False, p=flat_density_function)
    # Set the corresponding entries in the mask to 1
    mask.flat[samples] = 1

    return mask

def compressed_sensing(data_in,mask,lam,iterations):
    data_in_fft = fft2c(data_in)
    indices = np.where(mask == 1)  
    data = data_in[:]
    for i in range(iterations):
        #if i % 10 == 0:
        #    lam = lam/2
        data_wv = pywt.dwt2(data, 'db2')
        LL, (LH, HL, HH) = data_wv
        coeffs_combined = LL, (LH, HL, HH) 
        est_LL = pywt.threshold(LL, lam, mode='soft', substitute=0)#SoftThreshComplex(LL, lam)
        est_LH = pywt.threshold(LH, lam, mode='soft', substitute=0)#SoftThreshComplex(LH, lam)
        est_HL = pywt.threshold(HL, lam, mode='soft', substitute=0)#SoftThreshComplex(HL, lam)
        est_HH = pywt.threshold(HH, lam, mode='soft', substitute=0)#SoftThreshComplex(HH, lam)
        data_est_fft = fft2c(pywt.idwt2((est_LL, (est_LH, est_HL, est_HH)), 'db2'))
        data_est_fft[indices] = data_in_fft[indices]
        data = ifft2c(data_est_fft[:])

    return data




def wavelet(data_in):
    n = 3
    w = 'db2'
    coeffs = pywt.wavedec2(data_in,wavelet=w,level=n)
    arr,coeff_slices = pywt.coeffs_to_array(coeffs)

    return arr,coeff_slices

def iwavelet(data_wv,coeffs_slices):
    w = 'db2'
    coeffs_filt = pywt.array_to_coeffs(data_wv,coeffs_slices,output_format='wavedec2')
    arr = pywt.waverec2(coeffs_filt,wavelet=w)

    return arr  


def POCS_func(Y,mask,lambda_initial):
    beta = 0.5
    lambda_bar = 0.0025
    lambda_val = lambda_initial
    im_cs = ifft2c(Y)
    
    for iter in range(30):
        im_cs_wv, coeffs_slices = wavelet(im_cs)
        im_cs = iwavelet(SoftThresh(im_cs_wv, lambda_val),coeffs_slices)
        im_cs_fft = fft2c(im_cs)  
        im_cs_fft[mask != 0] = Y[mask != 0]
        im_cs = ifft2c(im_cs_fft)
        lambda_val = max(lambda_val * beta, lambda_bar)
    
    return im_cs

def SoftThresh(x, lambda_val):
    # Apply soft thresholding
    y = (np.abs(x) > lambda_val) * (x * (np.abs(x) - lambda_val) / (np.abs(x) + np.finfo(float).eps))
    return y

def FISTA_based_solver(Y, X0, W1, W2):
    numIter = 10
    beta = 0.8
    beta1 = 0.8
    beta2 = 0.8
    L = 100
    lambda_bar = 0.158
    lambda_bar_s = 0.158
    mu_bar = 0.025
    lambda_val = 30
    lambda2 = 5
    mu = 0.1
    t_k = 1
    t_k_m1 = 1

    X =  dwt2(ifft2c(Y))
    X_k_m1 = X

    for i in range(1, numIter + 1):
        Z = X + ((t_k_m1 - 1) / t_k) * (X - X_k_m1)

        temp_val1 = W2 * (idwt2(Z - X0))

        big_lambda = (np.abs(temp_val1) - lambda_val * mu)
        big_lambda = (big_lambda / (np.abs(temp_val1) + np.finfo(float).eps) )* temp_val1 * (big_lambda > 0)

        temp_val2 = W1 * Z
        big_lambda2 = (np.abs(temp_val2) - lambda2 * mu)
        big_lambda2 = (big_lambda2 / (np.abs(temp_val2) + np.finfo(float).eps) )* temp_val2 * (big_lambda2 > 0)
        
        element1 = dwt2(ifft2c((Y != 0) * fft2c(idwt2(X)) - Y))
        element2 = (W2 * (dwt2(temp_val1 - big_lambda)))
        element3 = W1 * (temp_val2 - big_lambda2)
        U = Z - (1 / L) *( element1 + (1 / mu) * (element2 + element3))

        temp_val = (np.abs(U) - mu / L)

        X_k_m1 = X

        X = temp_val / (np.abs(U) + np.finfo(float).eps) * U * (temp_val > 0)
        temp_Y = fft2c(idwt2(X)) 
        temp_Y[Y != 0] = Y[Y != 0]
        X = dwt2(ifft2c(temp_Y)) 

        t_k_m1 = t_k
        t_k = (1 + np.sqrt(4 * t_k * t_k + 1)) / 2
        lambda_val = max(beta1 * lambda_val, lambda_bar_s)
        mu = max(beta * mu, mu_bar)
        lambda2 = max(beta2 * lambda2, lambda_bar)

    return X

def FISTA_based_solver2(Y, X0, W1, W2,DC,W0):
    numIter = 10
    beta = 0.8
    beta1 = 0.8
    beta2 = 0.8
    L = 100
    lambda_bar = 0.158
    lambda_bar_s = 0.158
    mu_bar = 0.025
    lambda_val = 30
    lambda2 = 5
    mu = 0.1
    t_k = 1
    t_k_m1 = 1

    X =  dwt2(ifft2c(Y))
    X_k_m1 = X

    for i in range(1, numIter + 1):
        Z = X + ((t_k_m1 - 1) / t_k) * (X - X_k_m1)

        temp_val1 = W2 * (idwt2(Z - X0))

        big_lambda = (np.abs(temp_val1) - lambda_val * mu)
        big_lambda = (big_lambda / (np.abs(temp_val1) + np.finfo(float).eps) )* temp_val1 * (big_lambda > 0)

        temp_val2 = W1 * Z
        big_lambda2 = (np.abs(temp_val2) - lambda2 * mu)
        big_lambda2 = (big_lambda2 / (np.abs(temp_val2) + np.finfo(float).eps) )* temp_val2 * (big_lambda2 > 0)
        
        element1 = dwt2(ifft2c((Y != 0) * fft2c(idwt2(X)) - Y))
        element2 = (W2 * (dwt2(temp_val1 - big_lambda)))
        element3 = W1 * (temp_val2 - big_lambda2)
        U = Z - (1 / L) *( element1 + (1 / mu) * (element2 + element3))

        temp_val = (np.abs(U) - mu / L)

        X_k_m1 = X

        X = temp_val / (np.abs(U) + np.finfo(float).eps) * U * (temp_val > 0)
        temp_Y = fft2c(idwt2(X)) 
        ## Weighted data consistency
        if (DC == 0):
            #temp_Y[Y != 0] = (Y[Y != 0]*1 + temp_Y[Y != 0]*3)/4
            temp_Y[W0 == 1] = Y[W0 == 1]

        ## special data consistency
        #Kspace_diff_real = np.zeros(Y.shape)
        #Kspace_diff_imag = np.zeros(Y.shape)
        #Kspace_diff_real[Y != 0] = np.sqrt(pow(((temp_Y[Y != 0] - Y[Y != 0])).real,2))
        #Kspace_diff_imag[Y != 0] = np.sqrt(pow(((temp_Y[Y != 0] - Y[Y != 0])).imag,2))
        #thresh =  100
        #temp_Y[(Kspace_diff_real > thresh) | (Kspace_diff_imag > thresh)] = Y[(Kspace_diff_real > thresh) | (Kspace_diff_imag > thresh)]
        ## Classic data consistency
        if (DC == 1):
            temp_Y[Y != 0] = Y[Y != 0]
        X = dwt2(ifft2c(temp_Y)) 
        """
        ## Denoising option
        X_denoised_real = bm3d.bm3d(np.real(X).copy(),sigma_psd=0.1,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING) 
        X_denoised_imag = bm3d.bm3d(np.imag(X).copy(),sigma_psd=0.1,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)         
        X_denoised = X_denoised_real + 1j*X_denoised_imag
        X = (0*X.copy() + 10*X_denoised.copy())/10
        """
        t_k_m1 = t_k
        t_k = (1 + np.sqrt(4 * t_k * t_k + 1)) / 2
        lambda_val = max(beta1 * lambda_val, lambda_bar_s)
        mu = max(beta * mu, mu_bar)
        lambda2 = max(beta2 * lambda2, lambda_bar)

    return X

def FISTA_based_solver3(Y, X0, W1, W2,DC,W0):
    numIter = 10
    beta = 0.8
    beta1 = 0.8
    beta2 = 0.8
    L = 100
    lambda_bar = 0.158
    lambda_bar_s = 0.158
    mu_bar = 0.025
    lambda_val = 30
    lambda2 = 5
    mu = 0.1
    t_k = 1
    t_k_m1 = 1
    X =  dwt2(ifft2c(Y))
    X_k_m1 = X
    Y0 = fft2c(idwt2(X0))

    for i in range(1, numIter + 1):
        Z = X + ((t_k_m1 - 1) / t_k) * (X - X_k_m1)

        temp_val1 = W2 * (idwt2(Z - X0))

        mat1_expanded = np.expand_dims(fft2c(idwt2(X)), axis=( 2, 3, 4, 5, 6, 7, 8))
        mat2_expanded = np.expand_dims(fft2c(X0), axis=( 2, 3, 4, 5, 6, 7, 8))
        k_bart = np.stack((mat1_expanded, mat2_expanded), axis=0)
        sens = np.ones((2,X0.shape[0],X0.shape[1],1,1,1,1,1,1,1))
        #print(k_bart.shape)
        #print(sens.shape)
        #element4 = bart(1,'pics -d 0 -R L:7:7:0.0005 -i 10', k_bart,sens)
        #print(element4.shape)

        big_lambda = (np.abs(temp_val1) - lambda_val * mu)
        big_lambda = (big_lambda / (np.abs(temp_val1) + np.finfo(float).eps) )* temp_val1 * (big_lambda > 0)

        temp_val2 = W1 * Z
        big_lambda2 = (np.abs(temp_val2) - lambda2 * mu)
        big_lambda2 = (big_lambda2 / (np.abs(temp_val2) + np.finfo(float).eps) )* temp_val2 * (big_lambda2 > 0)
        
        element1 = dwt2(ifft2c((Y != 0) * fft2c(idwt2(X)) - Y))
        element2 = (W2 * (dwt2(temp_val1 - big_lambda)))
        element3 = W1 * (temp_val2 - big_lambda2)
        U = Z - (1 / L) *( element1 + (1 / mu) * (element2 + element3 ))
        #U = (9.2*U + 0.8*dwt2(element4[1,:,:]))/10
        temp_val = (np.abs(U) - mu / L)

        X_k_m1 = X

        X = temp_val / (np.abs(U) + np.finfo(float).eps) * U * (temp_val > 0)
        temp_Y = fft2c(idwt2(X)) 
        ## Weighted data consistency
        #if (DC == 0):
            #temp_Y[Y != 0] = (Y[Y != 0]*1 + temp_Y[Y != 0]*3)/4
            #temp_Y[W0 == 1] = Y[W0 == 1]

        ## special data consistency
        Kspace_diff_real = np.zeros(Y.shape)
        Kspace_diff_imag = np.zeros(Y.shape)
        Kspace_diff_real[Y != 0] = np.sqrt(pow(((temp_Y[Y != 0] - Y[Y != 0])).real,2))
        Kspace_diff_imag[Y != 0] = np.sqrt(pow(((temp_Y[Y != 0] - Y[Y != 0])).imag,2))
        thresh =  100
        thresh2 = 30
        temp_Y[(Kspace_diff_real > thresh) | (Kspace_diff_imag > thresh)] = Y[(Kspace_diff_real > thresh) | (Kspace_diff_imag > thresh)]
        temp_Y[(Kspace_diff_real < thresh2) | (Kspace_diff_imag < thresh2)] = Y0[(Kspace_diff_real < thresh2) | (Kspace_diff_imag < thresh2)]
        
        ## Classic data consistency
        if (DC == 1):
            temp_Y[Y != 0] = Y[Y != 0]
        X = dwt2(ifft2c(temp_Y)) 
        """
        ## Denoising option
        X_denoised_real = bm3d.bm3d(np.real(X).copy(),sigma_psd=0.2,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING) 
        X_denoised_imag = bm3d.bm3d(np.imag(X).copy(),sigma_psd=0.2,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)         
        X_denoised = X_denoised_real + 1j*X_denoised_imag
        X = (1*X.copy() + 9*X_denoised.copy())/10
        """
        t_k_m1 = t_k
        t_k = (1 + np.sqrt(4 * t_k * t_k + 1)) / 2
        lambda_val = max(beta1 * lambda_val, lambda_bar_s)
        mu = max(beta * mu, mu_bar)
        lambda2 = max(beta2 * lambda2, lambda_bar)

    return X

def coeffs2img(LL, coeffs):
    LH, HL, HH = coeffs
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))

def dwt2(im):
    coeffs = pywt.wavedec2(im, wavelet='db4', mode='per', level=4)
    Wim, rest = coeffs[0], coeffs[1:]
    for levels in rest:
        Wim = coeffs2img(Wim, levels)
    return Wim

def idwt2(Wim):
    coeffs = img2coeffs(Wim, levels=4)
    return pywt.waverec2(coeffs, wavelet='db4', mode='per')

def unstack_coeffs(Wim):
        L1, L2  = np.hsplit(Wim, 2) 
        LL, HL = np.vsplit(L1, 2)
        LH, HH = np.vsplit(L2, 2)
        return LL, [LH, HL, HH]

def img2coeffs(Wim, levels=4):
    LL, c = unstack_coeffs(Wim)
    coeffs = [c]
    for i in range(levels-1):
        LL, c = unstack_coeffs(LL)
        coeffs.insert(0,c)
    coeffs.insert(0, LL)
    return coeffs
    
def adaptive_CS_reconstruction(im0, im, iterations,mask):

    NUM_OF_ITERATIONS = iterations
    percentage_of_k_space = 0
    percentage_of_k_space_delta = 0.02
    epsilon1 = 0.1
    percentage_of_k_space = 0.0
    gamma = 0
    Y_0 = fft2c(im0)
    Y_full = fft2c(im)
    X0=dwt2(im0)
    W1 = np.ones(X0.shape)
    W2 = np.zeros(X0.shape)
    inverse_wav_baseline_representation=(1/(1+np.abs(X0)))

    for j in range(NUM_OF_ITERATIONS):
        # Increase the number of samples
        ##percentage_of_k_space += percentage_of_k_space_delta

        # Sample k-space
        ##sampl_mtx = generate_samples(f_R, f_VD, gamma, percentage_of_k_space)
        #Y = Y_full * sampl_mtx
        Y = Y_full * mask

        # Solve l1 minimization problem
        X = FISTA_based_solver(Y, X0, W1, W2)

        # Compute weighting coefficients for next iteration
        reconstructed_image = (idwt2(X))
        diff_in_wav = np.abs(dwt2(reconstructed_image - im0))
        map_diff = diff_in_wav / (diff_in_wav + 1)
        W1[map_diff > epsilon1] = 1
        W1[map_diff <= epsilon1] = inverse_wav_baseline_representation[map_diff <= epsilon1]
        W2 = 1 / (1 + np.abs(reconstructed_image - im0))
        #W2 = nonlinear_filter(W2, 6, 0.2, 0.8)
        gamma = np.mean(W2)

    # Transfer image to image domain
    im_adaptive = idwt2(X)

    return im_adaptive

def adaptive_CS_reconstruction2(im0, im, iterations,mask,DC,sigma_bm3d):

    NUM_OF_ITERATIONS = iterations
    percentage_of_k_space = 0
    percentage_of_k_space_delta = 0.02
    epsilon1 = 0.1
    epsilon2 = 0.9956
    percentage_of_k_space = 0.0
    gamma = 0
    Y_0 = fft2c(im0)
    Y_full = fft2c(im)
    X0=dwt2(im0)
    W0 = mask.copy()
    W1 = np.ones(X0.shape)
    W2 = np.zeros(X0.shape)
    inverse_wav_baseline_representation=(1/(1+np.abs(X0)))

    for j in range(NUM_OF_ITERATIONS):
        # Increase the number of samples
        ##percentage_of_k_space += percentage_of_k_space_delta

        # Sample k-space
        ##sampl_mtx = generate_samples(f_R, f_VD, gamma, percentage_of_k_space)
        #Y = Y_full * sampl_mtx
        Y = Y_full * mask

        # Solve l1 minimization problem
        # Assuming h is defined and implemented elsewhere
        X = FISTA_based_solver2(Y, X0, W1, W2,DC,W0)
        X_denoised_real = bm3d.bm3d(np.real(X).copy(),sigma_psd=sigma_bm3d,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING) 
        X_denoised_imag = bm3d.bm3d(np.imag(X).copy(),sigma_psd=sigma_bm3d,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)         
        X_denoised = X_denoised_real + 1j*X_denoised_imag
        X = (0*X.copy() + 10*X_denoised.copy())/10

        # Compute weighting coefficients for next iteration
        reconstructed_image = (idwt2(X))
        diff_in_wav = np.abs(dwt2(reconstructed_image - im0))
        map_diff = diff_in_wav / (diff_in_wav + 1)
        W1[map_diff > epsilon1] = 1
        W1[map_diff <= epsilon1] = inverse_wav_baseline_representation[map_diff <= epsilon1]
        W2 = 1 / (1 + np.abs(reconstructed_image - im0))
        diff_in_kspace = np.abs(fft2c(X) - Y_full)
        map_diff_kspace = diff_in_kspace / (diff_in_kspace + 1)
        W0[map_diff_kspace > epsilon2] = 1
        W0[map_diff_kspace <= epsilon2] = 0
        W0[mask == 0] = 0
        gamma = np.mean(W2)

    # Transfer image to image domain
    im_adaptive = idwt2(X)

    return im_adaptive


def adaptive_CS_reconstruction3(im0, im, iterations,mask,DC,sigma_bm3d):
    NUM_OF_ITERATIONS = iterations
    percentage_of_k_space = 0
    percentage_of_k_space_delta = 0.02
    epsilon1 = 0.1
    epsilon2 = 0.9956
    percentage_of_k_space = 0.0
    gamma = 0
    Y_0 = fft2c(im0)
    Y_full = fft2c(im)
    X0=dwt2(im0)
    W0 = mask.copy()
    W1 = np.ones(X0.shape)
    W2 = np.zeros(X0.shape)
    inverse_wav_baseline_representation=(1/(1+np.abs(X0)))

    for j in range(NUM_OF_ITERATIONS):
        # Increase the number of samples
        ##percentage_of_k_space += percentage_of_k_space_delta

        # Sample k-space
        ##sampl_mtx = generate_samples(f_R, f_VD, gamma, percentage_of_k_space)
        #Y = Y_full * sampl_mtx
        Y = Y_full * mask

        # Solve l1 minimization problem
        # Assuming h is defined and implemented elsewhere
        X = FISTA_based_solver3(Y, X0, W1, W2,DC,W0)
        X_denoised_real = bm3d.bm3d(np.real(X).copy(),sigma_psd=sigma_bm3d,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING) 
        X_denoised_imag = bm3d.bm3d(np.imag(X).copy(),sigma_psd=sigma_bm3d,stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)         
        X_denoised = X_denoised_real + 1j*X_denoised_imag
        X = (0*X.copy() + 10*X_denoised.copy())/10

        # Compute weighting coefficients for next iteration
        reconstructed_image = np.abs((idwt2(X)))
        diff_in_wav = np.abs(dwt2(reconstructed_image - im0))
        map_diff = diff_in_wav / (diff_in_wav + 1)
        W1[map_diff > epsilon1] = 1
        W1[map_diff <= epsilon1] = inverse_wav_baseline_representation[map_diff <= epsilon1]
        W2 = 1 / (1 + np.abs(reconstructed_image - im0))
        diff_in_kspace = np.abs(fft2c(X) - Y_full)
        map_diff_kspace = diff_in_kspace / (diff_in_kspace + 1)
        W0[map_diff_kspace > epsilon2] = 1
        W0[map_diff_kspace <= epsilon2] = 0
        W0[mask == 0] = 0
        gamma = np.mean(W2)

    # Transfer image to image domain
    im_adaptive = np.abs(idwt2(X))

    return im_adaptive


class error_metrics:

    def __init__(self,I_true,I_pred):
        # convert images from complex to magnitude (we do not want complex data for error calculation)
        self.I_true = np.abs(I_true)  
        self.I_pred = np.abs(I_pred)   
        
    def calc_NRMSE(self):    
        # Reshape the images into vectors
        I_true = np.reshape(self.I_true,(1,-1))   
        I_pred = np.reshape(self.I_pred,(1,-1))               
        # Mean Square Error
        self.MSE = np.square(np.subtract(I_true,I_pred)).mean()       
        # Root Mean Square Error
        self.RMSE = np.sqrt(self.MSE)
        # Normalized Root Mean Square Error
        rr = np.max(I_true) - np.min(I_true) # range
        self.NRMSE = self.RMSE/rr
        
    def calc_SSIM(self):
        # Note: in order to use the function compare_ssim, the images must be converted to PIL format

        # convert the images from float32 to uint8 format
        im1_mag_uint8 = (self.I_true * 255 / np.max(self.I_true)).astype('uint8')
        im2_mag_uint8 = (self.I_pred * 255 / np.max(self.I_pred)).astype('uint8')
        # convert from numpy array to PIL format
        im1_PIL = Image.fromarray(im1_mag_uint8)
        im2_PIL = Image.fromarray(im2_mag_uint8)

        self.SSIM = compare_ssim(im1_PIL, im2_PIL)


def nonlinear_filter(image, n, epsilon, threshold):
    height, width = image.shape
    filtered_image = image.copy()

    # Iterate over image patches
    for i in range(height - n + 1):
        for j in range(width - n + 1):
            # Extract patch of size nxn
            patch = image[i:i+n, j:j+n]

            # Calculate percentage of patch values > 1 - epsilon
            count_above_threshold = np.sum(patch > (1 - epsilon))
            patch_size = n * n
            percentage_above_threshold = count_above_threshold / patch_size

            # Apply thresholding rule
            if percentage_above_threshold >= threshold:
                filtered_image[i:i+n, j:j+n] = 1

    return filtered_image

def add_awgn_and_compute_mean(image, num_iterations, mean_type='arithmetic', noise_mean=0, noise_std=0.1):
    height, width = image.shape
    noisy_image = image.copy()
    images = []

    for _ in range(num_iterations):
        # Generate AWGN
        noise = np.random.normal(noise_mean, noise_std, size=(height, width))

        # Add noise to image
        noisy_image = noisy_image + noise

        # Store noisy image
        images.append(noisy_image)

    # Compute mean of all noisy images
    if mean_type == 'arithmetic':
        mean_image = np.mean(images, axis=0)
    elif mean_type == 'geometric':
        mean_image = np.exp(np.mean(np.log(np.array(images)), axis=0))
    else:
        raise ValueError("Invalid mean_type. Choose 'arithmetic' or 'geometric'.")

    return mean_image