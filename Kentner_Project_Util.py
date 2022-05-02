import Kentner_PA2 as kpa
import sys, getopt
import os
import cv2
import numpy as np
import math

# For reinforcement-learning (OpenAI Gym and Keras)
import Sharpen_Env_Setup as ses
import gym
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Suppress Tensorflow deprecation warnings
import warnings
warnings.filterwarnings('ignore')
import logging
logger = logging.getLogger()
logger.disabled = True

# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
def FFT(file_name, hpf, sharp_name):
    # Get the base image
    img_color = cv2.imread(file_name, -1)    # Color image
    img_fft = cv2.imread(file_name, 0)       # Grayscale image to be filtered
    img_original = cv2.imread(file_name, 0)  # Original grayscale image
    img_sharp = None
    
    # Get the target image if this is a paired (training) operation
    if sharp_name != '':
        # Grayscale sharp image
        img_sharp = cv2.imread(sharp_name, 0)

    # Fast Fourier Transform
    f = np.fft.fft2(img_fft)
    f_original = np.fft.fft2(img_original)
    
    # Move the original image's zero-frequency component to the middle of the spectrum for (allows filtering)
    fshift = np.fft.fftshift(f)
    fshift_original = np.fft.fftshift(f_original) # For recombination later

    # Row and column indices for the high-pass filter
    rows, cols = img_fft.shape
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # Apply the HPF to fshift
    if hpf != 0:
        fshift[mid_row - hpf:mid_row + hpf, mid_col - hpf:mid_col + hpf] = 0

    # Scale the signal of the sharpened image
    boost = 2

    # Add the filtered frequencies to the base frequencies to get the sharpened grayscale image
    # https://eeweb.engineering.nyu.edu/~yao/EE3414/image_filtering.pdf (slide 29)
    for i in range(len(fshift)):
        # Multiply the high-pass-filtered image's intensity by the boost coefficient
        fshift[i] *= boost
        
        # Sharpened image frequency is the boosted filtered intensity plus the original intensity
        fshift[i] += fshift_original[i]
        
        # Original image frequency (adjust for MSE if no high-pass filter is applied for accurate comparison)
        if hpf == 0:
            fshift_original[i] += boost * fshift_original[i]

    # Perform the inverse shift to return the frequency values to their original positions
    f_ishift = np.fft.ifftshift(fshift)

    # Perform the inverse FFT and take the absolute value to recover the filtered image in the space domain
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    #cv2.imwrite('img_back.jpg', img_back)
    
    # Get a modified version of the original image for comparison
    img_compare = np.fft.ifft2(fshift_original)
    img_compare = np.abs(img_compare)
    #cv2.imwrite('img_compare.jpg', img_compare)
    
    # Iterate over each pixel; get the RGB values, and scale based on the processed grayscale image to emphasize edges
    for y in range(rows):
        for x in range(cols):
            # Initialize a node corresponding to this pixel
            cur_node = kpa.Node()
            cur_node.location = [y, x]
            cur_node.rgb = img_color[y, x]
            
            # Determine how much the current pixel was boosted versus the original image
            if img_original[y, x] != 0:
                coeff = img_back[y, x] / img_original[y, x]
            else:
                coeff = 1
            
            # Multiply the color image by the coefficient to sharpen
            cur_node.rgb[0] = int(cur_node.rgb[0] * coeff) if cur_node.rgb[0] * coeff < 255 else 255
            cur_node.rgb[1] = int(cur_node.rgb[1] * coeff) if cur_node.rgb[1] * coeff < 255 else 255
            cur_node.rgb[2] = int(cur_node.rgb[2] * coeff) if cur_node.rgb[2] * coeff < 255 else 255
            
    # Compare the error between the final image and the original (unpaired) or target (paired) image
    if img_sharp is None:
        # Return the mean-squared error between the original and filtered grayscale images, along with the final color image
        return MSE(img_compare, img_back), None, img_color
    else:
        # Return the mean-squared error between the filtered and sharp grayscale images, along with the final color image
        return MSE(img_compare, img_back), MSE(img_sharp, img_back), img_color
    
    
# Calculate the peak signal-to-noise ratio (PSNR) between input and output images
# https://www.mathworks.com/help/vision/ref/psnr.html#:~:text=The%20mean%2Dsquare%20error%20(MSE,MSE%2C%20the%20lower%20the%20error.
def PSNR(mse):
    # Maximum fluctuation for an 8-bit RGB channel
    R = 255
    
    # Get the argument for the logarithm
    arg = R ** 2 / mse
    
    return 10 * math.log(arg, 10)


# Calculate the mean squared error between two images
# https://www.mathworks.com/help/vision/ref/psnr.html#:~:text=The%20mean%2Dsquare%20error%20(MSE,MSE%2C%20the%20lower%20the%20error.
def MSE(img_old, img_new):
    # Loop variables
    rows, cols = img_old.shape
    mse = 0
    
    # Iterate over each pixel
    for y in range(rows):
        for x in range(cols):
            mse += (img_old[y, x] - img_new[y, x]) ** 2
    
    # Divide the sum by the number of pixels
    mse /= (rows * cols)
    
    return mse


def overwrite_image(img_old, img_new):
    # Loop variables
    rows, cols, rgb = img_old.shape
    
    # Iterate over each pixel
    for y in range(rows):
        for x in range(cols):
            # Overwrite the RGB values of each pixel
            img_old[y, x][0] = 0
            img_old[y, x][1] = 0
            img_old[y, x][2] = 0
            
            '''
            img_old[y, x][0] = img_new[y, x][0]
            img_old[y, x][1] = img_new[y, x][1]
            img_old[y, x][2] = img_new[y, x][2]
            '''


def FFT_Test(file_name, hpf, sharp_name='', csp_demo=False):
    # Get the base image
    img_color = cv2.imread(file_name, -1)    # Color image
    img_fft = cv2.imread(file_name, 0)       # Grayscale image to be filtered
    img_original = cv2.imread(file_name, 0)  # Original grayscale image
    img_sharp = None
    
    # Get the target image if this is a paired (training) operation
    if sharp_name != '':
        # Grayscale sharp image
        img_sharp = cv2.imread(sharp_name, 0)

    # Fast Fourier Transform
    f = np.fft.fft2(img_fft)
    f_original = np.fft.fft2(img_original)
    
    # Move the original image's zero-frequency component to the middle of the spectrum for (allows filtering)
    fshift = np.fft.fftshift(f)
    fshift_original = np.fft.fftshift(f_original) # For recombination later

    # Row and column indices for the high-pass filter
    rows, cols = img_fft.shape
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # Apply the HPF to fshift
    if hpf != 0:
        fshift[mid_row - hpf:mid_row + hpf, mid_col - hpf:mid_col + hpf] = 0

    # Scale the signal of the sharpened image
    if not csp_demo:
        boost = 2
    else:
        boost = 20
    
    # Add the filtered frequencies to the base frequencies to get the sharpened grayscale image
    # https://eeweb.engineering.nyu.edu/~yao/EE3414/image_filtering.pdf (slide 29)
    for i in range(len(fshift)):
        # Multiply the high-pass-filtered image's intensity by the boost coefficient
        fshift[i] *= boost
        
        # Recombine the filtered image and the original image unless using the CSP method
        if not csp_demo:
            # Sharpened image frequency is the boosted filtered intensity plus the original intensity
            fshift[i] += fshift_original[i]
        
            # Original image frequency (adjust for MSE if no high-pass filter is applied for accurate comparison)
            if hpf == 0:
                fshift_original[i] += boost * fshift_original[i]

    # Perform the inverse shift to return the frequency values to their original positions
    f_ishift = np.fft.ifftshift(fshift)

    # Perform the inverse FFT and take the absolute value to recover the filtered image in the space domain
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # If this is the CSP method, return the filtered image and original color image
    if csp_demo:
        cv2.imwrite('img_filtered.jpg', img_back)
        return img_back, img_color
    
    # Get a modified version of the original image for comparison
    img_compare = np.fft.ifft2(fshift_original)
    img_compare = np.abs(img_compare)
    #cv2.imwrite('img_compare.jpg', img_compare)
    
    # Iterate over each pixel; get the RGB values, and scale based on the processed grayscale image to emphasize edges
    for y in range(rows):
        for x in range(cols):
            # Initialize a node corresponding to this pixel
            cur_node = kpa.Node()
            cur_node.location = [y, x]
            cur_node.rgb = img_color[y, x]
            
            # Determine how much the current pixel was boosted versus the original image
            if img_original[y, x] != 0:
                coeff = img_back[y, x] / img_original[y, x]
            else:
                coeff = 1
            
            # Multiply the color image by the coefficient to sharpen
            cur_node.rgb[0] = int(cur_node.rgb[0] * coeff) if cur_node.rgb[0] * coeff < 255 else 255
            cur_node.rgb[1] = int(cur_node.rgb[1] * coeff) if cur_node.rgb[1] * coeff < 255 else 255
            cur_node.rgb[2] = int(cur_node.rgb[2] * coeff) if cur_node.rgb[2] * coeff < 255 else 255
            
    # Write the final image
    cv2.imwrite('IMG_OUT.jpg', img_color)
    
    # Return the 
    return img_color
