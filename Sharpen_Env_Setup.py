# Kyle Kentner
# Dr. Doug Heisterkamp
# CS-5723: Artificial Intelligence
# 2 May 2022
# Image Sharpening via Reinforcement Learning
##################################################################################
# This file contains the setup and functions used by the RL model/agent.
##################################################################################

import Kentner_PA2 as kpa
import Kentner_Project_Util as kpu
import sys, getopt
import os
import cv2
import numpy as np
import math

# For reinforcement-learning (OpenAI Gym and Keras)
import Sharpen_Env_Setup as ses
import gym
from gym import Env
from gym.spaces import Discrete, Box
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

# Create custom RL environment; optimize sharpening for an image by adjusting the cutoff frequency for the FFT's high-pass filter
# https://www.youtube.com/watch?v=cO5g5qLrLSo
class SharpenEnv(Env):
    def __init__(self, f_name, i_processed, f_sharp=''):
        # Actions we can take when adjusting the HPF
        self.action_space = Discrete(9)
        
        # Observation space is an HPF array (cutoff frequencies below which image data will be discarded)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        
        # An HPF of 50 is the default
        self.state = 0
        
        # The base image's filename
        self.file_name = f_name
        
        # The output image (pass back by reference)
        self.img_processed = i_processed
        
        # Limit the HPF's extent based on image dimensions
        self.rows, self.columns, rgb = self.img_processed.shape
        self.hpf_limit = int(self.rows / 3) if self.rows < self.columns else int(self.columns / 3)
        print(self.hpf_limit)
        
        # The sharp image's filename, if applicable
        self.file_sharp = f_sharp
        
        # Set iteration limit
        if f_sharp == '':
            self.limit = 10
        else:
            self.limit = 10000
        
        # Optimized MSE sharp (training rewards will be based on improvements)
        self.mse_opt = 1000000000
        
        # Average image intensity (returned by FFT)
        self.intensity = 0
    
    # The step function performs the FFT with a new HPF cutoff each time
    def step(self, action):
		# Not done at start of step
        done = False
        
        # Get the mean-squared error between the output image and the blurry/sharp images, as well as the image itself
        mse_original, mse_sharp, img_out, self.intensity = kpu.FFT(self.file_name, self.state, self.file_sharp)
        print(' ')
        print(' ', self.state, mse_original, mse_sharp)
        
        # Convert from range [0..8] to [-9, -5, -3, -2, 0, 2, 3, 5, 9] (bias towards increasing the HPF cutoff)
        if action == 0:
            act = -9
        elif action == 1:
            act = -5
        elif action == 2:
            act = -3
        elif action == 3:
            act = -2
        elif action == 4:
            act = 0
        elif action == 5:
            act = 2
        elif action == 6:
            act = 3
        elif action == 7:
            act = 5
        elif action == 8:
            act = 9
        
        # Increment or decrement the current HPF cutoff and redo the sharpening
        self.state += act
        if self.state > self.hpf_limit:
            self.state = int(self.hpf_limit / 2)
        elif self.state <= 0:
            self.state = 5
        
        # Set placeholder for info
        info = {}
        
        # Check if the mean squared error is within the acceptable bounds
        if mse_sharp is not None and mse_original < mse_sharp:
            # Penalize a training iteration's result for being too similar to the original (blurry) image
            reward = -1
        
        elif mse_sharp is not None:
            # Reward a training iteration for approximating the sharp image
            if mse_sharp < self.mse_opt:
                # New optimum found
                reward = 1
                self.mse_opt = mse_sharp
            else:
                # Subtract the percent difference between the current MSE and the current optimum MSE
                reward = 1 - (mse_sharp - self.mse_opt)  / mse_sharp
                
            self.img_processed = img_out
        
        elif mse_original > self.intensity * 12:
            # Penalize a test iteration's result for being too different from the original (blurry) image
            reward = -1
            
        else:
            # Reward a test iteration for not deviating too much from the original image
            reward = 1
            self.img_processed = img_out
        
        # Decrement limit; quit when it equals zero
        self.limit -= 1
        if self.limit <= 0:
            done = True
        
		# Return step information
        return self.state, reward, done, info
    
    def render(self):
        # Implement visualization
        pass
    
    def reset(self):
        # Reset HPF cutoff
        self.state = 0
        
        # Reset duration of test
        if self.file_sharp == '':
            self.limit = 10
        else:
            self.limit = 10000
        
        # Optimized MSE sharp (training rewards will be based on improvements)
        self.mse_opt = 1000000000
        
        return self.state
