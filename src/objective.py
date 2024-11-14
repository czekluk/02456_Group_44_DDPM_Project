import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

class NoiseObjective(torch.nn.Module):
    def __init__(self):
        super(NoiseObjective, self).__init__()

    def forward(self, epsilon, epsilon_pred):
        '''
        Objective function for the diffusion model.

        Inputs:
        - epsilon: True sampled noise
        - epsilon_pred: Predicted noise
        '''
        # Calculate the element-wise difference between epsilon and epsilon_pred
        diff = epsilon - epsilon_pred
        
        # Compute the L2 norm over spatial dimensions (H, W, C) for each sample
        # We can flatten the last three dimensions (channel, height, width)
        norm = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=1)  # Compute L2 norm for each sample across (C, H, W)
        
        # Return the mean squared norm over the batch
        return torch.mean(norm ** 2)
