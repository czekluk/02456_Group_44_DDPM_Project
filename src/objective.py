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
        return torch.mean(torch.linalg.norm(epsilon - epsilon_pred, ord=2, dim=(1,2,3))**2)