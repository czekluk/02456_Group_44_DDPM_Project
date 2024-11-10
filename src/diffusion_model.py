import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from schedule import LinearSchedule
from objective import NoiseObjective

class DiffusionModel:
    def __init__(self, model: torch.nn.Module, T: int = 1000):
        '''
        Diffusion model class implemnting the diffusion model as described in the "Denoising Diffusion Probabilistic Models" paper.
        Source: https://arxiv.org/pdf/2006.11239

        Inputs:
        - model: The model used in the diffusion process to predict noise at every iteration. (e.g. UNet)
        '''
        # Model related parameters
        self.model = model
        self.T = T
        self.uniform = torch.distributions.uniform.Uniform(1, T)
        self.normal = torch.distributions.normal.Normal(0, 1)
        self.schedule = LinearSchedule(10e-4, 0.02, T)

        # Training related parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = NoiseObjective()
        self.model.to(self.device)

        # Sampling related parameters

    def train(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        '''
        Single training iteration of the diffusion model.

        Process:
        1) Sample t from uniform distribution
        2) Sample gaussia noise epsilon from N(0,I)
        3) Take gradient descent step to minimize the objective function

        Inputs:
        - x: Batch of training images [B, C, H, W]
        '''
        self.model.train()

        # sample t from uniform distribution
        t = self.uniform.sample(sample_shape=(x.shape[0],1)).to(self.device)

        # sample e from N(0,I)
        e = self.normal.rsample(sample_shape=x.shape).to(self.device)

        # calculate alpha_t
        at = self.schedule.alpha_dash(t).to(self.device)

        # calculate model inputs
        x = x.to(self.device)
        x = torch.sqrt(at) * x + torch.sqrt(1- at) * e

        # zero gradients
        optimizer.zero_grad()

        # calculate model outputs
        e_pred = self.model(x, t)

        # calculate loss
        loss = self.criterion(e, e_pred)

        # take gradient descent step
        loss.backward()
        optimizer.step()
        
        # print loss
        print(f'Loss: {loss.item()}')
        
        # return loss (for logging purposes)
        return loss.item().detach().cpu().numpy()

    def sample():
        '''
        Sampling operation of the diffusion model.
        '''
        pass

    