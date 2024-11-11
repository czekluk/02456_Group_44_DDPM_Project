import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from schedule import LinearSchedule
from objective import NoiseObjective

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DiffusionModel:
    def __init__(self, model: torch.nn.Module, T: int = 1000, b0: float = 10e-4, bT: float = 0.02):
        '''
        Diffusion model class implemnting the diffusion model as described in the "Denoising Diffusion Probabilistic Models" paper.
        Source: https://arxiv.org/pdf/2006.11239

        Inputs:
        - model: The model used in the diffusion process to predict noise at every iteration. (e.g. UNet)
        - T: Total number of iterations in the diffusion process
        - b0: Initial value of the beta schedule
        - bT: Final value of the beta schedule
        '''
        # Model related parameters
        self.model = model
        self.T = T
        self.uniform = torch.distributions.uniform.Uniform(1, T)
        self.normal = torch.distributions.normal.Normal(0, 1)
        self.schedule = LinearSchedule(b0, bT, T)

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
        2) Sample gaussian noise epsilon from N(0,I)
        3) Take gradient descent step to minimize the objective function

        Inputs:
        - x: Batch of training images [B, C, H, W]
        - optimizer: Optimizer used for training the model

        Returns:
        - loss: Loss value of the training step
        '''
        self.model.train()

        # sample t from uniform distribution
        t = self.uniform.sample(sample_shape=(x.shape[0],1))

        # sample e from N(0,I)
        e = self.normal.rsample(sample_shape=x.shape).to(self.device)

        # calculate alpha_t
        at = self.schedule.alpha_dash(t).to(self.device)

        # calculate model inputs
        x = x.to(self.device)
        t = t.to(self.device)
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

    def forward(self, x:torch.Tensor, t: int):
        '''
        Forward process of the diffusion model.

        Inputs:
        - x: Batch of images [B, C, H, W]
        - t: Time step in the diffusion process

        Returns:
        - x_t: Image with added noise at step t
        '''
        # calculate mean of forward sampling process
        mean = torch.sqrt(self.schedule.alpha_dash(t)) * x

        # calculate std of forward sampling process
        identity = torch.eye(x.shape[2], x.shape[3])
        identity = identity.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        std = (1-self.schedule.alpha_dash(t)) * identity

        # sample noise from N(mean, std)
        normal = torch.distributions.normal.Normal(mean, std)
        return normal.rsample(sample_shape=x.shape)

    def sample(n_samples: int):
        '''
        Sampling operation of the diffusion model.

        Inputs:
        - n_samples: Number of samples to generate (batch size)

        Returns:
        - samples: List of generated samples (List of tensors with shape [B, C, H, W])

        NOTE: This definition is just a proposal. Please feel free to change it to your needs.
        '''
        pass

    def save(self, path: str = os.path.join(PROJECT_BASE_DIR, 'results', 'models'),
             model_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-DiffusionModel.pth"):
        '''
        Save the model to a file.

        Inputs:
        - path: Path where to save the model
        - model_name: Name of the model file
        '''
        torch.save(self.model.state_dict(), os.path.join(path, model_name))
        print(f'Model saved to {os.path.join(path, model_name)}')

    def load(self, path: str):
        '''
        Load a diffusion model from file
        
        Inputs:
        - path: Path to the model file
        '''
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f'Model loaded from {path}')

    