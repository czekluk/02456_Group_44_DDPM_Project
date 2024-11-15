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
    def __init__(self, model: torch.nn.Module, T: int = 1000, b0: float = 10e-4, bT: float = 0.02, img_shape: tuple = (1, 28, 28)):
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
        self.img_shape = img_shape

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
        t = torch.randint(1, self.T+1, (x.shape[0], 1), device=self.device)

        # sample e from N(0,I)
        e = self.normal.rsample(sample_shape=x.shape).to(self.device)

        # calculate alpha_t for every batch image
        ats = self.schedule.alpha_dash_list(t.squeeze().tolist()).to(self.device)
        # ats is of shape [batch_size, 1], expand it to match the shape of x (which is [batch_size, C, H, W])
        ats = ats.view(-1, 1, 1, 1)  # shape (batch_size, 1, 1, 1)
        ats = ats.expand(-1, x.shape[1], x.shape[2], x.shape[3])  # expand to (batch_size, C, H, W)

        # calculate model inputs
        x = x.to(self.device)
        t = t.to(self.device)

        x = torch.sqrt(ats) * x + torch.sqrt(1- ats) * e

        # zero gradients
        optimizer.zero_grad()

        # calculate model outputs
        e_pred = self.model(x, t.squeeze())

        # calculate loss
        loss = self.criterion(e, e_pred)

        # take gradient descent step
        loss.backward()
        optimizer.step()
        
        # print loss
        print(f'Loss: {loss.item()}')
        
        # return loss (for logging purposes)
        return loss.item()

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
        identity = torch.ones(x.shape[2], x.shape[3])
        identity = identity.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        std = (1-self.schedule.alpha_dash(t)) * identity

        # sample noise from N(mean, std)
        normal = torch.distributions.normal.Normal(mean, std)
        return normal.sample()
    
    def backward(self, x: torch.Tensor, t: int):
        '''
        Reverse process of the diffusion model.

        Inputs:
        - x: Noisy image at timestep t [B, C, H, W]
        - t: Current timestep in the reverse process

        Returns:
        - x_t_minus_1: Denoised image at timestep t-1
        '''
        # Predict the noise in the image at timestep t
        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.int64)

        noise_pred = self.model(x, t_tensor)

        # Retrieve alpha_t and beta_t from the schedule
        alpha_dash_t = self.schedule.alpha_dash(t).to(self.device)
        alpha_t = self.schedule.alpha(t).to(self.device)
        beta_t = self.schedule.beta(t).to(self.device)  # Variance for timestep t

        # Compute the mean for x_t_minus_1 using the noise prediction
        mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t)/torch.sqrt(1 - alpha_dash_t)) * noise_pred)
        
        # Sample noise using beta_t as the variance for the current timestep
        if t > 1:
            noise = torch.randn_like(x).to(self.device)  # Standard Gaussian noise
            x_t_minus_1 = mean + torch.sqrt(beta_t) * noise
        else:
            x_t_minus_1 = mean  # No noise at the final step

        return x_t_minus_1
    
    def sample(self, n_samples=10, t=0):
        '''
        Sampling operation of the diffusion model.

        Inputs:
        - n_samples: Number of samples to generate (batch size)
        - t: Sample a backward process at time t

        Returns:
        - samples: Generated samples as a tensor with shape [n_samples, C, H, W]
        '''
        # Step 1: Initialize with Gaussian noise with mean 0 and variance 1
        C, H, W = self.img_shape  # Assume img_shape is defined in the model as (channels, height, width)
        x_t = torch.randn((n_samples, C, H, W), device=self.device)  # Starting with pure noise

        # Step 2: Loop through timesteps in reverse
        for t in reversed(range(t, self.T)):  # Assumes num_timesteps is defined
            x_t = self.backward(x_t, t)

        # Step 3: Return the batch of generated samples
        return x_t
    
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

    