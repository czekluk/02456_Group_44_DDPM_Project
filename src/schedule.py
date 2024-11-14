import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

class LinearSchedule:
    def __init__(self, start_value: float, end_value: float, T: int):
        '''
        Linear schedule class for the diffusion model
        
        Inputs:
        - start_value: Initial value of the schedule (at t=0)
        - end_value: Final value of the schedule (at t=T)
        - T: Total number of iterations in the diffusion process
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.start_value = start_value
        self.end_value = end_value
        self.T = T

        self.schedule = torch.linspace(start_value, end_value, T)

        

    def beta(self, t: int):
        '''
        Get beta at iteration t
        
        Inputs: 
        - t: Iteration number
        '''
        return self.schedule[t]
    
    def alpha(self, t: int):
        '''
        Get alpha at iteration t
        
        Inputs:
        -t: Iteration number
        '''
        return 1 - self.beta(t)
    
    def alpha_dash(self, t: int):
        '''
        Get alpha dash at iteration t
        Product of alphas up to timestep t
        
        Inputs:
        -t: Iteration number
        '''
        alpha_dash = self.alpha(0)
        for i in range(1,t):
            alpha_dash *= self.alpha(i)
        return alpha_dash
    
    def alpha_dash_list(self, ts: torch.tensor):
        # Compute alphas as a list comprehension, with each alpha_dash(t) being a scalar value
        alphas = [self.alpha_dash(t) for t in ts]

        # Convert to a tensor and reshape to (batch_size, 1)
        return torch.tensor(alphas).view(-1, 1)