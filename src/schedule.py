import torch

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
    
    def plot_schedule(self):
        '''
        Plot the schedule
        '''
        plt.plot(self.schedule)
        plt.xlabel('Iteration t')
        plt.ylabel('Noise Variance β')
        plt.title(f'Linear schedule from {self.start_value} to {self.end_value} (T={self.T})')
        plt.show()
    

class CosineSchedule:
    def __init__(self, T: int, jitter: float = 0.008):
        '''
        Cosine schedule class for the diffusion model as described in 'Improved Denoising Diffusion Probabilistic Models' at
        https://arxiv.org/pdf/2102.09672 section 3.2
        
        Inputs:
        - T: Total number of iterations in the diffusion process
        - jitter: Random noise to add to the schedule to improve stability (default: 0.008)
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.s = jitter
        self.T = T
        
        self.f_0 = self._get_f_for_t(0)
        self.schedule = torch.tensor([self._beta(t) for t in range(T+1)]).to(self.device)

    def _get_f_for_t(self, t: int):
        '''
        Get f(t) for iteration t as described in 'Improved Denoising Diffusion Probabilistic Models'
        '''
        return np.cos(((t / self.T + self.s) / (1 + self.s)) * (np.pi / 2)) ** 2

    def _alpha(self, t: int):
        '''
        Get alpha at iteration t
        
        Inputs:
        -t: Iteration number
        '''
        return self._get_f_for_t(t) / self.f_0
    
    def _beta(self, t: int):
        '''
        Get beta at iteration t
        
        Inputs: 
        - t: Iteration number
        '''
        if t == 0: # No alpha for t-1
            return 0
        beta = 1 - (self._alpha(t) / self._alpha(t - 1))
        if beta >= 0.999: # Clamp beta to 0.999
            beta = 0.999
        return beta

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
    
    def plot_schedule(self):
        '''
        Plot the schedule
        '''
        plt.plot(self.schedule)
        plt.xlabel('Iteration t')
        plt.ylabel('Noise Variance β')
        plt.title(f'Cosine schedule (T={self.T})')
        plt.show()
    
if __name__ == "__main__":
    # Test the linear schedule
    linear_schedule = LinearSchedule(1e-4, 1, 1000)
    linear_schedule.plot_schedule()

    # Test the cosine schedule
    cosine_schedule = CosineSchedule(1000)
    cosine_schedule.plot_schedule()
