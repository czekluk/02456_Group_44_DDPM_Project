import torch
import os
import sys
PROJECT_BASE_DIR =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_BASE_DIR, 'src')
sys.path.append(SRC_DIR)
from schedule import LinearSchedule
from diffusion_model import DiffusionModel
CPU_DETACH = False
class DiffClassifierFreeGuidance(DiffusionModel):
    def __init__(self, model: torch.nn.Module, T: int = 1000, schedule = LinearSchedule(10e-4, 0.02, 1000), img_shape: tuple = (1, 28, 28)):
        super().__init__(model=model, T=T, schedule = schedule, img_shape= img_shape)
    
    def train(self, x: torch.Tensor, optimizer: torch.optim.Optimizer, verbose: bool = False, class_label: int = None):
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
        t = torch.randint(1, self.T, (x.shape[0], 1), device=self.device)

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
        e_pred = self.model(x, t.squeeze(), class_label)


        # calculate loss
        loss = self.criterion(e, e_pred)

        # take gradient descent step
        loss.backward()
        optimizer.step()
        
        # print loss
        if verbose:
            print(f'Loss: {loss.item()}')
        
        # return loss (for logging purposes)
        return loss.item()
    
    def sample(self, n_samples=10, t=0, class_label: torch.Tensor= torch.Tensor([0])):
        '''
        Inputs:
        - n_samples: Number of samples to generate (batch size)
        - t: Sample a backward process at time t
        - class_label: Class label to guide the diffusion process

        Returns:
        - samples: Generated samples as a tensor with shape [n_samples, C, H, W]
        '''
        assert isinstance(class_label, torch.Tensor), "class_label must be a torch.Tensor"
        # Step 1: Initialize with Gaussian noise with mean 0 and variance 1
        C, H, W = self.img_shape  # Assume img_shape is defined in the model as (channels, height, width)
        x_t = torch.randn((n_samples, C, H, W), device=self.device)  # Starting with pure noise

        # Step 2: Loop through timesteps in reverse
        for t in reversed(range(t, self.T)): 
            x_t = self.backward(x_t, t, class_label)

        # Step 3: Return the batch of generated samples
        return x_t.cpu().detach().numpy()
    
    def backward(self, x: torch.Tensor, t: int, class_label: torch.Tensor):
        '''
        Reverse process of the diffusion model.

        Inputs:
        - x: Noisy image at timestep t [B, C, H, W]
        - t: Current timestep in the reverse process

        Returns:
        - x_t_minus_1: Denoised image at timestep t-1
        '''
        assert isinstance(class_label, torch.Tensor), "class_label must be a torch.Tensor"

        # Predict the noise in the image at timestep t
        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.int64)
        x = x.to(self.device)
        noise_pred = self.model(x, t_tensor, class_label)

        # Retrieve alpha_t and beta_t from the schedule
        if CPU_DETACH:
            alpha_dash_t = torch.tensor(self.schedule.alpha_dash(t)).cpu().detach()
            alpha_t = torch.tensor(self.schedule.alpha(t)).cpu().detach()
            beta_t = torch.tensor(self.schedule.beta(t)).cpu().detach()  # Variance for timestep t
            noise_pred = noise_pred.cpu().detach()
            x = x.cpu().detach()
        else:
            alpha_dash_t = torch.tensor(self.schedule.alpha_dash(t)).detach()
            alpha_t = torch.tensor(self.schedule.alpha(t)).detach()
            beta_t = torch.tensor(self.schedule.beta(t)).detach()  # Variance for timestep t
            noise_pred = noise_pred.detach()
            x = x.detach()

        # Compute the mean for x_t_minus_1 using the noise prediction
        mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t)/torch.sqrt(1 - alpha_dash_t)) * noise_pred)
        
        # Sample noise using beta_t as the variance for the current timestep
        if t > 1:
            if CPU_DETACH:
                noise = torch.randn_like(x).cpu() # Standard Gaussian noise
            else:
                noise = torch.randn_like(x)
            x_t_minus_1 = mean + torch.sqrt(beta_t) * noise
        else:
            x_t_minus_1 = mean  # No noise at the final step

        return x_t_minus_1
    def val_loss(self, x: torch.Tensor, label:torch.Tensor):
        '''
        Calculate the validation loss of the model.

        Inputs:
        - x: Batch of validation images [B, C, H, W]

        Returns:
        - loss: Validation loss of the model
        '''
        self.model.eval()
        with torch.no_grad():
            # sample t from uniform distribution
            t = torch.randint(1, self.T, (x.shape[0], 1), device=self.device)

            # sample e from N(0,I)
            e = self.normal.rsample(sample_shape=x.shape).to(self.device)

            # calculate alpha_t for every batch image
            ats = self.schedule.alpha_dash_list(t.squeeze().tolist()).to(self.device)
            # ats is of shape [batch_size, 1], expand it to match the shape of x (which is [batch_size, C, H, W])
            ats = ats.view(-1, 1, 1, 1)
            ats = ats.expand(-1, x.shape[1], x.shape[2], x.shape[3])  # expand to (batch_size, C, H, W)

            # calculate model inputs
            x = x.to(self.device)
            t = t.to(self.device)

            x = torch.sqrt(ats) * x + torch.sqrt(1- ats) * e

            # calculate model outputs
            e_pred = self.model(x, t.squeeze(), label)

            # calculate loss
            loss = self.criterion(e, e_pred)

            # return loss (for logging purposes)
            return loss.item()