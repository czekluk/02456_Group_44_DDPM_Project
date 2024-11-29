import torch
import os

from datetime import datetime

from schedule import LinearSchedule, CosineSchedule
from objective import NoiseObjective

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPU_DETACH = False
class DiffusionModel:
    def __init__(self, model: torch.nn.Module, T: int = 1000, schedule = LinearSchedule(10e-4, 0.02, 1000), img_shape: tuple = (1, 28, 28), classifier: torch.nn.Module = None, lambda_guidance: float = 0.1):
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
        self.schedule = schedule
        self.img_shape = img_shape

        # Training related parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = NoiseObjective()
        self.model.to(self.device)

        # Sampling related parameters

        # Classifier guidance related parameters
        self.classifier = classifier
        if self.classifier is not None:
            self.classifier.to(self.device)
        self.lambda_guidance = lambda_guidance

    def train(self, x: torch.Tensor, optimizer: torch.optim.Optimizer, verbose: bool = False):
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
        e_pred = self.model(x, t.squeeze())

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
    
    def val_loss(self, x: torch.Tensor):
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
            e_pred = self.model(x, t.squeeze())

            # calculate loss
            loss = self.criterion(e, e_pred)

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
        mean = torch.sqrt(torch.tensor([self.schedule.alpha_dash(t)])) * x

        # calculate std of forward sampling process
        identity = torch.ones(x.shape[2], x.shape[3])
        identity = identity.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        std = (1-torch.Tensor([self.schedule.alpha_dash(t)])) * identity

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
        x = x.to(self.device)
        noise_pred = self.model(x, t_tensor)

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
        return x_t.cpu().detach().numpy()
    
    def guided_sample(self, n_samples=10, t=0, class_label=0):
        '''
        Sampling operation of diffusion model guided by a classifier.

        Inputs:
        - n_samples: Number of samples to generate (batch size)
        - t: Sample a backward process at time t
        - class_label: Class label to guide the diffusion process

        Returns:
        - samples: Generated samples as a tensor with shape [n_samples, C, H, W]
        '''
        assert self.classifier is not None, 'Classifier is not defined for guided sampling'

        # Step 1: Initialize with Gaussian noise with mean 0 and variance 1
        C, H, W = self.img_shape  # Assume img_shape is defined in the model as (channels, height, width)
        x_t = torch.randn((n_samples, C, H, W), device=self.device)  # Starting with pure noise

        # Step 2: Loop through timesteps in reverse
        for t in reversed(range(t, self.T)):  # Assumes num_timesteps is defined
            x_t = self.guided_backward(x_t, t, class_label)

        # Step 3: Return the batch of generated samples
        return x_t.cpu().detach().numpy()
    
    def guided_backward(self, x: torch.Tensor, t: int, class_label: int):
        '''
        Reverse process of the diffusion model.

        Inputs:
        - x: Noisy image at timestep t [B, C, H, W]
        - t: Current timestep in the reverse process
        - class_label: Class label to guide the diffusion process

        Returns:
        - x_t_minus_1: Denoised image at timestep t-1
        '''
        # Predict the noise in the image at timestep t
        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.int64)
        x = x.to(self.device)
        noise_pred = self.model(x, t_tensor)

        # claculate gradient for the classifier guidance part
        x.requires_grad_()
        class_probs = torch.nn.Softmax(dim=1)(self.classifier(x))
        class_loss = torch.log(class_probs[:, class_label] + 1e-8)  # Avoid log(0)
        class_loss.mean().backward()
        gradient = x.grad.data
        if CPU_DETACH:
            gradient = gradient.cpu().detach()

            # gradient = torch.zeros_like(x).cpu().detach()
            # for idx in range(class_loss.shape[0]):
            #     class_loss[idx].backward(retain_graph=True)
            #     gradient[idx] = x.grad.data[idx]
            #     x.grad.data.zero_()
            # gradient = gradient.cpu().detach()

            # Retrieve alpha_t and beta_t from the schedule
            alpha_dash_t = self.schedule.alpha_dash(t).cpu().detach()
            alpha_t = self.schedule.alpha(t).cpu().detach()
            beta_t = self.schedule.beta(t).cpu().detach()  # Variance for timestep t

            noise_pred = noise_pred.cpu().detach()
            x = x.cpu().detach()
        else:
            gradient = gradient.detach()

            # gradient = torch.zeros_like(x).cpu().detach()
            # for idx in range(class_loss.shape[0]):
            #     class_loss[idx].backward(retain_graph=True)
            #     gradient[idx] = x.grad.data[idx]
            #     x.grad.data.zero_()
            # gradient = gradient.cpu().detach()

            # Retrieve alpha_t and beta_t from the schedule
            alpha_dash_t = self.schedule.alpha_dash(t).detach()
            alpha_t = self.schedule.alpha(t).detach()
            beta_t = self.schedule.beta(t).detach()  # Variance for timestep t
            noise_pred = noise_pred.detach()
            x = x.detach()
        # apply classifier guidance
        noise_pred = noise_pred - self.lambda_guidance * gradient

        # Compute the mean for x_t_minus_1 using the noise prediction
        mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t)/torch.sqrt(1 - alpha_dash_t)) * noise_pred)
        
        # Sample noise using beta_t as the variance for the current timestep
        if t > 1:
            if CPU_DETACH:
                noise = torch.randn_like(x).cpu()
            else:
                noise = torch.randn_like(x)
            x_t_minus_1 = mean + torch.sqrt(beta_t) * noise
        else:
            x_t_minus_1 = mean  # No noise at the final step

        return x_t_minus_1
    
    def sample_from_noise(self, x_t: torch.Tensor, t: int):
        '''
        Generate samples from the noise tensor at timestep t.

        Inputs:
        - noise: Noise tensor at timestep t [B, C, H, W]
        - t: Timestep at which to sample the image

        Returns:
        - samples: Generated samples as a tensor with shape [B, C, H, W]
        '''
        # Step 2: Loop through timesteps in reverse
        for t in reversed(range(t, self.T)):  # Assumes num_timesteps is defined
            x_t = self.backward(x_t, t)

        # Step 3: Return the batch of generated samples
        return x_t.cpu().detach().numpy()
    
    def all_step_sample(self, n_samples=10):
        '''
        Sampling operation of the diffusion model.

        Inputs:
        - n_samples: Number of samples to generate (batch size)

        Returns:
        - samples: List of generated samples of shape [n_samples, C, H, W] corresponding to all time steps
        '''
        # Step 1: Initialize with Gaussian noise with mean 0 and variance 1
        C, H, W = self.img_shape
        x_t = torch.randn((n_samples, C, H, W), device=self.device)  # Starting with pure noise
        x_ts = [x_t.detach().cpu().numpy()]

        # Step 2: Loop through timesteps in reverse
        for t in reversed(range(0, self.T)):  # Assumes num_timesteps is defined
            x_t = self.backward(x_t, t)
            x_ts.append(x_t.detach().cpu().numpy())

        # Step 3: Return the list of generated samples
        return x_ts
    
    def save(self, path: str = os.path.join(PROJECT_BASE_DIR, 'results', 'models'),
             model_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-DiffusionModel.pth"):
        '''
        Save the model to a file.

        Inputs:
        - path: Path where to save the model
        - model_name: Name of the model file
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, model_name))
        print(f'Model saved to {os.path.join(path, model_name)}')

    def load(self, path: str):
        '''
        Load a diffusion model from file
        
        Inputs:
        - path: Path to the model file
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print(f'Model loaded from {path}')

    