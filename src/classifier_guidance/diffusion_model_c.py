import torch
import os
import sys
PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_BASE_DIR, 'src')
sys.path.append(SRC_DIR)
from schedule import LinearSchedule
from diffusion_model import DiffusionModel

CPU_DETACH = False
class DiffClassifierGuidance(DiffusionModel):
    def __init__(self, model: torch.nn.Module, T: int = 1000, schedule = LinearSchedule(10e-4, 0.02, 1000), img_shape: tuple = (1, 28, 28), classifier: torch.nn.Module = None, lambda_guidance: float = 0.1):
        super().__init__(model=model, T=T, schedule = schedule, img_shape= img_shape)
        self.classifier = classifier
        if self.classifier is not None:
            self.classifier.to(self.device)
        self.lambda_guidance = lambda_guidance

    def sample(self, n_samples=10, t=0, class_label=0):
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
            x_t = self.backward(x_t, t, class_label)

        # Step 3: Return the batch of generated samples
        return x_t.cpu().detach().numpy()
    
    def backward(self, x: torch.Tensor, t: int, class_label: int):
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