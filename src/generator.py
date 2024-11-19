import os
import torch
from torchvision import datasets, transforms

from model import Model
from diffusion_model import DiffusionModel
from visualizer import Visualizer
from dataset import DiffusionDataModule

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Generator:
    def __init__(self, model: DiffusionModel, model_path: str):
        self.diffusion_model = model
        self.diffusion_model.load(model_path)
        self.visualizer = Visualizer()

    def generate(self, num_samples: int = 10, plot: bool = False):
        '''
        Method to generate samples using the diffusion model.

        Inputs:
        - num_samples: Number of samples to generate

        Returns:
        - samples: Generated samples as a tensor with shape [num_samples, C, H, W]
        '''
        samples = self.diffusion_model.sample(n_samples=num_samples)

        if plot:
            if num_samples >= 2:
                self.visualizer.plot_multiple_images(samples)
            else:
                self.visualizer.plot_single_image(samples[0].squeeze())

        return samples
    
    def generate_all_steps(self, num_samples: int = 10, plot: bool = False, plot_steps: list = [0, 250, 500, 750, 1000]):
        '''
        Method to generate samples using the diffusion model at all time steps.

        Inputs:
        - num_samples: Number of samples to generate

        Returns:
        - samples: List of generated samples of shape [num_samples, C, H, W] corresponding to all time steps
        '''
        samples = self.diffusion_model.all_step_sample(n_samples=num_samples)

        if plot:
            self.visualizer.plot_reverse_process(samples, plot_steps)

        return samples
    
    def reconstruct(self, x: torch.Tensor, plot: bool = False):
        '''
        Method to reconstruct the input image using the diffusion model.

        Inputs:
        - x: Input images to be reconstructed in shape [B, C, H, W]

        Returns:
        - recon_x: Reconstructed image
        '''
        xT = self.diffusion_model.forward(x, t=self.diffusion_model.T)
        recon_x = self.diffusion_model.sample_from_noise(x_t=xT, t=0)

        if plot:
            if x.shape[0] >= 2:
                self.visualizer.plot_reconstructed_images(x, recon_x)
            else:
                self.visualizer.plot_reconstructed_image(x.squeeze(), recon_x.squeeze())

        return recon_x

if __name__ == "__main__":
    model = Model(ch=64, out_ch=1, ch_down_mult=(1, 2), num_res_blocks=2, attn_resolutions=[7], dropout=0.1, resamp_with_conv=True)
    gen = Generator(DiffusionModel(model, T=1000),
                    os.path.join(PROJECT_BASE_DIR, 'results/models/2024-11-16_21-08-13-Epoch_0004-FID_5.76-DiffusionModel.pth'))
    
    samples = gen.generate(num_samples=16, plot=True)
    all_samples = gen.generate_all_steps(num_samples=1, plot=True)

    data_module = DiffusionDataModule()
    val_loader = data_module.get_MNIST_dataloader(
        train=False,
        batch_size=16,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    x, _ = next(iter(val_loader))
    recon_x = gen.reconstruct(x, plot=True)