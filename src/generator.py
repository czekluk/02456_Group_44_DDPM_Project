import os
import torch
import argparse
import sys
from torchvision import datasets, transforms
import numpy as np

from unet import SimpleModel
from diffusion_model import DiffusionModel
from visualizer import Visualizer
from dataset import DiffusionDataModule
from schedule import LinearSchedule, CosineSchedule
from mnist_guidance import MNISTGuidanceClassifier
from cifar10_guidance import CIFAR10GuidanceClassifier

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
                self.visualizer.plot_single_image(samples)

        return samples
    
    def generate_guided(self, num_samples: int = 1, plot: bool = False, class_label: int = 0):
        '''
        Method to generate samples using the diffusion model with classifier guidance

        Inputs:
        - num_samples: Number of samples to generate
        - class_label: label of class to plot

        Returns:
        - samples: Generated samples as a tensor with shape [num_samples, C, H, W]
        '''
        samples = self.diffusion_model.guided_sample(n_samples=num_samples, class_label=class_label)

        if plot:
            if num_samples >= 2:
                self.visualizer.plot_multiple_images(samples)
            else:
                self.visualizer.plot_single_image(samples)

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
                self.visualizer.plot_reconstructed_image(x, recon_x)

        return recon_x
    
def main(args):
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument("--data_type", type=str, choices=["mnist", "cifar10"], required=True, help="Dataset to use: 'mnist' or 'cifar10'")
    parser.add_argument("--schedule", type=str, choices=["linear", "cosine"], required=True, help="Schedule type: 'linear' or 'cosine'")
    parser.add_argument("--attention", type=str, choices=["attention", "noattention"], required=True, help="Attention type: 'attention' or 'noattention'")
    parser.add_argument("--model", type=str,required=True, help="Model path: relative path to the trained UNet to use for sampling")
    parser.add_argument("--guided_class", type=int, required=False, help="Guided class: Class to use for guided sampling")

    args = parser.parse_args(args)

    DATA_FLAG = args.data_type
    SCHEDULE_FLAG = args.schedule
    ATTENTION_FLAG = args.attention
    MODEL_PATH = args.model
    GUIDED_CLASS = args.guided_class

    # Initialize diffusion model
    T = 1000
    if DATA_FLAG == "cifar10":
        if ATTENTION_FLAG=="attention":
            model = SimpleModel(ch_layer0=32, out_ch=3, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
        elif ATTENTION_FLAG=="noattention":
            model = SimpleModel(ch_layer0=32, out_ch=3, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True)
        
        if SCHEDULE_FLAG == "linear":
            schedule = LinearSchedule(10e-4, 0.02, T)
        elif SCHEDULE_FLAG == "cosine":
            schedule = CosineSchedule(T)

        classifier = CIFAR10GuidanceClassifier()
        classifier.load_state_dict(torch.load(os.path.join(PROJECT_BASE_DIR,'resources','models','guidance','cifar10_guidance_classifier.pth'), weights_only=True))

        diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(3, 32, 32), classifier=classifier, lambda_guidance=100)
        gen = Generator(diffusion_model, os.path.join(PROJECT_BASE_DIR, MODEL_PATH))

    elif DATA_FLAG == "mnist":
        if ATTENTION_FLAG=="attention":
            model = SimpleModel(ch_layer0=32, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
        elif ATTENTION_FLAG=="noattention":
            model = SimpleModel(ch_layer0=32, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True)
        
        if SCHEDULE_FLAG == "linear":
            schedule = LinearSchedule(10e-4, 0.02, T)
        elif SCHEDULE_FLAG == "cosine":
            schedule = CosineSchedule(T)

        classifier = MNISTGuidanceClassifier()
        classifier.load_state_dict(torch.load(os.path.join(PROJECT_BASE_DIR,'resources','models','guidance','mnist_guidance_classifier.pth'), weights_only=True))

        diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(1, 28, 28))
        gen = Generator(diffusion_model, os.path.join(PROJECT_BASE_DIR, MODEL_PATH))

    else:
        raise NotImplementedError
    
    samples = gen.generate(num_samples=16, plot=True)

    all_samples = gen.generate_all_steps(num_samples=1, plot=True, plot_steps=[0, 250, 500, 750, 1000])

    data_module = DiffusionDataModule()
    if DATA_FLAG == "mnist":
        test_loader = data_module.get_MNIST_dataloader(
            train=False,
            batch_size=16,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))    
            ])
        )
    elif DATA_FLAG == "cifar10":
        test_loader = data_module.get_CIFAR10_dataloader(
            train=False,
            batch_size=16,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])    
            ])
        )
    else:
        raise NotImplementedError

    x, _ = next(iter(val_loader))
    recon_x = gen.reconstruct(x, plot=True)

    if GUIDED_CLASS is not None:
        gen.generate_guided(num_samples=1, plot=True, class_label=GUIDED_CLASS)

if __name__ == "__main__":
    main(sys.argv[1:])
