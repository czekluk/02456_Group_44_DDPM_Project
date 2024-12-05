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
SRC_DIR= os.path.dirname(os.path.abspath(__file__))
CLASS_GUIDANCE_DIR = os.path.join(SRC_DIR, 'classifier_guidance')
sys.path.append(CLASS_GUIDANCE_DIR)
from classifier_free_guidance.diffusion_model_cf import DiffClassifierFreeGuidance
from classifier_free_guidance.unet_cf import SimpleModelClassFreeGuidance
from classifier_guidance.diffusion_model_c import DiffClassifierGuidance
from classifier_guidance.test_mnist import MNISTGuidanceClassifier
from classifier_guidance.test_cifar10 import CIFAR10GuidanceClassifier

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
        samples = self.diffusion_model.sample(n_samples=num_samples, class_label=class_label)

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
    parser.add_argument("-mode", type=str, choices=["default", "guided_classifier", "guided_free"], required=True, help="Model mode: 'default' or 'guided_classifier' or 'guided_free'")
    parser.add_argument("--data_type", type=str, choices=["mnist", "cifar10"], required=True, help="Dataset to use: 'mnist' or 'cifar10'")
    parser.add_argument("--schedule", type=str, choices=["linear", "cosine"], required=True, help="Schedule type: 'linear' or 'cosine'")
    parser.add_argument("--attention", type=str, choices=["attention", "noattention"], required=True, help="Attention type: 'attention' or 'noattention'")
    parser.add_argument("--model", type=str,required=True, help="Model path: relative path to the trained UNet to use for sampling")
    parser.add_argument("--guided_class", type=int, required=False, help="Guided class: Class to use for guided sampling")

    args = parser.parse_args(args)

    MODE = args.mode
    DATA_FLAG = args.data_type
    SCHEDULE_FLAG = args.schedule
    ATTENTION_FLAG = args.attention
    MODEL_PATH = args.model
    GUIDED_CLASS = args.guided_class

    # Initialize diffusion model
    T = 1000
    if DATA_FLAG == "cifar10":
        if ATTENTION_FLAG=="attention":
            if MODE == "default" or "guided_classifier":
                model = SimpleModel(ch_layer0=64, out_ch=3, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
            elif MODE == "guided_free":
                model = SimpleModelClassFreeGuidance(ch_layer0=64, out_ch=3, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True, lambda_cf=10.)
            else:
                raise NotImplementedError
        elif ATTENTION_FLAG=="noattention":
            if MODE == "default" or "guided_classifier":
                model = SimpleModel(ch_layer0=64, out_ch=3, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True)
            elif MODE == "guided_free":
                model = SimpleModelClassFreeGuidance(ch_layer0=64, out_ch=3, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True, lambda_cf=10.)
            else:
                raise NotImplementedError
            
        if SCHEDULE_FLAG == "linear":
            schedule = LinearSchedule(10e-4, 0.02, T)
        elif SCHEDULE_FLAG == "cosine":
            schedule = CosineSchedule(T)

        if MODE == "guided_classifier":
            classifier = CIFAR10GuidanceClassifier()
            classifier.load_state_dict(torch.load(os.path.join(PROJECT_BASE_DIR,'resources','models','guidance','cifar10_guidance_classifier.pth'), weights_only=True))

            diffusion_model = DiffClassifierGuidance(model, T=T, schedule=schedule, img_shape=(3, 32, 32), classifier=classifier, lambda_guidance=100)
        elif MODE == "default":
            diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(3, 32, 32))
        elif MODE == "guided_free":
            diffusion_model = DiffClassifierFreeGuidance(model, T=T, schedule=schedule, img_shape=(3, 32, 32))
        else:
            raise NotImplementedError

        gen = Generator(diffusion_model, os.path.join(PROJECT_BASE_DIR, MODEL_PATH))

    elif DATA_FLAG == "mnist":
        if ATTENTION_FLAG=="attention":
            if MODE == "default" or "guided_classifier":
                model = SimpleModel(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
            elif MODE == "guided_free":
                model = SimpleModelClassFreeGuidance(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True, lambda_cf=10.)
            else:
                raise NotImplementedError
        elif ATTENTION_FLAG=="noattention":
            if MODE == "default" or "guided_classifier":
                model = SimpleModel(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True)
            elif MODE == "guided_free":
                model = SimpleModelClassFreeGuidance(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True, lambda_cf=10.)
            else:
                raise NotImplementedError
        
        if SCHEDULE_FLAG == "linear":
            schedule = LinearSchedule(10e-4, 0.02, T)
        elif SCHEDULE_FLAG == "cosine":
            schedule = CosineSchedule(T)

        if MODE == "guided_classifier":
            classifier = MNISTGuidanceClassifier()
            classifier.load_state_dict(torch.load(os.path.join(PROJECT_BASE_DIR,'resources','models','guidance','mnist_guidance_classifier.pth'), weights_only=True))

            diffusion_model = DiffClassifierGuidance(model, T=T, schedule=schedule, img_shape=(1, 28, 28), classifier=classifier, lambda_guidance=100)
        elif MODE == "default":
            diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(1, 28, 28))
        elif MODE == "guided_free":
            diffusion_model = DiffClassifierFreeGuidance(model, T=T, schedule=schedule, img_shape=(1, 28, 28))
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    
    if MODE == "default":
        samples = gen.generate(num_samples=16, plot=True)

        all_samples = gen.generate_all_steps(num_samples=1, plot=True, plot_steps=[0, 250, 500, 750, 1000])

        data_module = DiffusionDataModule()
        if DATA_FLAG == "cifar10":
            train_loader, val_loader, test_loader = data_module.get_CIFAR10_data_split(
                batch_size=16,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
            )
        elif DATA_FLAG == "mnist":
            train_loader, val_loader, test_loader = data_module.get_MNIST_data_split(
                batch_size=16,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )
        else:
            raise NotImplementedError

        x, _ = next(iter(test_loader))
        recon_x = gen.reconstruct(x, plot=True)

    if MODE == "guided_classifier" or "guided_free":
        if GUIDED_CLASS is not None:
            gen.generate_guided(num_samples=12, plot=True, class_label=GUIDED_CLASS)

if __name__ == "__main__":
    main(sys.argv[1:])
