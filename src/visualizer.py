import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from typing import List

from diffusion_model import DiffusionModel

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Visualizer:
    def __init__(self):
        pass

    def plot_loss(self, loss: List[float], 
                  save_path: str = os.path.join(PROJECT_BASE_DIR,'results','plots','training'),
                  fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Loss-DiffusionModel.png"):
        '''
        Plot the loss over training iterations

        Inputs:
        - loss: List of loss values
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # create plot
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(loss, color='blue')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss of DDPM')
        ax.set_xticks(np.arange(0, len(loss), step=10))
        ax.grid(True)

        # save plot
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_is_fid_score(self, is_score: List[float], fid_score: List[float], 
                          save_path: str = os.path.join(PROJECT_BASE_DIR,'results','plots','evaluation'),
                          fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-IS-FID-DiffusionModel.png"):
        '''
        Plot the inception score and FID score over evaluation iterations

        Inputs:
        - is_score: List of inception scores
        - fid_score: List of FID scores
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # create plot
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(is_score, color='blue', label='Inception Score')
        ax.plot(fid_score, color='red', label='FID Score')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_title('Inception Score and FID Score of DDPM')
        ax.set_xticks(np.arange(0, len(is_score), step=10))
        ax.legend()
        ax.grid(True)

        # save plot
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_single_image(self, image: np.ndarray, 
                          save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','generated'),
                          fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Generated-Image.png",
                          title: str = 'Generated Image'):
        '''
        Plot a single image

        Inputs:
        - image: Image to plot
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # create plot
        fig, ax = plt.subplots(figsize=(10,6))
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

        # save plot
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_reconstructed_image(self, original_image: np.ndarray, reconstructed_image: np.ndarray,
                                 save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','reconstructed'),
                                 fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Reconstructed-Image.png",
                                 title: str = 'Reconstructed Image'):
        '''
        Plot two images side by side.
        Original image on the left and reconstructed image on the right.

        Inputs:
        - original_image: Original image
        - reconstructed_image: Reconstructed image
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # create plot
        fig, ax = plt.subplots(1, 2, figsize=(10,6))
        ax[0,0].imshow(original_image)
        ax[0,0].set_title('Original Image')
        ax[0,0].axis('off')
        ax[0,1].imshow(reconstructed_image)
        ax[0,1].set_title('Reconstructed Image')
        ax[0,1].axis('off')
        fig.suptitle(title)

        # save plot
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_multiple_images(self, images: List[np.ndarray], 
                             save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','generated'),
                             fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Generated-Images.png",
                             title: str = 'Generated Images'):
        '''
        Plot multiple images in a grid

        Inputs:
        - images: List of images to plot (should be less than 16)
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # preprocessing for plot
        if len(images) >= 16:
            images = images[:16]
            print('Only the first 16 images will be plotted')
        if len(images) < 4:
            num_rows = 1
        else:
            num_rows = len(images) // 4

        # create plot
        fig, ax = plt.subplots(num_rows, 4, figsize=(10,10))
        for i in range(num_rows):
            for j in range(4):
                if i*4+j < len(images):
                    ax[i,j].imshow(images[i*4+j])
                    ax[i,j].axis('off')
        fig.suptitle(title)

        # save plot
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_reconstructed_images(self, original_images: List[np.ndarray], reconstructed_images: List[np.ndarray],
                                  save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','reconstructed'),
                                  fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Reconstructed-Images.png",
                                  title: str = 'Reconstructed Images'):
        '''
        Plot multiple pairs of images side by side.
        Original images on the left and reconstructed images on the right.

        Inputs:
        - original_images: List of original images
        - reconstructed_images: List of reconstructed images
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # preprocessing for plot
        assert len(original_images) == len(reconstructed_images), 'Number of original and reconstructed images should be the same'
        if len(original_images) >= 16:
            original_images = original_images[:16]
            reconstructed_images = reconstructed_images[:16]
            print('Only the first 16 images will be plotted')
        if len(original_images) < 4:
            num_rows = 1
        else:
            num_rows = len(original_images) // 4

        # create plot
        fig, ax = plt.subplots(num_rows, 4*2, figsize=(10,10))
        for i in range(num_rows):
            for j in range(8):
                if i*4+(j//2) < len(original_images):
                    if j % 2 == 0:
                        ax[i,j].imshow(original_images[i*4+(j//2)])
                        ax[i,j].axis('off')
                    else:
                        ax[i,j].imshow(reconstructed_images[i*4+(j//2)])
                        ax[i,j].axis('off')
        fig.suptitle(title)

        # save plot
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_forward_process(self, diffusion_model: DiffusionModel, x: torch.Tensor, t: List[int], 
                             save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','forward_process'),
                             fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Forward-Process.png",
                             title: str = 'Forward Process', cmap: str = 'gray'):
        '''
        Plot the forward process of the diffusion model

        Inputs:
        - diffusion_model: Diffusion model
        - x: Single image as tensor [1, C, H, W]
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # create plot
        fig, ax = plt.subplots(1, len(t), figsize=(10,4))
        for idx in range(len(t)):
            x_t = diffusion_model.forward(x, t[idx])
            ax[idx].imshow(x_t[0].squeeze().detach().cpu().numpy(), cmap=cmap)
            ax[idx].axis('off')
            ax[idx].set_title(f't={t[idx]}')
        fig.suptitle(title)

        # save plot
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_reverse_process(self, diffusion_model: DiffusionModel, t: List[int], 
                             save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','reverse_process'),
                             fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Reverse-Process.png",
                             title: str = 'Reverse Process'):
        '''
        Plot the reverse process of the diffusion model

        Inputs:
        - diffusion_model: Diffusion model
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''

        # create plot
        fig, ax = plt.subplots(1, len(t), figsize=(10,6))
        for idx in range(len(t)):
            x_t = diffusion_model.sample(n_samples=1, t=t[idx])
            ax[idx].imshow(x_t.squeeze().detach().cpu().numpy())
            ax[idx].axis('off')
            ax[idx].set_title(f't={t[idx]}')
        fig.suptitle(title)

        # save plot
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()