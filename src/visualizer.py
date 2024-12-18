import torch
import os

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from typing import List

from diffusion_model import DiffusionModel

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Visualizer:
    def __init__(self):
        pass

    def plot_loss(self, loss: List[float], val_loss: List[float], loss_conf: List[tuple], val_loss_conf: List[tuple], conf: bool = False,
                  save_path: str = os.path.join(PROJECT_BASE_DIR,'results','plots','training'),
                  fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Loss-DiffusionModel.png"):
        '''
        Plot the loss over training iterations

        Inputs:
        - loss: List of loss values
        - val_loss: List of validation loss values
        - loss_conf: List of confidence intervals for loss
        - val_loss_conf: List of confidence intervals for val_loss
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # extract lower & upper bounds
        lower, upper = [], []
        for conf_ in loss_conf:
            lower.append(conf_[0])
            upper.append(conf_[1])
        val_lower, val_upper = [], []
        for conf_ in val_loss_conf:
            val_lower.append(conf_[0])
            val_upper.append(conf_[1])

        # create plot
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(loss, color='blue', label='Train Loss')
        if conf:
            ax.fill_between(range(len(loss)), lower, upper, color='blue', alpha=0.1)
        ax.plot(val_loss, color='red', label='Validation Loss')
        if conf:
            ax.fill_between(range(len(val_loss)), val_lower, val_upper, color='red', alpha=0.1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss of DDPM')
        ax.set_xticks(np.arange(0, len(loss), step=10))
        ax.grid(True)
        plt.legend()

        # save plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
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
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_fid_score(self, fid_score: List[float], fid_conf: List[tuple],
                          save_path: str = os.path.join(PROJECT_BASE_DIR,'results','plots','evaluation'),
                          fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-FID-DiffusionModel.png"):
        '''
        Plot the inception score and FID score over evaluation iterations

        Inputs:
        - fid_score: List of FID scores
        - fid_conf: List of confidence intervals for fid score
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # extract lower & upper bounds
        lower, upper = [], []
        for conf in fid_conf:
            lower.append(conf[0])
            upper.append(conf[1])

        # create plot
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(fid_score, color='red', label='FID Score')
        ax.fill_between(range(len(fid_score)), lower, upper, color='red', alpha=0.1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_title(' Approximated FID Score of DDPM on Validation Set')
        ax.set_xticks(np.arange(0, len(fid_score), step=10))
        ax.legend()
        ax.grid(True)

        # save plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def denormalize(self, image: np.ndarray):
        '''
        Denormalize image from [-1, 1] to [0, 1]
        '''
        return (image + 1) / 2

    def plot_single_image(self, image: np.ndarray, 
                          save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','generated'),
                          fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Generated-Image.png",
                          title: str = 'Generated Image', cmap: str = 'gray', denormalize: bool = True):
        '''
        Plot a single image

        Inputs:
        - image: Image to plot in shape [B, C, H, W]
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # create plot
        fig, ax = plt.subplots(figsize=(10,6))
        if denormalize:
            image = self.denormalize(image)

        image = np.clip(image, 0, 1)
        image = np.transpose(image[0],(1,2,0))
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

        # save plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_reconstructed_image(self, original_image: np.ndarray, reconstructed_image: np.ndarray,
                                 save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','reconstructed'),
                                 fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Reconstructed-Image.png",
                                 title: str = 'Reconstructed Image', cmap: str = 'gray', denormalize: bool = True):
        '''
        Plot two images side by side.
        Original image on the left and reconstructed image on the right.

        Inputs:
        - original_image: Original image in shape [B, C, H, W]
        - reconstructed_image: Reconstructed image
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        if denormalize:
            original_image = self.denormalize(original_image)
            reconstructed_image = self.denormalize(reconstructed_image)
        original_image = np.transpose(original_image[0],(1,2,0))
        reconstructed_image = np.transpose(reconstructed_image[0],(1,2,0))

        reconstructed_image = np.clip(reconstructed_image, 0, 1)

        # create plot
        fig, ax = plt.subplots(1, 2, figsize=(10,6))
        ax[0].imshow(original_image, cmap=cmap)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(reconstructed_image, cmap=cmap)
        ax[1].set_title('Reconstructed Image')
        ax[1].axis('off')
        fig.suptitle(title)

        # save plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_multiple_images(self, images: np.ndarray, 
                             save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','generated'),
                             fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Generated-Images.png",
                             title: str = 'Generated Images', cmap: str = 'gray', denormalize: bool = True):
        '''
        Plot multiple images in a grid

        Inputs:
        - images: numpy array of shape [B, C, H, W]. Shall be at least 5 images
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # preprocessing for plot
        if images.shape[0] > 16:
            images = images[:16]
            print('Only the first 16 images will be plotted')
        num_rows = int(np.ceil(images.shape[0] / 4))

        if denormalize:
            images = self.denormalize(images)

        images = np.clip(images, 0, 1)

        # create plot
        fig, ax = plt.subplots(num_rows, 4, figsize=(10,10))
        for i in range(num_rows):
            for j in range(4):
                if num_rows > 1:
                    if i*4+j < images.shape[0]:
                        img = np.transpose(images[i*4+j],(1,2,0))
                        ax[i,j].imshow(img, cmap=cmap)
                    ax[i,j].axis('off')
                else:
                    if j < images.shape[0]:
                        img = np.transpose(images[j],(1,2,0))
                        ax[j].imshow(img, cmap=cmap)
                    ax[j].axis('off')
        fig.suptitle(title)

        # save plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_reconstructed_images(self, original_images: List[np.ndarray], reconstructed_images: List[np.ndarray],
                                  save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','reconstructed'),
                                  fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Reconstructed-Images.png",
                                  title: str = 'Reconstructed Images', cmap: str = 'gray', denormalize: bool = True):
        '''
        Plot multiple pairs of images side by side.
        Original images on the left and reconstructed images on the right.

        Inputs:
        - original_images: original images in shape [B, C, H, W]
        - reconstructed_images: reconstructed images in shape [B, C, H, W]
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        # preprocessing for plot
        assert len(original_images) == len(reconstructed_images), 'Number of original and reconstructed images should be the same'
        if original_images.shape[0] > 16:
            original_images = original_images[:16]
            reconstructed_images = reconstructed_images[:16]
            print('Only the first 16 images will be plotted')

        if denormalize:
            original_images = self.denormalize(original_images)
            reconstructed_images = self.denormalize(reconstructed_images)

        original_images = np.clip(original_images, 0, 1)
        reconstructed_images = np.clip(reconstructed_images, 0, 1)

        num_rows = int(np.ceil(original_images.shape[0] / 4))
        # create plot
        fig, ax = plt.subplots(num_rows, 4*2, figsize=(10,10))
        for i in range(num_rows):
            for j in range(8):
                if i*4+(j//2) < len(original_images):
                    if num_rows > 1:
                        if j % 2 == 0:
                            img = np.transpose(original_images[i*4+(j//2)], (1,2,0))
                            ax[i,j].imshow(img, cmap=cmap)
                            ax[i,j].set_title('Orig.')
                        else:
                            img = np.transpose(reconstructed_images[i*4+(j//2)],(1,2,0))
                            ax[i,j].imshow(img, cmap=cmap)
                            ax[i,j].set_title('Reconst.')
                        ax[i,j].axis('off')
                    else:
                        if j % 2 == 0:
                            img = np.transpose(original_images[j//2],(1,2,0))
                            ax[j].imshow(img, cmap=cmap)
                            ax[j].set_title('Orig.')
                        else:
                            img = np.transpose(reconstructed_images[j//2],(1,2,0))
                            ax[j].imshow(img, cmap=cmap)
                            ax[j].set_title('Reconst.')
                        ax[j].axis('off')
        fig.suptitle(title)

        # save plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_forward_process(self, diffusion_model: DiffusionModel, x: torch.Tensor, t: List[int], 
                             save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','forward_process'),
                             fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Forward-Process.png",
                             title: str = 'Forward Process', cmap: str = 'gray', denormalize: bool = True):
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
            x_t = x_t.permute(0, 2, 3, 1).detach().cpu().numpy()
            if denormalize:
                x_t = self.denormalize(x_t)
            x_t = np.clip(x_t, 0, 1)
            ax[idx].imshow(x_t[0], cmap=cmap)
            ax[idx].axis('off')
            ax[idx].set_title(f't={t[idx]}')
        fig.suptitle(title)

        # save plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()

    def plot_reverse_process(self, samples: List[np.ndarray], t: List[int], 
                             save_path: str = os.path.join(PROJECT_BASE_DIR,'results','images','reverse_process'),
                             fig_name: str = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Reverse-Process.png",
                             title: str = 'Reverse Process', cmap: str = 'gray', denormalize: bool = True):
        '''
        Plot the reverse process of the diffusion model

        Inputs:
        - samples: List of samples of shape [num_samples, C, H, W] corresponding to all time steps
        - save_path: Path to save-directory
        - fig_name: Name of the saved file
        '''
        samples.reverse() # reverse order of samples
        # create plot
        fig, ax = plt.subplots(1, len(t), figsize=(10,6))
        for idx in range(len(t)):
            sample = np.transpose(samples[t[idx]][0],(1,2,0))
            if denormalize:
                sample = self.denormalize(sample)
            sample = np.clip(sample, 0, 1)
            ax[idx].imshow(sample, cmap=cmap)
            ax[idx].axis('off')
            ax[idx].set_title(f't={t[idx]}')
        fig.suptitle(title)

        # save plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, fig_name))
        plt.close()