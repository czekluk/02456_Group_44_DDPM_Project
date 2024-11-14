import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import sqrtm
from datetime import datetime
from torchvision.models import inception_v3
from torchvision import transforms

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ScoreInceptionV3(torch.nn.Module):
    def __init__(self):
        '''
        Class for the InceptionV3 model used for calculating the Scores.

        Inputs:
        - None
        '''
        super(ScoreInceptionV3, self).__init__()
        self.model = inception_v3(weights='IMAGENET1K_V1', progress=True).eval()

        # replace fully connected layer by identity to maintain feature activations
        self.model.fc = torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        '''
        Forward pass of the InceptionV3 model.

        Inputs:
        - x: Input images [N, C, H, W]

        Returns:
        - out: Feature activations of the model right before the fully connected layer
        '''
        return self.model(x)

class FIDScore:
    def __init__(self):
        '''
        Class implementing the Fréchet Inception Distance (FID) score.
        https://arxiv.org/pdf/1706.08500

        Inputs:
        - None
        '''
        self.inception_model = ScoreInceptionV3()

        # transforms for resizing and normalizing the images
        # normalizing w.r.t. the ImageNet mean & std.
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def calculate_fid(self, real_img: torch.Tensor, gen_img: torch.Tensor):
        '''
        Calculate the FID score between the real and generated images.
        Assumes images have been sampled from their respective distributions.

        Inputs:
        - real_img: Real images [N, 3, H, W] as torch.Tensor
        - gen_img: generated images [N, 3, H, W] as torch.Tensor

        Returns:
        - fid_score: FID score between the real and generated images
        '''
        # make sure dimensions are correct
        if real_img.dim() != 4 or gen_img.dim() != 4:
            raise ValueError('Input tensors must have 4 dimensions')
        
        if real_img.shape != gen_img.shape:
            raise ValueError('Input tensors must have the same shape')
        
        if real_img.shape[1] != 3 or gen_img.shape[1] != 3:
            # Check if images are grayscale (1 channel) and convert to RGB (3 channels) if needed (to work for MNIST)
            if real_img.shape[1] == 1:
                real_img = real_img.repeat(1, 3, 1, 1)  # Repeat across the channel dimension
            else:
                raise ValueError('Input tensor must have 3 channels')
            if gen_img.shape[1] == 1:
                gen_img = gen_img.repeat(1, 3, 1, 1)  # Repeat across the channel dimension
            else:
                raise ValueError('Input tensor must have 3 channels')

        with torch.no_grad():
            # resize & normalize images
            real = torch.zeros((real_img.shape[0], 3, 299, 299))
            gen = torch.zeros((gen_img.shape[0], 3, 299, 299))
            for i in range(real_img.shape[0]):
                real[i] = self.transform(real_img[i])
                gen[i] = self.transform(gen_img[i])

            # get fully-connected activations from the inception model
            real_act = self.inception_model(real)
            gen_act = self.inception_model(gen)

            # calculate the mean and covariance of the activations
            real_mean = torch.mean(real_act, dim=0)
            gen_mean = torch.mean(gen_act, dim=0)

            real_cov = torch.cov(torch.transpose(real_act, 0, 1))
            gen_cov = torch.cov(torch.transpose(gen_act, 0, 1))

            # calculate the FID score
            cov_prod = (real_cov @ gen_cov).detach().numpy()
            calc_sqrtm = sqrtm(cov_prod)
            if np.iscomplexobj(calc_sqrtm):
                calc_sqrtm = calc_sqrtm.real
            calc_sqrtm = torch.from_numpy(calc_sqrtm)
            fid_score = torch.norm(real_mean - gen_mean)**2 + torch.trace(real_cov + gen_cov - 2*calc_sqrtm)

            return fid_score


class InceptionScore:
    def __init__(self):
        '''
        Class implementing the Inception Score.
        https://arxiv.org/pdf/1606.03498

        Inputs:
        - None
        '''
        self.inception_model = inception_v3(weights='IMAGENET1K_V1', progress=True).eval()

        # transforms for resizing and normalizing the images
        # normalizing w.r.t. the ImageNet mean & std.
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def calculate_is(self, gen_img: torch.Tensor, eps=1e-6):
        '''
        Calculate the Inception score between the real and generated images.

        Inputs:
        - gen_img: generated images [N, 3, H, W] as torch.Tensor
        - eps: small value to prevent ln(0)

        Returns:
        - is_score: Inception score of the generated images
        '''
        # make sure dimensions are correct
        if gen_img.dim() != 4:
            raise ValueError('Input tensor must have 4 dimensions')
        
        if  gen_img.shape[1] != 3:
            # Check if images are grayscale (1 channel) and convert to RGB (3 channels) if needed (to work for MNIST)
            if gen_img.shape[1] == 1:
                gen_img = gen_img.repeat(1, 3, 1, 1)  # Repeat across the channel dimension
            else:
                raise ValueError('Input tensor must have 3 channels')

        with torch.no_grad():
            # resize & normalize images
            gen = torch.zeros((gen_img.shape[0], 3, 299, 299))
            for i in range(gen_img.shape[0]):
                gen[i] = self.transform(gen_img[i])

            # calculate p(y|x) using the inception model
            p_yx = self.inception_model(gen)
            p_yx = torch.nn.functional.softmax(p_yx, dim=1)

            # calculate p(y) by averaging over x
            p_y = torch.mean(p_yx, dim=0)

            # calculate KL divergence
            kl_div = p_yx * (torch.log(p_yx + eps) - torch.log(p_y.unsqueeze(0) + eps))

            # sum & average kl divergence over classes
            kl_div = torch.mean(torch.sum(kl_div, dim=1))

            # calculate the Inception score
            is_score = torch.exp(kl_div)

            return is_score


def test_fid_score():
    '''
    Test the FID score calculation.
    '''
    fid = FIDScore()

    # generate random images from same distribution
    real_img = torch.rand((50, 3, 64, 64))
    gen_img = torch.rand((50, 3, 64, 64))

    fid_score = fid.calculate_fid(real_img, gen_img)
    print(f'FID Score from similar distributions: {fid_score:.4f}')

    # generate random imagesfrom dissimilar distribution
    real_img = torch.rand((50, 3, 64, 64))
    gen_img = torch.rand((50, 3, 64, 64)) + 5

    fid_score = fid.calculate_fid(real_img, gen_img)
    print(f'FID Score from dissimilar distributions: {fid_score:.4f}')

def test_is_score():
    '''
    Test the Inception score calculation.
    '''
    iSc = InceptionScore()

    # generate random images from same distribution
    gen_img = torch.rand((50, 3, 64, 64))

    inception_score = iSc.calculate_is(gen_img)
    print(f'Inception Score: {inception_score:.4f}')


if __name__ == '__main__':
    # Note: Process is killed when batch size is too large
    test_fid_score()

    test_is_score()