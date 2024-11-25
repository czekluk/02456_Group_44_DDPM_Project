import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime

from diffusion_model import DiffusionModel
from metrics import FIDScore, InceptionScore, tfFIDScore
from logger import Logger

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Trainer:
    def __init__(self, model: DiffusionModel, 
                 train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, 
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int = 100,
                 normalized: bool = True,
                 validate: bool = True):
        self.diffusion_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.logger = Logger()
        self.normalized = normalized
        self.validate = validate

    def train(self):
        '''
        Method to train the diffusion model.

        Inputs:
        - None
        '''
        # Training & Validation loop
        for epoch in tqdm(range(self.num_epochs), unit='epoch'):
            # Training loop
            epoch_loss = []
            for minibatch_idx, (x, _) in tqdm(enumerate(self.train_loader), unit='minibatch', total=len(self.train_loader)):
                x = x.to(self.diffusion_model.device)
                loss = self.diffusion_model.train(x, self.optimizer)
                epoch_loss.append(loss)
            print(f'Epoch: {epoch+1} | Train Loss: {np.mean(epoch_loss)}')

            # Validation loop
            epoch_fid = []
            epoch_val_loss = []
            for minibatch_idx, (x, _) in tqdm(enumerate(self.val_loader), unit='minibatch', total=len(self.val_loader)):
                x = x.to(self.diffusion_model.device)
                val_loss = self.diffusion_model.val_loss(x)
                epoch_val_loss.append(val_loss)
                # Calculate fid score for first 5 minibatches
                if minibatch_idx < 5:
                    if self.validate:
                        fid_score= self.validate(x)
                    else:
                        fid_score = -1
                    epoch_fid.append(fid_score)
            fid = np.mean(epoch_fid)
            print(f'Epoch: {epoch+1} | Validation Loss: {np.mean(val_loss)} | Approx. FID Score: {fid}')

            # Log the training & validation metrics
            self.logger.log_training(np.mean(epoch_loss), np.mean(val_loss), fid)
            self.logger.log_model(self.diffusion_model, epoch+1)

        # Return the logger of the training process
        return self.logger
    
    def validate(self, x: torch.Tensor):
        '''
        Method to validate the diffusion model during training.

        Inputs:
        - x: Batch of validation images [B, C, H, W]

        Returns:
        - fid: Frechet Inception Distance
        - is_score: Inception Score
        '''
        # fid = FIDScore()
        # iSc = InceptionScore()
        fid = tfFIDScore(normalized=self.normalized)

        self.diffusion_model.model.eval() # should possibly be inside the sample method
        with torch.no_grad():
            # Generate samples using the diffusion model
            gen = self.diffusion_model.sample(len(x))

            # Calculate FID score
            fid_score = fid.calculate_fid(x, torch.from_numpy(gen))

            # Calculate Inception Score
            # is_score = iSc.calculate_is(gen)
        
        return fid_score