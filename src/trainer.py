import torch
import os

import numpy as np
from tqdm import tqdm
import scipy.stats as stats

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
                 validate = None):
        self.diffusion_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.logger = Logger()
        self.normalized = normalized
        self.validate_flag = validate

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
            loss = np.mean(epoch_loss)
            loss_interval = stats.norm.interval(0.95, loc=np.mean(epoch_loss), scale=np.std(epoch_loss))
            loss_conf = (float(loss_interval[0]), float(loss_interval[1]))
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
                    if self.validate_flag:
                        fid_score= self.validate(x)
                    else:
                        fid_score = -1
                    epoch_fid.append(fid_score)
            # calculate fid-statistics
            fid = np.mean(epoch_fid)
            fid_interval = stats.t.interval(0.95, len(epoch_fid)-1, loc=np.mean(epoch_fid), scale=stats.sem(epoch_fid))
            fid_conf = (float(fid_interval[0]), float(fid_interval[1]))
            # calculate val_loss-statistics
            val_loss = np.mean(epoch_val_loss)
            val_loss_interval = stats.norm.interval(0.95, loc=np.mean(epoch_val_loss), scale=np.std(epoch_val_loss))
            val_loss_conf = (float(val_loss_interval[0]), float(val_loss_interval[1]))

            print(f'Epoch: {epoch+1} | Validation Loss: {val_loss} | Approx. FID Score: {fid}')
            # Log the training & validation metrics
            self.logger.log_training(loss, val_loss, fid, loss_conf, val_loss_conf, fid_conf)
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
        fid = tfFIDScore(normalized=self.normalized, mode=self.validate_flag)

        self.diffusion_model.model.eval() # should possibly be inside the sample method
        with torch.no_grad():
            # Generate samples using the diffusion model
            gen = self.diffusion_model.sample(len(x))

            # Calculate FID score
            fid_score = fid.calculate_fid(x, torch.from_numpy(gen))

            # Calculate Inception Score
            # is_score = iSc.calculate_is(gen)
        
        return fid_score