import torch
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import csv

from datetime import datetime
from torchvision import datasets, transforms

from visualizer import Visualizer

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Logger:
    def __init__(self):
        self.visualizer = Visualizer()

        # List containing training & validation metrics
        self.loss =[]
        self.val_loss = []
        self.fid_scores = []

        # Best model
        self.best_model = None
        self.best_epoch = None
        self.best_scores = np.array([np.inf, np.inf, 0])

        # Last model
        self.last_model = None
        self.last_epoch = None
        self.last_scores = np.array([np.inf, np.inf, 0])

    def log_training(self, loss, val_loss, fid_score):
        '''
        Method to log the training & validation metrics.
        To be called after every epoch.

        Inputs:
        - losses: List of training losses
        - fid_scores: List of FID scores
        - is_scores: List of Inception scores
        '''
        self.loss.append(loss)
        self.val_loss.append(val_loss)
        self.fid_scores.append(fid_score)

    def log_model(self, model, epoch):
        '''
        Method to log the best model based on the FID score.
        To be called after every epoch & after logger.log_training().

        Inputs:
        - model: Model to be saved
        - epoch: Epoch number
        '''
        if self.best_model is None:
            self.best_model = model
            self.best_scores = np.array([self.loss[-1], self.val_loss[-1], self.fid_scores[-1]])
            self.best_epoch = epoch
        else:
            if self.val_loss[-1] == min(self.val_loss):
                self.best_model = model
                self.best_scores = np.array([self.loss[-1], self.val_loss[-1], self.fid_scores[-1]])
                self.best_epoch = epoch

        self.last_model = model
        self.last_scores = np.array([self.loss[-1], self.val_loss[-1], self.fid_scores[-1]])
        self.last_epoch = epoch

    def plot(self):
        '''
        Method to plot the training & validation metrics.
        To be called after training.
        '''
        # Plot the loss & scores
        self.visualizer.plot_loss(self.loss, self.val_loss)
        self.visualizer.plot_fid_score(self.fid_scores)

    def save(self):
        '''
        Method to save the best model and all logs.
        To be called after training.
        '''
        # Save the logs
        epochs = list(np.arange(0, len(self.loss)))
        data = [epochs, self.loss, self.val_loss, self.fid_scores]
        file_name = os.path.join(PROJECT_BASE_DIR, 'results', 'logs', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Logs.json")
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        json_dict = {'epochs': [int(item) for item in epochs], 
                     'loss': [float(item) for item in self.loss], 
                     'val_loss': [float(item) for item in self.val_loss], 
                     'fid_scores': [float(item) for item in self.fid_scores]}
        with open(file_name, 'w') as f:
            json.dump(json_dict, f)
        print(f'Logs saved to {file_name}')

        # Save the best model
        self.best_model.save(model_name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Epoch_{self.best_epoch:04}-ValLoss_{self.best_scores[1]:.2f}-BestDiffusionModel.pth")
        self.last_model.save(model_name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Epoch_{self.last_epoch:04}-ValLoss_{self.last_scores[1]:.2f}-LastDiffusionModel.pth")