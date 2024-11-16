import torch
import os
import sys

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
        self.fid_scores = []
        self.is_scores = []

        # Best model
        self.best_model = None
        self.best_epoch = None
        self.best_scores = np.array([np.inf, np.inf, 0])

    def log_training(self, loss, fid_score, is_score):
        '''
        Method to log the training & validation metrics.
        To be called after every epoch.

        Inputs:
        - losses: List of training losses
        - fid_scores: List of FID scores
        - is_scores: List of Inception scores
        '''
        self.loss.append(loss)
        self.fid_scores.append(fid_score)
        self.is_scores.append(is_score)

    def log_model(self, model, epoch):
        '''
        Method to log the best model based on the FID score.
        To be called after every epoch & after logger.log_training().

        Inputs:
        - model: Model to be saved
        - epoch: Epoch number
        '''
        self.best_epoch = epoch

        if self.best_model is None:
            self.best_model = model
            self.best_scores = np.array([self.loss[-1], self.fid_scores[-1], self.is_scores[-1]])
        else:
            if self.fid_scores[-1] < self.fid_scores[-2]:
                self.best_model = model
                self.best_scores = np.array([self.loss[-1], self.fid_scores[-1], self.is_scores[-1]])

    def plot(self):
        '''
        Method to plot the training & validation metrics.
        To be called after training.
        '''
        # Plot the loss & scores
        self.visualizer.plot_loss(self.loss)
        self.visualizer.plot_is_fid_score(self.is_scores, self.fid_scores)

    def save(self):
        '''
        Method to save the best model and all logs.
        To be called after training.
        '''
        # Save the logs
        epochs = list(np.arange(0, len(self.loss)))
        data = [[epochs, self.loss, self.fid_scores, self.is_scores]]
        file_name = os.path.join(PROJECT_BASE_DIR, 'results', 'logs', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Logs.csv")
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f'Logs saved to {file_name}')

        # Save the best model
        self.best_model.save(model_name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Epoch_{self.best_epoch:4}-FID_{self.best_scores[1]}-DiffusionModel.pth")