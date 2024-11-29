import os
import json
import numpy as np
from datetime import datetime
from visualizer import Visualizer

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Logger:
    def __init__(self):
        self.visualizer = Visualizer()

        # List containing training & validation metrics
        self.loss = []
        self.val_loss = []
        self.fid_scores = []
        self.loss_conf = []
        self.val_loss_conf = []
        self.fid_conf = []

        # Best model
        self.best_model = None
        self.best_epoch = None
        self.best_scores = np.array([np.inf, np.inf, 0])

        # Last model
        self.last_model = None
        self.last_epoch = None
        self.last_scores = np.array([np.inf, np.inf, 0])

    def log_training(self, loss, val_loss, fid_score, loss_conf, val_loss_conf, fid_conf):
        '''
        Method to log the training & validation metrics.
        To be called after every epoch.

        Inputs:
        - losses: List of training losses
        - fid_scores: List of FID scores
        - is_scores: List of Inception scores
        - loss_conf: upper & lower confidence interval bounds (gaussian)
        - val_loss_conf: upper & lower confidence interval bounds (gaussian)
        - fid_conf: upper & lower confidence interval bounds (student-t)
        '''
        self.loss.append(loss)
        self.val_loss.append(val_loss)
        self.fid_scores.append(fid_score)
        self.loss_conf.append(loss_conf)
        self.val_loss_conf.append(val_loss_conf)
        self.fid_conf.append(fid_conf)

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

    def plot(self, save_path=None):
        '''
        Method to plot the training & validation metrics.
        To be called after training.
        '''
        # Plot the loss & scores
        self.visualizer.plot_loss(self.loss, self.val_loss, self.loss_conf, self.val_loss_conf, save_path=save_path)
        self.visualizer.plot_fid_score(self.fid_scores, self.fid_conf, save_path=save_path)

    def save(self, save_dir=None):
        '''
        Method to save the best model and all logs.
        To be called after training.
        '''
        # Save the logs
        epochs = list(np.arange(0, len(self.loss)))
        data = [epochs, self.loss, self.val_loss, self.fid_scores]
        if save_dir is None:
            file_name = os.path.join(PROJECT_BASE_DIR, 'results', 'logs', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Logs.csv")
        else:
            file_name = os.path.join(save_dir, 'logs', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Logs.csv")
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
        best_model_name=os.path.join(save_dir,f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Epoch_{self.best_epoch:04}-ValLoss_{self.best_scores[1]:.2f}-BestDiffusionModel.pth")
        self.best_model.save(best_model_name)
        last_model_name=os.path.join(save_dir,f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-Epoch_{self.last_epoch:04}-ValLoss_{self.last_scores[1]:.2f}-LastDiffusionModel.pth")
        self.last_model.save(last_model_name)