import torch
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from datetime import datetime
from torchvision import datasets, transforms

from diffusion_model import DiffusionModel
from dataset import DiffusionDataModule
from trainer import Trainer
from logger import Logger
from visualizer import Visualizer
from model import Model
from schedule import LinearSchedule, CosineSchedule

def main():

    DATA_FLAG = "mnist" # change to "mnist" or "cifar10"

    # Initialize data module & get data loaders
    data_module = DiffusionDataModule()

    if DATA_FLAG == "cifar10":
        train_loader = data_module.get_CIFAR10_dataloader(
            train=True,
            batch_size=128,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        val_loader = data_module.get_CIFAR10_dataloader(
            train=False,
            batch_size=128,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    elif DATA_FLAG == "mnist":
        train_loader = data_module.get_MNIST_dataloader(
            train=True,
            batch_size=128,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        val_loader = data_module.get_MNIST_dataloader(
            train=False,
            batch_size=128,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    else:
        raise NotImplementedError

    # Initialize diffusion model
    T = 1000
    if DATA_FLAG == "cifar10":
        model = Model(ch=64, out_ch=3, ch_down_mult=(1, 2), num_res_blocks=2, attn_resolutions=[7], dropout=0.1, resamp_with_conv=True)
        schedule = LinearSchedule(10e-4, 0.02, T)
        diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(3, 32, 32))
        # Inititalize trainer object
        trainer = Trainer(
            model=diffusion_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=torch.optim.Adam(diffusion_model.model.parameters(), lr=1e-4),
            num_epochs=2,
            normalized=True,
            validate=False
        )
    elif DATA_FLAG == "mnist":
        model = Model(ch=64, out_ch=1, ch_down_mult=(1, 2), num_res_blocks=2, attn_resolutions=[7], dropout=0.1, resamp_with_conv=True)
        schedule = LinearSchedule(10e-4, 0.02, T)
        diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(1, 28, 28))
        # Inititalize trainer object
        trainer = Trainer(
            model=diffusion_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=torch.optim.Adam(diffusion_model.model.parameters(), lr=1e-4),
            num_epochs=2,
            normalized=True
        )
    else:
        raise NotImplementedError

    # Train the model
    logger = trainer.train()

    # Plot the plots & save the model & logs
    logger.plot()
    logger.save()

    # Get the best model from the logger
    diffusion_model = logger.best_model

    # Plot the samples
    visualizer = Visualizer()
    x, _ = next(iter(val_loader))
    visualizer.plot_forward_process(diffusion_model, x, [0, T//4, T//2, T*3//4, T])
    samples = diffusion_model.all_step_sample(n_samples=16)
    visualizer.plot_reverse_process(samples, [0, T//4, T//2, T*3//4, T])

    # Sample from the model
    samples = diffusion_model.sample(n_samples=16)
    visualizer.plot_multiple_images(samples)

if __name__ == "__main__":
    main()
