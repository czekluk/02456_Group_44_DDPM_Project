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

def main():
    # Initialize data module & get data loaders
    data_module = DiffusionDataModule()
    train_loader = data_module.get_MNIST_dataloader(
        train=True,
        batch_size=128,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    val_loader = data_module.get_MNIST_dataloader(
        train=False,
        batch_size=128,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    

    # Initialize diffusion model
    T =1000
    model = Model(ch=64, out_ch=1, ch_down_mult=(2, 4), num_res_blocks=2, attn_resolutions=[64, 128], dropout=0.1, resamp_with_conv=True)
    diffusion_model = DiffusionModel(model, T=T)

    # Inititalize trainer object
    trainer = Trainer(
        model=diffusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch.optim.Adam(diffusion_model.model.parameters(), lr=1e-4),
        num_epochs=30
    )

    # Train the model
    logger = trainer.train(n_scores=10)

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
