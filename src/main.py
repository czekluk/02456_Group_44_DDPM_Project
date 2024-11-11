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
from model import UNet

def main():
    # Initialize data module & get data loaders
    data_module = DiffusionDataModule()
    train_loader = data_module.get_MNIST_dataloader(
        train=True,
        batch_size=32,
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
    model = UNet()
    diffusion_model = DiffusionModel(model)

    # Inititalize trainer object
    trainer = Trainer(
        model=diffusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch.optim.Adam(diffusion_model.model.parameters(), lr=1e-4),
        num_epochs=10
    )

    # Train the model
    logger = trainer.train()

    # Plot the plots & save the model & logs
    logger.plot()
    logger.save()

if __name__ == "__main__":
    main()