# 02456_Group_44_DDPM_Project

The project is to re-implement the Denoising Diffusion Probabilistic Model (DDPM) in PyTorch and reproduce their results at least on MNIST and ideally on CIFAR-10. This paper is the one that kicked off the diffusion movement, it is a great way to learn what diffusion is all about and have hands-on experience. Link to paper: https://arxiv.org/abs/2006.11239.

## Project Structure
```
├── docs                    # Documentation 
├── resources               # Images, plots, etc.
└── src                     # source code
    ├── dataset.py              # Dataloader, Dataset
    ├── model.py                # PyTorch neural network
    ├── main.py                 # Main code
    ├── trainer.py              # Model training
    ├── utils.py                # Extra functions
    └── visualizer.py           # Plot generation, model outputs, etc.
```
# TODO:

* Dataloaders (MNIST, CIFAR10) - Alex
* Temporal Encoding (Transformer Sinusoidal Embedding) - Alex
* Model Architerture (Unet with Attention & ResNet Blocks) - Nandor
* Training Procedure (sample, train, loss, etc.) - Lukas
* Sampling Procedure (generate new images from noise) - Zeljko
* Visualizer - Lukas