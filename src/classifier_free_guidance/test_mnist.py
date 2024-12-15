import torch
import os
import sys

from datetime import datetime
from torchvision import transforms

from diffusion_model_cf import DiffClassifierFreeGuidance
from dataset import DiffusionDataModule
from trainer_cf import TrainerClassFreeGuidance
from logger import Logger
from visualizer import Visualizer
from unet_cf import SimpleModelClassFreeGuidance
from schedule import LinearSchedule, CosineSchedule
import sys

PROJECT_BASE_DIR =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train():
    T = 1000
    # Initialize data module & get data loaders
    data_module = DiffusionDataModule()
    train_loader, val_loader, test_loader = data_module.get_MNIST_data_split(
            batch_size=128,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
    model = SimpleModelClassFreeGuidance(ch_layer0=32, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True, lambda_cf=10.)
    schedule = LinearSchedule(10e-4, 0.02, T)
    diffusion_model = DiffClassifierFreeGuidance(model, T=T, schedule=schedule, img_shape=(1, 28, 28))
    trainer = TrainerClassFreeGuidance(
        model=diffusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch.optim.Adam(diffusion_model.model.parameters(), lr=1e-4),
        num_epochs=30,
        normalized=True,
        validate="mnist"
    )
      # Train the model
    logger = trainer.train()
    # Create unique save path for the model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(PROJECT_BASE_DIR, "results", "classifier_free_guidance", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    # Plot the plots & save the model & logs
    logger.plot(save_path=save_dir)
    logger.save(save_dir=save_dir)
    # Get the best model from the logger
    diffusion_model = logger.best_model
    # Plot the samples
    samples = diffusion_model.sample(n_samples=16, class_label=torch.tensor([3], dtype=torch.int).to(diffusion_model.device))
    vis = Visualizer()
    vis.plot_multiple_images(samples, title='Classifier Free Guided Sampling', 
                             save_path=os.path.join(save_dir,'guided_sampling'),
                            denormalize=False)
def guided_sampling():
    T = 1000
    model = SimpleModelClassFreeGuidance(ch_layer0=32, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True, lambda_cf=10.)
    schedule = LinearSchedule(10e-4, 0.02, T)
    diffusion_model = DiffClassifierFreeGuidance(model, T=T, schedule=schedule, img_shape=(1, 28, 28))
    model_path = os.path.join("/zhome/25/a/202562/Deep-Learning/02456_Group_44_DDPM_Project/results/classifier_free_guidance/2024-12-04_05-28-31/2024-12-04_05-28-32-Epoch_0027-ValLoss_13.72-BestDiffusionModel.pth/2024-12-04_00-34-25-DiffusionModel.pth")
    diffusion_model.load(os.path.join(PROJECT_BASE_DIR, model_path))
    samples = diffusion_model.sample(n_samples=16, class_label=torch.tensor([7], dtype=torch.int).to(diffusion_model.device))
    vis = Visualizer()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vis.plot_multiple_images(samples, title='Guided Sampling', 
                             save_path=os.path.join(PROJECT_BASE_DIR,'results','classifier_free_guidance',timestamp),
                             denormalize=False)
if __name__ == "__main__":
    TRAIN_FLAG = False
    if TRAIN_FLAG:
        train()
    else:
        guided_sampling()
