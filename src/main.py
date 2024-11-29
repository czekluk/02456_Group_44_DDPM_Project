# import torch
import os
# import sys

# from datetime import datetime
# from torchvision import transforms

# from diffusion_model import DiffusionModel
# from dataset import DiffusionDataModule
# from trainer import Trainer
# from logger import Logger
# from visualizer import Visualizer
# from unet import SimpleModel
# from schedule import LinearSchedule, CosineSchedule
import argparse
import sys

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument("--data_type", type=str, choices=["mnist", "cifar10"], required=True, help="Dataset to use: 'mnist' or 'cifar10'")
    parser.add_argument("--schedule", type=str, choices=["linear", "cosine"], required=True, help="Schedule type: 'linear' or 'cosine'")
    parser.add_argument("--attention", type=str, choices=["attention", "noattention"], required=True, help="Attention type: 'attention' or 'noattention'")

    args = parser.parse_args(args)

    # DATA_FLAG = args.data_type
    # SCHEDULE_FLAG = args.schedule
    # ATTENTION_FLAG = args.attention

    # # Initialize data module & get data loaders
    # data_module = DiffusionDataModule()

    # if DATA_FLAG == "cifar10":
    #     train_loader = data_module.get_CIFAR10_dataloader(
    #         train=True,
    #         batch_size=128,
    #         shuffle=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5,), (0.5,))
    #         ])
    #     )
    #     val_loader = data_module.get_CIFAR10_dataloader(
    #         train=False,
    #         batch_size=128,
    #         shuffle=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5,), (0.5,))
    #         ])
    #     )
    # elif DATA_FLAG == "mnist":
    #     train_loader = data_module.get_MNIST_dataloader(
    #         train=True,
    #         batch_size=128,
    #         shuffle=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5,), (0.5,))
    #         ])
    #     )
    #     val_loader = data_module.get_MNIST_dataloader(
    #         train=False,
    #         batch_size=128,
    #         shuffle=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5,), (0.5,))
    #         ])
    #     )
    # else:
    #     raise NotImplementedError

    # # Initialize diffusion model
    # T = 1000
    # if DATA_FLAG == "cifar10":
    #     if ATTENTION_FLAG=="attention":
    #         model = SimpleModel(ch_layer0=32, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
    #     elif ATTENTION_FLAG=="noattention":
    #         model = SimpleModel(ch_layer0=32, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True)
    #     if SCHEDULE_FLAG == "linear":
    #         schedule = LinearSchedule(10e-4, 0.02, T)
    #     elif SCHEDULE_FLAG == "cosine":
    #         schedule = CosineSchedule(T)
    #     diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(3, 32, 32))
    #     # Inititalize trainer object
    #     trainer = Trainer(
    #         model=diffusion_model,
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         optimizer=torch.optim.Adam(diffusion_model.model.parameters(), lr=1e-4),
    #         num_epochs=60,
    #         normalized=True,
    #         validate=False
    #     )
    # elif DATA_FLAG == "mnist":
    #     if ATTENTION_FLAG=="attention":
    #         model = SimpleModel(ch_layer0=32, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
    #     elif ATTENTION_FLAG=="noattention":
    #         model = SimpleModel(ch_layer0=32, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True)
    #     if SCHEDULE_FLAG == "linear":
    #         schedule = LinearSchedule(10e-4, 0.02, T)
    #     elif SCHEDULE_FLAG == "cosine":
    #         schedule = CosineSchedule(T)
    #     diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(1, 28, 28))
    #     # Inititalize trainer object
    #     trainer = Trainer(
    #         model=diffusion_model,
    #         train_loader=train_loader,
    #         val_loader=val_loader,
    #         optimizer=torch.optim.Adam(diffusion_model.model.parameters(), lr=1e-4),
    #         num_epochs=30,
    #         normalized=True
    #     )
    # else:
    #     raise NotImplementedError

    # # Train the model
    # logger = trainer.train()

    # # Create unique save path for the model
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_dir = os.path.join(PROJECT_BASE_DIR, "results", DATA_FLAG, SCHEDULE_FLAG, ATTENTION_FLAG, timestamp)
    # os.makedirs(save_dir, exist_ok=True)
    # # Plot the plots & save the model & logs
    # logger.plot(save_path=save_dir)
    # logger.save(save_dir=save_dir)

    # # Get the best model from the logger
    # diffusion_model = logger.best_model

    # # Plot the samples
    # images_path = os.path.join(save_dir, "images")
    # visualizer = Visualizer()
    # x, _ = next(iter(val_loader))
    # visualizer.plot_forward_process(diffusion_model, x, [0, T//4, T//2, T*3//4, T], save_path=images_path)
    # samples = diffusion_model.all_step_sample(n_samples=16)
    # visualizer.plot_reverse_process(samples, [0, T//4, T//2, T*3//4, T], save_path=images_path)

    # # Sample from the model
    # samples = diffusion_model.sample(n_samples=16)
    # visualizer.plot_multiple_images(samples,save_path=images_path)

if __name__ == "__main__":
    main(sys.argv[1:])
