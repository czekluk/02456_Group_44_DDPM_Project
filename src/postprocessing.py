import os
import json
import argparse
import sys
import numpy as np
import torch
import math

from datetime import datetime
from torchvision import transforms
from tqdm import tqdm

from dataset import DiffusionDataModule
from generator import Generator
from diffusion_model import DiffusionModel
from unet import SimpleModel
from metrics import tfFIDScore
from schedule import LinearSchedule, CosineSchedule

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    parser = argparse.ArgumentParser(description="Train a diffusion model.")
    parser.add_argument("--data_type", type=str, choices=["mnist", "cifar10"], required=True, help="Dataset to use: 'mnist' or 'cifar10'")
    parser.add_argument("--schedule", type=str, choices=["linear", "cosine"], required=True, help="Schedule type: 'linear' or 'cosine'")
    parser.add_argument("--attention", type=str, choices=["attention", "noattention"], required=True, help="Attention type: 'attention' or 'noattention'")
    parser.add_argument("--model", type=str,required=True, help="Model path: relative path to the trained UNet to use for sampling")
    parser.add_argument("--batch_size", type=int, required=True, help="Bacth size: choose batch size for FID calculation")

    args = parser.parse_args(args)

    DATA_FLAG = args.data_type
    SCHEDULE_FLAG = args.schedule
    ATTENTION_FLAG = args.attention
    MODEL_PATH = args.model
    BATCH_SIZE = args.batch_size

    # Initialize diffusion model
    T = 1000
    if DATA_FLAG == "cifar10":
        if ATTENTION_FLAG=="attention":
            model = SimpleModel(ch_layer0=64, out_ch=3, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
        elif ATTENTION_FLAG=="noattention":
            model = SimpleModel(ch_layer0=64, out_ch=3, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True)
        
        if SCHEDULE_FLAG == "linear":
            schedule = LinearSchedule(10e-4, 0.02, T)
        elif SCHEDULE_FLAG == "cosine":
            schedule = CosineSchedule(T)

        diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(3, 32, 32))
        gen = Generator(diffusion_model, os.path.join(PROJECT_BASE_DIR, MODEL_PATH))

    elif DATA_FLAG == "mnist":
        if ATTENTION_FLAG=="attention":
            model = SimpleModel(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
        elif ATTENTION_FLAG=="noattention":
            model = SimpleModel(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[], dropout=0.1, resamp_with_conv= True)
        
        if SCHEDULE_FLAG == "linear":
            schedule = LinearSchedule(10e-4, 0.02, T)
        elif SCHEDULE_FLAG == "cosine":
            schedule = CosineSchedule(T)

        diffusion_model = DiffusionModel(model, T=T, schedule=schedule, img_shape=(1, 28, 28))
        gen = Generator(diffusion_model, os.path.join(PROJECT_BASE_DIR, MODEL_PATH))

    else:
        raise NotImplementedError

    data_module = DiffusionDataModule()
    if DATA_FLAG == "cifar10":
        train_loader, val_loader, test_loader = data_module.get_CIFAR10_data_split(
            batch_size=BATCH_SIZE,
            val_split=0.0,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        )
        scorer = tfFIDScore(mode='cifar10')
        SIZE_TRAIN = 60000
        SIZE_TEST = 10000
    elif DATA_FLAG == "mnist":
        train_loader, val_loader, test_loader = data_module.get_MNIST_data_split(
            batch_size=BATCH_SIZE,
            val_split=0.0,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        scorer = tfFIDScore(mode='mnist')
        SIZE_TRAIN = 50000
        SIZE_TEST = 10000
    else:
        raise NotImplementedError
    
    overall_train_fid = []
    overall_test_fid = []

    for i in range(0, math.ceil(SIZE_TRAIN/BATCH_SIZE)):
        train_fid = []

        gen_samples = gen.generate(num_samples=BATCH_SIZE)
        gen_samples = torch.from_numpy(gen_samples)

        for minibatch_idx, (x, _) in tqdm(enumerate(train_loader), unit='minibatch', total=len(train_loader)):
            if x.shape[0] == gen_samples.shape[0]:
                _train_fid = scorer.calculate_fid(x, gen_samples)
            else:
                temp_samples = gen_samples[:x.shape[0]]
                _train_fid = scorer.calculate_fid(x, temp_samples)
            train_fid.append(_train_fid)
        
        overall_train_fid.append(np.mean(train_fid))
        print(f"Iteration {i}/{math.ceil(SIZE_TRAIN/BATCH_SIZE)}: Mean FID on training dataset: {np.mean(train_fid)}")

    for i in range(0, math.ceil(SIZE_TEST/BATCH_SIZE)):
        test_fid = []

        gen_samples = gen.generate(num_samples=BATCH_SIZE)
        gen_samples = torch.from_numpy(gen_samples)

        for minibatch_idx, (x, _) in tqdm(enumerate(test_loader), unit='minibatch', total=len(test_loader)):
            if x.shape[0] == gen_samples.shape[0]:
                _test_fid = scorer.calculate_fid(x, gen_samples)
            else:
                temp_samples = gen_samples[:x.shape[0]]
                _test_fid = scorer.calculate_fid(x, temp_samples)
            test_fid.append(_test_fid)

        overall_test_fid.append(np.mean(test_fid))
        print(f"Iteration {i}/{math.ceil(SIZE_TEST/BATCH_SIZE)}: Mean FID on test dataset: {np.mean(test_fid)}")

    print(f"Mean FID on training dataset: {np.mean(overall_train_fid)}")
    print(f"Mean FID on test dataset: {np.mean(overall_test_fid)}")

    path_name = os.path.join(PROJECT_BASE_DIR, 'results', 'metrics', DATA_FLAG, SCHEDULE_FLAG, ATTENTION_FLAG)
    if not os.path.exists(path_name):
        os.makedirs(os.path.dirname(path_name))

    file_name = os.path.join(path_name, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-FID.json")
    json_dict = {'model_path': MODEL_PATH,
                 'train_fid': np.mean(overall_train_fid), 
                 'test_fid': np.mean(overall_test_fid)}
    with open(file_name, 'w') as f:
        json.dump(json_dict, f, indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])