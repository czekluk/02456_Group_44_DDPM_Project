import os
import json

from datetime import datetime
from torchvision import transforms

from dataset import DiffusionDataModule
from generator import Generator
from diffusion_model import DiffusionModel
from unet import SimpleModel
from metrics import tfFIDScore

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def post_mnist():
    model = SimpleModel(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
    gen = Generator(DiffusionModel(model, T=1000),
                    os.path.join(PROJECT_BASE_DIR, 'results/models/2024-11-16_21-08-13-Epoch_0004-FID_5.76-DiffusionModel.pth'))
    
    train_loader = DiffusionDataModule().get_MNIST_dataloader(
        train=True,
        batch_size=60000,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])
    )
    train_samples = next(iter(train_loader))

    test_loader = DiffusionDataModule().get_MNIST_dataloader(
        train=False,
        batch_size=10000,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])
    )
    test_samples = next(iter(test_loader))

    scorer = tfFIDScore(mode='mnist')

    gen_train_samples = gen.generate(num_samples=60000)
    gen_test_samples = gen.generate(num_samples=10000)

    train_fid = scorer.calculate_fid(train_samples, gen_train_samples)
    print(f'FID Score for training samples: {train_fid}')

    test_fid = scorer.calculate_fid(test_samples, gen_test_samples)
    print(f'FID Score for test samples: {test_fid}')

    file_name = os.path.join(PROJECT_BASE_DIR, 'results', 'metrics')
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    file_name = os.path.join(file_name, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-FID.json")
    json_dict = {'train_fid': train_fid, 'test_fid': test_fid}
    with open(file_name, 'w') as f:
        json.dump(json_dict, f)

def post_cifar10():
    model = SimpleModel(ch_layer0=64, out_ch=1, num_layers=3, num_res_blocks_per_layer=2, layer_ids_with_attn=[0,1,2], dropout=0.1, resamp_with_conv= True)
    gen = Generator(DiffusionModel(model, T=1000),
                    os.path.join(PROJECT_BASE_DIR, 'results/models/2024-11-16_21-08-13-Epoch_0004-FID_5.76-DiffusionModel.pth'))
    
    train_loader = DiffusionDataModule().get_CIFAR10_dataloader(
        train=True,
        batch_size=50000,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])
    )
    train_samples = next(iter(train_loader))

    test_loader = DiffusionDataModule().get_CIFAR10_dataloader(
        train=False,
        batch_size=10000,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])
    )
    test_samples = next(iter(test_loader))

    scorer = tfFIDScore(mode='cifar10')

    gen_train_samples = gen.generate(num_samples=60000)
    gen_test_samples = gen.generate(num_samples=10000)

    train_fid = scorer.calculate_fid(train_samples, gen_train_samples)
    print(f'FID Score for training samples: {train_fid}')

    test_fid = scorer.calculate_fid(test_samples, gen_test_samples)
    print(f'FID Score for test samples: {test_fid}')

    file_name = os.path.join(PROJECT_BASE_DIR, 'results', 'metrics')
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    file_name = os.path.join(file_name, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-FID.json")
    json_dict = {'train_fid': train_fid, 'test_fid': test_fid}
    with open(file_name, 'w') as f:
        json.dump(json_dict, f)

if __name__ == "__main__":
    post_mnist()