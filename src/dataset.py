import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_BASE_DIR, "data")

class DiffusionDataModule:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

    def get_MNIST_dataloader(self, train: bool, batch_size: int, shuffle: bool, transform):
        dataset = datasets.MNIST(root=self.data_dir, train=train, transform=transform, download=True)
        # NOTE: uncomment if you want to limit the dataset for testing purposes
        # dataset = Subset(dataset, list(range(128)))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_CIFAR10_dataloader(self, train: bool, batch_size: int, shuffle: bool, transform):
        dataset = datasets.CIFAR10(root=self.data_dir, train=train, transform=transform, download=True)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def test_MNIST_loader():
    print("Evaluating MNIST data batch:")
    data_module = DiffusionDataModule(DATA_DIR)
    train_loader = data_module.get_MNIST_dataloader(
        train=True,
        batch_size=32,
        shuffle=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    test_loader = data_module.get_MNIST_dataloader(
        train=False,
        batch_size=32,
        shuffle=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    for x, y in train_loader:
        print("Train data:")
        print(" x shape: ", x.shape)
        print(" y shape: ", y.shape)
        print(" Max value: ", x.max(), "Min value: ", x.min())
        break

    for x, y in test_loader:
        print("Test data:")
        print(" x shape: ", x.shape)
        print(" y shape: ", y.shape)
        print(" Max value: ", x.max(), "Min value: ", x.min())
        break
    
def test_CIFAR10_loader():
    print("Evaluating CIFAR10 data batch:")
    data_module = DiffusionDataModule(DATA_DIR)
    train_loader = data_module.get_CIFAR10_dataloader(
        train=True,
        batch_size=32,
        shuffle=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))    
        ])
    )
    test_loader = data_module.get_CIFAR10_dataloader(
        train=False,
        batch_size=32,
        shuffle=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))    
        ])
    )
    for x, y in train_loader:
        print("Train data:")
        print(" x shape: ", x.shape)
        print(" y shape: ", y.shape)
        print(" Max value: ", x.max(), "Min value: ", x.min())
        break

    for x, y in test_loader:
        print("Test data:")
        print(" x shape: ", x.shape)
        print(" y shape: ", y.shape)
        print(" Max value: ", x.max(), "Min value: ", x.min())
        break

if __name__ == "__main__":
    print("Downloading data to: " + DATA_DIR)
    test_MNIST_loader()
    test_CIFAR10_loader()