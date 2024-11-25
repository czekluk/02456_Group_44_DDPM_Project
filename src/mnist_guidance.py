import torch
from torchvision import datasets, transforms
import numpy as np
import os
import sys

from dataset import DiffusionDataModule
from model import Model
from diffusion_model import DiffusionModel
from visualizer import Visualizer

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MNISTGuidanceClassifier(torch.nn.Module):
    def __init__(self, img_shape=(1, 28, 28), num_classes=10):
        super(MNISTGuidanceClassifier, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = torch.nn.ReLU()

        self.fc = torch.nn.Linear(64*int(self.img_shape[1]/4)*int(self.img_shape[2]/4), self.num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def train():
    model = MNISTGuidanceClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Load MNIST dataset
    data_module = DiffusionDataModule()
    train_loader = data_module.get_MNIST_dataloader(
        train=True,
        batch_size=128,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.5], std=[0.5])
                                      ])
    )
    val_loader = data_module.get_MNIST_dataloader(
        train=False,
        batch_size=128,
        shuffle=True,
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.5], std=[0.5])
                                      ])
    )

    NUM_EPOCHS = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = []
        train_accuracy = []
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            train_accuracy.append((y_pred.argmax(1) == y).float().mean().item())

        print(f'Epoch: {epoch+1} | Loss: {np.mean(train_loss)} | Accuracy: {np.mean(train_accuracy)}')

        model.eval()
        test_loss = []
        test_accuracy = []
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss.append(loss.item())
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            test_accuracy.append((y_pred.argmax(1) == y).float().mean().item())

        print(f'Epoch: {epoch+1} | Validation Loss: {np.mean(test_loss)} | Validation Accuracy: {np.mean(test_accuracy)}')

    torch.save(model.state_dict(), os.path.join(PROJECT_BASE_DIR,'results','guidance','mnist_guidance_classifier.pth'))

def guided_sampling():
    classifier = MNISTGuidanceClassifier()
    classifier.load_state_dict(torch.load(os.path.join(PROJECT_BASE_DIR,'results','guidance','mnist_guidance_classifier.pth'), weights_only=True))

    model = Model(ch=64, out_ch=1, ch_down_mult=(2, 4), num_res_blocks=2, attn_resolutions=[64, 128], dropout=0.1, resamp_with_conv=True)
    diff_model = DiffusionModel(model, T=1000, classifier=classifier, img_shape=(1, 28, 28), lambda_guidance=100)
    diff_model.load(os.path.join(PROJECT_BASE_DIR, 'results/models/2_downsampling_2resnet_with_attention_in_every_layer_64ch_128ch_256ch_30_epochs_2024-11-22_04-22-17-Epoch_0030-ValLoss_5.85-DiffusionModel.pth'))

    samples = diff_model.guided_sample(n_samples=16, class_label=3)

    vis = Visualizer()
    vis.plot_multiple_images(samples, title='Guided Sampling', 
                             save_path=os.path.join(PROJECT_BASE_DIR,'results','images','guided_sampling'),
                             denormalize=False)

if __name__ == "__main__":
    #train()
    guided_sampling()