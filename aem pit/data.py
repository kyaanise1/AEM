from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_mnist_loaders(batch_size=8):
    transform = transforms.ToTensor()

    # Your custom MNIST path
    data_path = r"C:\Users\User\AEM Final PIT - Copy\AEM\aem pit\data\MNIST"

    train_dataset = datasets.MNIST(
        root=data_path, train=True, transform=transform, download=True)

    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
