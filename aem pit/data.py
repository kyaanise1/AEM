from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_mnist_loaders(batch_size=8):
    transform = transforms.ToTensor()

    # Use relative path for data directory
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'MNIST')

    train_dataset = datasets.MNIST(
        root=data_path, train=True, transform=transform, download=True)

    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
