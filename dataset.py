import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_loaders(config):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root=config.data_root, train=True,
                                download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=config.data_root, train=False,
                               download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers)
    return train_loader, test_loader
