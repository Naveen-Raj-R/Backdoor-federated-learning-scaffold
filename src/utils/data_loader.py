import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def load_data():
    transform = get_transforms()
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return dataset, test_dataset

def split_dataset(dataset, num_clients):
    data_per_client = len(dataset) // num_clients
    client_data = []
    for i in range(num_clients):
        indices = list(range(i * data_per_client, (i + 1) * data_per_client))
        client_data.append(torch.utils.data.Subset(dataset, indices))
    return client_data