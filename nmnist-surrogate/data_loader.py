"""
MNIST data loader for spiking neural network training.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np


def load_mnist(data_dir='./data', batch_size=128, flatten=True):
    """
    Load MNIST dataset using PyTorch.

    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for DataLoader
        flatten: If True, flatten images to 784-dim vectors

    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    # Define transforms
    if flatten:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W) format
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784-dim
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W) format
        ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"MNIST loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {'784 (flattened)' if flatten else '1×28×28'}")

    return train_loader, test_loader


def create_binary_mnist(data_dir='./data', batch_size=128, class_0=0, class_1=1):
    """
    Create binary MNIST dataset (two classes only) for simpler testing.

    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for DataLoader
        class_0: First class label
        class_1: Second class label

    Returns:
        train_loader: DataLoader for binary training set
        test_loader: DataLoader for binary test set
    """
    # Load full MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Filter for two classes only
    def filter_dataset(dataset, class_0, class_1):
        indices = [i for i, (_, label) in enumerate(dataset)
                  if label == class_0 or label == class_1]

        data = torch.stack([dataset[i][0] for i in indices])
        labels = torch.tensor([0 if dataset[i][1] == class_0 else 1 for i in indices])

        return TensorDataset(data, labels)

    train_binary = filter_dataset(train_dataset, class_0, class_1)
    test_binary = filter_dataset(test_dataset, class_0, class_1)

    train_loader = DataLoader(train_binary, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_binary, batch_size=batch_size, shuffle=False)

    print(f"Binary MNIST loaded (classes {class_0} vs {class_1}):")
    print(f"  Training samples: {len(train_binary)}")
    print(f"  Test samples: {len(test_binary)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing full MNIST loader:")
    train_loader, test_loader = load_mnist(batch_size=64)

    # Check a batch
    data, labels = next(iter(train_loader))
    print(f"\nBatch shape: {data.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Sample labels: {labels[:10].tolist()}")

    print("\n" + "="*50 + "\n")

    # Test binary MNIST
    print("Testing binary MNIST loader (0 vs 1):")
    train_binary, test_binary = create_binary_mnist(batch_size=64, class_0=0, class_1=1)

    data, labels = next(iter(train_binary))
    print(f"\nBatch shape: {data.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Sample labels: {labels[:10].tolist()}")
