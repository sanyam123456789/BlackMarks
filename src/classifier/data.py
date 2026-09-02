"""
CIFAR-10 Data Foundation Module for BlackMarks

Provides consistent, reusable data loading, standard normalization,
and DataLoader creation across all stages of the BlackMarks project
(clean training, watermark generation, testing, and black-box inference).
"""

import os
from typing import Tuple, List, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 standard class names in canonical index order (0 to 9)
CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Standard CIFAR-10 channel-wise normalization constants
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def get_cifar10_transform(normalize: bool = True) -> transforms.Compose:
    """
    Returns the standard deterministic preprocessing transform for CIFAR-10.

    Converts 32x32 PIL image to torch.FloatTensor in range [0, 1] and
    optionally applies standard channel-wise mean/std normalization.

    Args:
        normalize: If True, applies standard CIFAR-10 mean/std normalization.

    Returns:
        torchvision.transforms.Compose transform pipeline.
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD))
    return transforms.Compose(transform_list)


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = CIFAR10_MEAN,
    std: Tuple[float, float, float] = CIFAR10_STD,
) -> torch.Tensor:
    """
    Reverses CIFAR-10 normalization for visualization and inspection.

    Args:
        tensor: Normalized image tensor of shape (C, H, W) or (B, C, H, W).
        mean: Tuple of channel means used in normalization.
        std: Tuple of channel standard deviations used in normalization.

    Returns:
        Tensor in range [0.0, 1.0] with values clamped.
    """
    mean_t = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    denorm = tensor * std_t + mean_t
    return torch.clamp(denorm, 0.0, 1.0)


def get_cifar10_datasets(
    data_dir: str = "./data",
    download: bool = True,
    normalize: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Downloads (if necessary) and returns train and test CIFAR-10 dataset instances.

    Args:
        data_dir: Root directory where CIFAR-10 dataset is stored.
        download: Whether to download CIFAR-10 if not present in data_dir.
        normalize: Whether to apply standard mean/std normalization.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    os.makedirs(data_dir, exist_ok=True)
    transform = get_cifar10_transform(normalize=normalize)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    return train_dataset, test_dataset


def get_cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    normalize: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns PyTorch DataLoaders for CIFAR-10 training and testing.

    Args:
        data_dir: Root directory where CIFAR-10 is stored.
        batch_size: Mini-batch size for loading.
        num_workers: Number of subprocess workers for data loading.
        download: Whether to download the dataset if missing.
        normalize: Whether to apply standard normalization.
        pin_memory: If True, copies Tensors into CUDA pinned memory.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        download=download,
        normalize=normalize,
    )

    # Windows safe num_workers handling
    if os.name == "nt" and num_workers > 0:
        # Avoid multi-process spawn issues when run in non-main guards
        effective_workers = num_workers
    else:
        effective_workers = num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader
