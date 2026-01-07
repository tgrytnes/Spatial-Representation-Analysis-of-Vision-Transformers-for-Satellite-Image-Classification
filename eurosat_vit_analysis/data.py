from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

AugmentationStrength = Literal["none", "light", "medium", "strong"]


def get_transforms(
    image_size: int = 224, strength: AugmentationStrength = "none"
) -> transforms.Compose:
    """
    Get transforms for EuroSAT dataset based on augmentation strength.
    Using ImageNet normalization statistics as we typically use pretrained models.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Base transforms (always applied at the end)
    normalize = transforms.Normalize(mean=mean, std=std)
    base_transforms = [
        transforms.ToTensor(),
        normalize,
    ]

    if strength == "none":
        # Just resize and normalize
        transform_list = [transforms.Resize((image_size, image_size))] + base_transforms

    elif strength == "light":
        # Resize + Horizontal Flip
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ] + base_transforms

    elif strength == "medium":
        # RandomResizedCrop (less aggressive) + Flip
        transform_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ] + base_transforms

    elif strength == "strong":
        # RandAugment (standard for modern ViT training) + Resize
        # Note: RandAugment inputs PIL, outputs PIL
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandAugment(num_ops=2, magnitude=9),
        ] + base_transforms

    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")

    return transforms.Compose(transform_list)


def prepare_data(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    augmentation: AugmentationStrength = "none",
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Prepare DataLoaders for EuroSAT.

    Args:
        data_dir: Path to dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        val_split: Validation split fraction
        seed: Random seed for splitting
        augmentation: Augmentation strength ('none', 'light', 'medium', 'strong')
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at {data_path}")

    # Training transform (with augmentation)
    train_transform = get_transforms(strength=augmentation)
    # Validation transform (always none/deterministic)
    val_transform = get_transforms(strength="none")

    # We load the full dataset twice to apply different transforms
    # Efficiency: Loading ImageFolder twice is negligible for EuroSAT (~27k images).

    # Efficiency: Loading ImageFolder twice is negligible for EuroSAT (~27k images).

    full_dataset_train = datasets.ImageFolder(
        root=str(data_path), transform=train_transform
    )
    full_dataset_val = datasets.ImageFolder(
        root=str(data_path), transform=val_transform
    )

    class_names = full_dataset_train.classes
    total_len = len(full_dataset_train)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    # Use the same seed to ensure indices match between the two dataset copies
    generator = torch.Generator().manual_seed(seed)

    # We get the INDICES for the split
    train_subset, _ = random_split(
        full_dataset_train, [train_len, val_len], generator=generator
    )
    _, val_subset = random_split(
        full_dataset_val, [train_len, val_len], generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names
