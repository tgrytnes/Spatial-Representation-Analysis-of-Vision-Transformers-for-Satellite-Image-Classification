from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PatchShuffleReport:
    """Report for patch-shuffle robustness evaluation."""

    clean_accuracy: float
    shuffled_accuracy: float
    accuracy_drop: float
    per_class_clean_accuracy: dict[int, float]
    per_class_shuffled_accuracy: dict[int, float]
    per_class_accuracy_drop: dict[int, float]


def shuffle_patches(
    images: torch.Tensor, grid_size: tuple[int, int], seed: int | None = None
) -> torch.Tensor:
    """
    Shuffle spatial patches of images to test spatial reasoning.

    Args:
        images: Batch of images with shape (B, C, H, W)
        grid_size: Tuple (grid_h, grid_w) defining the patch grid
        seed: Optional random seed for reproducibility

    Returns:
        Shuffled images with same shape as input
    """
    B, C, H, W = images.shape
    grid_h, grid_w = grid_size

    if H % grid_h != 0 or W % grid_w != 0:
        raise ValueError(
            f"Image dimensions ({H}, {W}) must be divisible by "
            f"grid_size ({grid_h}, {grid_w})"
        )

    patch_h = H // grid_h
    patch_w = W // grid_w

    # Reshape to separate patches: (B, C, grid_h, patch_h, grid_w, patch_w)
    patches = images.reshape(B, C, grid_h, patch_h, grid_w, patch_w)

    # Permute to group patches: (B, C, grid_h, grid_w, patch_h, patch_w)
    patches = patches.permute(0, 1, 2, 4, 3, 5)

    # Reshape to treat patches as a flat dimension:
    # (B, C, grid_h * grid_w, patch_h, patch_w)
    patches = patches.reshape(B, C, grid_h * grid_w, patch_h, patch_w)

    # Shuffle patches independently for each image in batch
    if seed is not None:
        generator = torch.Generator(device=images.device).manual_seed(seed)
    else:
        generator = None

    shuffled_patches = torch.zeros_like(patches)
    for b in range(B):
        # Generate random permutation for this image
        perm = torch.randperm(
            grid_h * grid_w, generator=generator, device=images.device
        )
        shuffled_patches[b] = patches[b, :, perm]

    # Reshape back: (B, C, grid_h, grid_w, patch_h, patch_w)
    shuffled_patches = shuffled_patches.reshape(B, C, grid_h, grid_w, patch_h, patch_w)

    # Permute back: (B, C, grid_h, patch_h, grid_w, patch_w)
    shuffled_patches = shuffled_patches.permute(0, 1, 2, 4, 3, 5)

    # Reshape to original image shape: (B, C, H, W)
    shuffled_images = shuffled_patches.reshape(B, C, H, W)

    return shuffled_images


def evaluate_patch_shuffle(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    grid_size: tuple[int, int],
    num_classes: int,
    seed: int | None = None,
) -> PatchShuffleReport:
    """
    Evaluate model robustness to patch shuffling.

    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation
        device: Device to run evaluation on
        grid_size: Tuple (grid_h, grid_w) defining the patch grid
        num_classes: Number of classes in the dataset
        seed: Optional random seed for reproducibility

    Returns:
        PatchShuffleReport with overall and per-class accuracy metrics
    """
    model.eval()

    clean_correct = 0
    shuffled_correct = 0
    total = 0

    per_class_clean_correct = {i: 0 for i in range(num_classes)}
    per_class_shuffled_correct = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Evaluate on clean images
            clean_outputs = model(inputs)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_correct += clean_preds.eq(targets).sum().item()

            # Evaluate on shuffled images
            shuffled_inputs = shuffle_patches(inputs, grid_size, seed=seed)
            shuffled_outputs = model(shuffled_inputs)
            shuffled_preds = shuffled_outputs.argmax(dim=1)
            shuffled_correct += shuffled_preds.eq(targets).sum().item()

            # Track per-class accuracy
            for i in range(num_classes):
                class_mask = targets == i
                class_count = class_mask.sum().item()

                if class_count > 0:
                    per_class_total[i] += class_count
                    per_class_clean_correct[i] += (
                        clean_preds[class_mask].eq(targets[class_mask]).sum().item()
                    )
                    per_class_shuffled_correct[i] += (
                        shuffled_preds[class_mask].eq(targets[class_mask]).sum().item()
                    )

            total += targets.size(0)

    # Calculate overall accuracy
    clean_accuracy = clean_correct / total if total > 0 else 0.0
    shuffled_accuracy = shuffled_correct / total if total > 0 else 0.0
    accuracy_drop = clean_accuracy - shuffled_accuracy

    # Calculate per-class accuracy
    per_class_clean_accuracy = {
        i: per_class_clean_correct[i] / per_class_total[i]
        if per_class_total[i] > 0
        else 0.0
        for i in range(num_classes)
    }
    per_class_shuffled_accuracy = {
        i: per_class_shuffled_correct[i] / per_class_total[i]
        if per_class_total[i] > 0
        else 0.0
        for i in range(num_classes)
    }
    per_class_accuracy_drop = {
        i: per_class_clean_accuracy[i] - per_class_shuffled_accuracy[i]
        for i in range(num_classes)
    }

    return PatchShuffleReport(
        clean_accuracy=clean_accuracy,
        shuffled_accuracy=shuffled_accuracy,
        accuracy_drop=accuracy_drop,
        per_class_clean_accuracy=per_class_clean_accuracy,
        per_class_shuffled_accuracy=per_class_shuffled_accuracy,
        per_class_accuracy_drop=per_class_accuracy_drop,
    )


def occlusion_sensitivity(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    occlusion_size: int = 16,
    stride: int = 8,
) -> torch.Tensor:
    """
    Generate occlusion sensitivity maps for images.

    Args:
        model: Model to evaluate
        images: Batch of images with shape (B, C, H, W)
        targets: Target class indices with shape (B,)
        device: Device to run evaluation on
        occlusion_size: Size of the occlusion square
        stride: Stride for sliding the occlusion window

    Returns:
        Sensitivity maps with shape (B, H_out, W_out) where each value
        represents the drop in target class probability when that region is occluded
    """
    model.eval()
    B, C, H, W = images.shape
    images = images.to(device)
    targets = targets.to(device)

    # Get baseline predictions
    with torch.no_grad():
        baseline_outputs = model(images)
        baseline_probs = torch.softmax(baseline_outputs, dim=1)
        # Get probability of target class for each image
        baseline_target_probs = baseline_probs[torch.arange(B), targets]

    # Calculate output dimensions
    h_steps = (H - occlusion_size) // stride + 1
    w_steps = (W - occlusion_size) // stride + 1

    # Initialize sensitivity maps
    sensitivity_maps = torch.zeros(B, h_steps, w_steps, device=device)

    # Slide occlusion window
    for i in range(h_steps):
        for j in range(w_steps):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + occlusion_size
            w_end = w_start + occlusion_size

            # Create occluded images
            occluded_images = images.clone()
            occluded_images[:, :, h_start:h_end, w_start:w_end] = 0

            # Get predictions on occluded images
            with torch.no_grad():
                occluded_outputs = model(occluded_images)
                occluded_probs = torch.softmax(occluded_outputs, dim=1)
                occluded_target_probs = occluded_probs[torch.arange(B), targets]

            # Calculate sensitivity (drop in probability)
            sensitivity_maps[:, i, j] = baseline_target_probs - occluded_target_probs

    return sensitivity_maps


__all__ = [
    "PatchShuffleReport",
    "shuffle_patches",
    "evaluate_patch_shuffle",
    "occlusion_sensitivity",
]
