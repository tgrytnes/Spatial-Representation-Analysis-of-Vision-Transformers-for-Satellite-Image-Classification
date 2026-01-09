"""
Model inference utilities for the Streamlit dashboard (Epic 4, Story 4.1).

This module provides functionality for:
- Loading trained models for inference
- Preprocessing images for model input
- Running single-image inference with timing
- Formatting results for UI display
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from eurosat_vit_analysis.models import create_model

# EuroSAT class names (alphabetically sorted as in ImageFolder)
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


@dataclass(frozen=True)
class ModelInfo:
    """Information about a loaded model."""

    name: str  # Model architecture name (e.g., "vit_base")
    display_name: str  # Human-readable name (e.g., "ViT-Base")
    num_params: int  # Total number of parameters
    architecture: str  # Architecture family (e.g., "Vision Transformer")


@dataclass(frozen=True)
class InferenceResult:
    """Result from single-image inference."""

    predictions: list[str]  # Top-k predicted class names
    confidences: list[float]  # Top-k confidence scores
    all_probabilities: list[float]  # All class probabilities (for charts)
    inference_time_ms: float  # Inference time in milliseconds
    model_info: ModelInfo  # Information about the model used


def get_model_display_info(model_name: str) -> tuple[str, str]:
    """
    Get human-readable display name and architecture family for a model.

    Args:
        model_name: Model architecture name (e.g., "vit_base")

    Returns:
        Tuple of (display_name, architecture_family)
    """
    model_info_map = {
        "vit_base": ("ViT-Base", "Vision Transformer"),
        "swin_t": ("Swin-Tiny", "Swin Transformer"),
        "resnet50": ("ResNet-50", "Residual CNN"),
        "convnext_t": ("ConvNeXt-Tiny", "ConvNext CNN"),
    }

    if model_name not in model_info_map:
        # Fallback for unknown models
        return (model_name.replace("_", "-").title(), "Unknown")

    return model_info_map[model_name]


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def preprocess_image(image: Image.Image | np.ndarray) -> torch.Tensor:
    """
    Preprocess an image for model inference.

    Applies:
    - Conversion from PIL/numpy to tensor
    - Resize to 224x224
    - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    Args:
        image: Input image (PIL Image or numpy array in HWC format)

    Returns:
        Preprocessed image tensor (1, 3, 224, 224)
    """
    # Define ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    return tensor


def load_model_for_inference(
    model_name: str,
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, ModelInfo]:
    """
    Load a model for inference.

    Args:
        model_name: Model architecture name (e.g., "vit_base", "resnet50")
        checkpoint_path: Optional path to checkpoint file
        device: Device to load model on (defaults to CPU)

    Returns:
        Tuple of (model, model_info)

    Raises:
        ValueError: If model_name is not supported
        FileNotFoundError: If checkpoint_path doesn't exist
    """
    if device is None:
        device = torch.device("cpu")

    # Create model (pretrained=True if no checkpoint, False if loading checkpoint)
    pretrained = checkpoint_path is None
    model = create_model(model_name, num_classes=10, pretrained=pretrained)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Get model information
    display_name, architecture = get_model_display_info(model_name)
    num_params = count_parameters(model)

    model_info = ModelInfo(
        name=model_name,
        display_name=display_name,
        num_params=num_params,
        architecture=architecture,
    )

    return model, model_info


def predict_single_image(
    model: nn.Module,
    image: Image.Image | np.ndarray,
    model_info: ModelInfo,
    device: torch.device,
    top_k: int = 3,
) -> InferenceResult:
    """
    Run inference on a single image.

    Args:
        model: PyTorch model in eval mode
        image: Input image (PIL Image or numpy array)
        model_info: Information about the model
        device: Device to run inference on
        top_k: Number of top predictions to return

    Returns:
        InferenceResult with predictions, confidences, and timing
    """
    # Preprocess image
    input_tensor = preprocess_image(image).to(device)

    # Run inference with timing
    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000

    # Get all probabilities
    all_probs = probabilities[0].cpu().numpy().tolist()

    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probabilities[0], k=top_k)
    top_k_probs = top_k_probs.cpu().numpy().tolist()
    top_k_indices = top_k_indices.cpu().numpy().tolist()

    # Map indices to class names
    predictions = [EUROSAT_CLASSES[idx] for idx in top_k_indices]
    confidences = top_k_probs

    return InferenceResult(
        predictions=predictions,
        confidences=confidences,
        all_probabilities=all_probs,
        inference_time_ms=inference_time_ms,
        model_info=model_info,
    )
