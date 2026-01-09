"""Tests for model inference and dashboard backend (Epic 4, Story 4.1)."""

import numpy as np
import pytest
import torch
from PIL import Image

from eurosat_vit_analysis.inference import (
    InferenceResult,
    ModelInfo,
    load_model_for_inference,
    predict_single_image,
    preprocess_image,
)

# EuroSAT class names (fixed for this dataset)
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


# ============================================================================
# Tests for preprocess_image
# ============================================================================


def test_preprocess_image_pil():
    """Test preprocessing a PIL Image."""
    # Create a simple RGB image
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))

    tensor = preprocess_image(img)

    # Should return a 4D tensor (batch_size=1, channels=3, height=224, width=224)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_image_numpy():
    """Test preprocessing a NumPy array (H, W, C format)."""
    # Create a simple RGB image as numpy array
    img_array = np.ones((64, 64, 3), dtype=np.uint8) * 128

    tensor = preprocess_image(img_array)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_image_resizes():
    """Test that preprocessing resizes images to 224x224."""
    # Create images of different sizes
    small_img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    large_img = Image.new("RGB", (512, 512), color=(100, 100, 100))

    small_tensor = preprocess_image(small_img)
    large_tensor = preprocess_image(large_img)

    # Both should be resized to 224x224
    assert small_tensor.shape == (1, 3, 224, 224)
    assert large_tensor.shape == (1, 3, 224, 224)


def test_preprocess_image_normalizes():
    """Test that preprocessing applies ImageNet normalization."""
    # Create a white image (255, 255, 255)
    white_img = Image.new("RGB", (64, 64), color=(255, 255, 255))

    tensor = preprocess_image(white_img)

    # After normalization with ImageNet stats, values should not be 1.0
    # ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # For white (1.0 after ToTensor), normalized = (1.0 - mean) / std
    assert not torch.allclose(tensor, torch.ones_like(tensor))

    # Check that values are in a reasonable range after normalization
    # For white pixels: max channel = (1.0 - 0.406) / 0.225 ≈ 2.64
    assert tensor.max() < 3.0
    assert tensor.min() > -3.0


# ============================================================================
# Tests for ModelInfo
# ============================================================================


def test_model_info_creation():
    """Test creating ModelInfo dataclass."""
    info = ModelInfo(
        name="vit_base",
        display_name="ViT-Base",
        num_params=86_000_000,
        architecture="Vision Transformer",
    )

    assert info.name == "vit_base"
    assert info.display_name == "ViT-Base"
    assert info.num_params == 86_000_000
    assert info.architecture == "Vision Transformer"


def test_model_info_immutable():
    """Test that ModelInfo is immutable (frozen dataclass)."""
    info = ModelInfo(
        name="resnet50",
        display_name="ResNet-50",
        num_params=25_000_000,
        architecture="CNN",
    )

    with pytest.raises(AttributeError):
        info.name = "modified"


# ============================================================================
# Tests for InferenceResult
# ============================================================================


def test_inference_result_creation():
    """Test creating InferenceResult dataclass."""
    result = InferenceResult(
        predictions=["Forest", "AnnualCrop", "Pasture"],
        confidences=[0.85, 0.10, 0.03],
        all_probabilities=[0.01, 0.85, 0.03, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.01],
        inference_time_ms=45.2,
        model_info=ModelInfo("vit_base", "ViT-Base", 86_000_000, "Transformer"),
    )

    assert len(result.predictions) == 3
    assert len(result.confidences) == 3
    assert len(result.all_probabilities) == 10  # EuroSAT has 10 classes
    assert result.inference_time_ms > 0
    assert result.model_info.name == "vit_base"


def test_inference_result_top_k_consistent():
    """Test that predictions and confidences are top-k sorted."""
    result = InferenceResult(
        predictions=["Forest", "AnnualCrop", "Pasture"],
        confidences=[0.85, 0.10, 0.03],
        all_probabilities=[0.01, 0.85, 0.03, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.01],
        inference_time_ms=45.2,
        model_info=ModelInfo("vit_base", "ViT-Base", 86_000_000, "Transformer"),
    )

    # Confidences should be in descending order
    assert result.confidences[0] >= result.confidences[1]
    assert result.confidences[1] >= result.confidences[2]

    # Sum of all probabilities should be ~1.0
    assert abs(sum(result.all_probabilities) - 1.0) < 0.01


def test_inference_result_immutable():
    """Test that InferenceResult is immutable."""
    result = InferenceResult(
        predictions=["Forest"],
        confidences=[0.9],
        all_probabilities=[0.0] * 10,
        inference_time_ms=50.0,
        model_info=ModelInfo("vit_base", "ViT", 86_000_000, "Transformer"),
    )

    with pytest.raises((AttributeError, TypeError)):
        result.predictions = ["Modified"]


# ============================================================================
# Tests for load_model_for_inference
# ============================================================================


def test_load_model_for_inference_vit():
    """Test loading ViT model for inference without checkpoint."""
    model, info = load_model_for_inference("vit_base", checkpoint_path=None)

    # Model should be in eval mode
    assert not model.training

    # Check model info
    assert info.name == "vit_base"
    assert info.display_name == "ViT-Base"
    assert info.num_params > 0
    assert "Transformer" in info.architecture or "ViT" in info.architecture

    # Model should accept 224x224 images
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 10)  # 10 classes for EuroSAT


def test_load_model_for_inference_resnet():
    """Test loading ResNet model for inference."""
    model, info = load_model_for_inference("resnet50", checkpoint_path=None)

    assert not model.training
    assert info.name == "resnet50"
    assert info.display_name == "ResNet-50"
    assert "ResNet" in info.architecture or "CNN" in info.architecture

    # Model should accept 224x224 images
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 10)


def test_load_model_for_inference_invalid_name():
    """Test that invalid model name raises ValueError."""
    with pytest.raises(ValueError, match="Model .* not supported"):
        load_model_for_inference("invalid_model_name")


def test_load_model_for_inference_with_checkpoint(tmp_path):
    """Test loading model with a checkpoint."""
    # Create a dummy checkpoint
    model, _ = load_model_for_inference("vit_base", checkpoint_path=None)
    checkpoint_path = tmp_path / "dummy_checkpoint.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
        },
        checkpoint_path,
    )

    # Load with checkpoint
    loaded_model, info = load_model_for_inference(
        "vit_base", checkpoint_path=str(checkpoint_path)
    )

    assert not loaded_model.training
    assert info.name == "vit_base"


def test_load_model_for_inference_counts_params():
    """Test that parameter counting is accurate."""
    model, info = load_model_for_inference("vit_base", checkpoint_path=None)

    # Count parameters manually
    total_params = sum(p.numel() for p in model.parameters())

    assert info.num_params == total_params
    assert info.num_params > 1_000_000  # ViT-Base should have millions of params


# ============================================================================
# Tests for predict_single_image
# ============================================================================


def test_predict_single_image_basic():
    """Test basic single image prediction."""
    # Load a model
    model, model_info = load_model_for_inference("vit_base", checkpoint_path=None)

    # Create a dummy image
    dummy_img = Image.new("RGB", (64, 64), color=(100, 150, 200))

    # Run inference
    device = torch.device("cpu")
    result = predict_single_image(model, dummy_img, model_info, device, top_k=3)

    # Check result structure
    assert isinstance(result, InferenceResult)
    assert len(result.predictions) == 3
    assert len(result.confidences) == 3
    assert len(result.all_probabilities) == 10
    assert result.inference_time_ms > 0
    assert result.model_info == model_info


def test_predict_single_image_top_k():
    """Test that top_k parameter controls number of predictions."""
    model, model_info = load_model_for_inference("resnet50", checkpoint_path=None)
    dummy_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    device = torch.device("cpu")

    # Test different top_k values
    for k in [1, 3, 5, 10]:
        result = predict_single_image(model, dummy_img, model_info, device, top_k=k)
        assert len(result.predictions) == k
        assert len(result.confidences) == k


def test_predict_single_image_probabilities_valid():
    """Test that predicted probabilities are valid."""
    model, model_info = load_model_for_inference("vit_base", checkpoint_path=None)
    dummy_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    device = torch.device("cpu")

    result = predict_single_image(model, dummy_img, model_info, device)

    # All probabilities should be between 0 and 1
    for prob in result.all_probabilities:
        assert 0.0 <= prob <= 1.0

    # Confidences should be between 0 and 1
    for conf in result.confidences:
        assert 0.0 <= conf <= 1.0

    # Sum of all probabilities should be ~1.0
    assert abs(sum(result.all_probabilities) - 1.0) < 0.01


def test_predict_single_image_sorted():
    """Test that predictions are sorted by confidence (descending)."""
    model, model_info = load_model_for_inference("vit_base", checkpoint_path=None)
    dummy_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    device = torch.device("cpu")

    result = predict_single_image(model, dummy_img, model_info, device, top_k=5)

    # Confidences should be in descending order
    for i in range(len(result.confidences) - 1):
        assert result.confidences[i] >= result.confidences[i + 1]


def test_predict_single_image_class_names():
    """Test that predictions use correct EuroSAT class names."""
    model, model_info = load_model_for_inference("vit_base", checkpoint_path=None)
    dummy_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    device = torch.device("cpu")

    result = predict_single_image(model, dummy_img, model_info, device, top_k=10)

    # All predictions should be valid EuroSAT class names
    for pred in result.predictions:
        assert pred in EUROSAT_CLASSES


def test_predict_single_image_timing():
    """Test that inference time is reasonable."""
    model, model_info = load_model_for_inference("resnet50", checkpoint_path=None)
    dummy_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    device = torch.device("cpu")

    result = predict_single_image(model, dummy_img, model_info, device)

    # Inference time should be positive and reasonable (< 5 seconds on CPU)
    assert result.inference_time_ms > 0
    assert result.inference_time_ms < 5000


def test_predict_single_image_numpy_input():
    """Test prediction with NumPy array input."""
    model, model_info = load_model_for_inference("vit_base", checkpoint_path=None)

    # Create numpy array image
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    device = torch.device("cpu")
    result = predict_single_image(model, img_array, model_info, device)

    assert isinstance(result, InferenceResult)
    assert len(result.predictions) > 0
    assert result.inference_time_ms > 0


def test_predict_single_image_deterministic():
    """Test that same input produces same output (with eval mode)."""
    model, model_info = load_model_for_inference("vit_base", checkpoint_path=None)
    dummy_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    device = torch.device("cpu")

    # Run inference twice
    result1 = predict_single_image(model, dummy_img, model_info, device)
    result2 = predict_single_image(model, dummy_img, model_info, device)

    # Results should be identical (model is in eval mode, no dropout)
    assert result1.predictions == result2.predictions
    assert result1.confidences == result2.confidences
    assert result1.all_probabilities == result2.all_probabilities
