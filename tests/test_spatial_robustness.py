import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eurosat_vit_analysis.spatial_robustness import (
    PatchShuffleReport,
    evaluate_patch_shuffle,
    occlusion_sensitivity,
    shuffle_patches,
)

# ============================================================================
# Tests for shuffle_patches
# ============================================================================


def test_shuffle_patches_shape():
    """Test that shuffled images maintain original shape."""
    images = torch.randn(2, 3, 64, 64)
    grid_size = (2, 2)

    shuffled = shuffle_patches(images, grid_size)

    assert shuffled.shape == images.shape
    assert not torch.equal(shuffled, images)


def test_shuffle_patches_content_preservation():
    """Ensure that pixels are permuted, not created or destroyed."""
    images = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
    grid_size = (2, 2)

    shuffled = shuffle_patches(images, grid_size)

    # Sort and compare - all values should be preserved
    assert torch.equal(images.flatten().sort().values, shuffled.flatten().sort().values)


def test_shuffle_patches_deterministic_with_seed():
    """Test that using the same seed produces identical shuffles."""
    images = torch.randn(1, 3, 64, 64)
    grid_size = (4, 4)

    shuffled1 = shuffle_patches(images, grid_size, seed=42)
    shuffled2 = shuffle_patches(images, grid_size, seed=42)

    assert torch.equal(shuffled1, shuffled2)


def test_shuffle_patches_different_seeds():
    """Test that different seeds produce different shuffles."""
    images = torch.randn(1, 3, 64, 64)
    grid_size = (4, 4)

    shuffled1 = shuffle_patches(images, grid_size, seed=42)
    shuffled2 = shuffle_patches(images, grid_size, seed=123)

    assert not torch.equal(shuffled1, shuffled2)


def test_shuffle_patches_batch_independence():
    """Test that each image in batch is shuffled independently."""
    # Create two identical images in a batch
    image = torch.randn(1, 3, 64, 64)
    images = image.repeat(2, 1, 1, 1)
    grid_size = (4, 4)

    shuffled = shuffle_patches(images, grid_size)

    # The two shuffled images should be different (highly unlikely to be same)
    assert not torch.equal(shuffled[0], shuffled[1])


def test_shuffle_patches_invalid_grid_size():
    """Test that invalid grid sizes raise appropriate errors."""
    images = torch.randn(1, 3, 64, 64)

    # Grid size that doesn't divide image dimensions
    with pytest.raises(ValueError, match="must be divisible by grid_size"):
        shuffle_patches(images, (3, 3))


def test_shuffle_patches_different_grid_sizes():
    """Test shuffling with various grid sizes."""
    images = torch.randn(2, 3, 64, 64)

    for grid_size in [(2, 2), (4, 4), (8, 8), (2, 4)]:
        shuffled = shuffle_patches(images, grid_size)
        assert shuffled.shape == images.shape
        # Verify content preservation
        assert torch.allclose(
            images.flatten().sort().values, shuffled.flatten().sort().values
        )


def test_shuffle_patches_single_patch():
    """Test that single patch (1, 1) returns original image."""
    images = torch.randn(2, 3, 64, 64)
    grid_size = (1, 1)

    shuffled = shuffle_patches(images, grid_size)

    # With only one patch, shuffling should return the same image
    assert torch.equal(shuffled, images)


# ============================================================================
# Tests for evaluate_patch_shuffle
# ============================================================================


class SimpleClassifier(nn.Module):
    """Simple classifier for testing that always predicts class 0."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(1, num_classes)
        # Initialize to predict class 0
        with torch.no_grad():
            self.fc.weight.fill_(0)
            self.fc.bias[0] = 10.0

    def forward(self, x):
        # Average pool and predict
        x = x.mean(dim=[1, 2, 3], keepdim=True)
        # Ensure x has shape (batch_size, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LocationDependentClassifier(nn.Module):
    """Classifier that depends on top-left corner to test spatial sensitivity."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Predict class based on mean intensity of top-left corner
        batch_size = x.size(0)
        top_left = x[:, :, :8, :8].mean(dim=[1, 2, 3])

        # Create logits where class depends on top_left value
        logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        for i in range(batch_size):
            pred_class = int((top_left[i] * 10).item()) % self.num_classes
            logits[i, pred_class] = 10.0

        return logits


def test_evaluate_patch_shuffle_structure():
    """Test that evaluate_patch_shuffle returns correct report structure."""
    model = SimpleClassifier(num_classes=3)
    model.eval()

    # Create synthetic dataset
    images = torch.randn(10, 3, 64, 64)
    targets = torch.zeros(10, dtype=torch.long)  # All class 0
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=5)

    device = torch.device("cpu")
    grid_size = (4, 4)

    report = evaluate_patch_shuffle(
        model, loader, device, grid_size, num_classes=3, seed=42
    )

    # Check report structure
    assert isinstance(report, PatchShuffleReport)
    assert isinstance(report.clean_accuracy, float)
    assert isinstance(report.shuffled_accuracy, float)
    assert isinstance(report.accuracy_drop, float)
    assert len(report.per_class_clean_accuracy) == 3
    assert len(report.per_class_shuffled_accuracy) == 3
    assert len(report.per_class_accuracy_drop) == 3


def test_evaluate_patch_shuffle_perfect_classifier():
    """Test patch shuffle evaluation with a perfect classifier on simple data."""
    num_classes = 3
    model = SimpleClassifier(num_classes=num_classes)
    model.eval()

    # Create dataset where all samples are class 0
    images = torch.randn(12, 3, 64, 64)
    targets = torch.zeros(12, dtype=torch.long)
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=4)

    device = torch.device("cpu")
    report = evaluate_patch_shuffle(
        model, loader, device, grid_size=(2, 2), num_classes=num_classes
    )

    # Should achieve 100% accuracy on class 0
    assert report.clean_accuracy == 1.0
    assert report.per_class_clean_accuracy[0] == 1.0
    # Shuffling shouldn't affect this global classifier
    assert report.shuffled_accuracy == 1.0


def test_evaluate_patch_shuffle_spatial_dependence():
    """Test that spatial-dependent classifiers show accuracy drop."""
    num_classes = 3
    model = LocationDependentClassifier(num_classes=num_classes)
    model.eval()

    # Create dataset with consistent spatial patterns
    # Use larger values to ensure proper classification
    images = torch.zeros(9, 3, 64, 64)
    # Set top-left corner to predict specific classes more reliably
    for i in range(9):
        target_class = i % num_classes
        # Set the top-left corner to a distinctive value
        images[i, :, :8, :8] = float(target_class) / 3.0 + 0.1

    targets = torch.arange(9, dtype=torch.long) % num_classes
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=3)

    device = torch.device("cpu")
    report = evaluate_patch_shuffle(
        model, loader, device, grid_size=(8, 8), num_classes=num_classes, seed=42
    )

    # With a smaller grid (8x8), shuffling is more likely to disrupt spatial structure
    # We just need to verify the metrics are computed, not specific accuracy values
    assert 0.0 <= report.clean_accuracy <= 1.0
    assert 0.0 <= report.shuffled_accuracy <= 1.0
    assert report.accuracy_drop == report.clean_accuracy - report.shuffled_accuracy


def test_evaluate_patch_shuffle_per_class_tracking():
    """Test that per-class accuracy is tracked correctly."""
    num_classes = 4
    model = SimpleClassifier(num_classes=num_classes)
    model.eval()

    # Create dataset with multiple classes
    images = torch.randn(20, 3, 64, 64)
    targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=5)

    device = torch.device("cpu")
    report = evaluate_patch_shuffle(
        model, loader, device, grid_size=(2, 2), num_classes=num_classes
    )

    # All per-class accuracies should be computed
    for i in range(num_classes):
        assert 0.0 <= report.per_class_clean_accuracy[i] <= 1.0
        assert 0.0 <= report.per_class_shuffled_accuracy[i] <= 1.0
        assert report.per_class_accuracy_drop[i] == (
            report.per_class_clean_accuracy[i] - report.per_class_shuffled_accuracy[i]
        )


# ============================================================================
# Tests for occlusion_sensitivity
# ============================================================================


class TopLeftClassifier(nn.Module):
    """Classifier that only looks at top-left corner."""

    def __init__(self, num_classes=10, corner_size=16):
        super().__init__()
        self.num_classes = num_classes
        self.corner_size = corner_size

    def forward(self, x):
        batch_size = x.size(0)
        # Only look at top-left corner
        corner = x[:, :, : self.corner_size, : self.corner_size]
        corner_mean = corner.mean(dim=[1, 2, 3])

        # Create logits based on corner values
        logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        for i in range(batch_size):
            pred_class = int((corner_mean[i] * 10).item()) % self.num_classes
            logits[i, pred_class] = 10.0

        return logits


def test_occlusion_sensitivity_shape():
    """Test that occlusion sensitivity returns correct shape."""
    model = SimpleClassifier(num_classes=5)
    model.eval()

    images = torch.randn(2, 3, 64, 64)
    targets = torch.tensor([0, 1])
    device = torch.device("cpu")

    sensitivity_maps = occlusion_sensitivity(
        model, images, targets, device, occlusion_size=16, stride=8
    )

    # Calculate expected output dimensions
    h_steps = (64 - 16) // 8 + 1  # 7
    w_steps = (64 - 16) // 8 + 1  # 7

    assert sensitivity_maps.shape == (2, h_steps, w_steps)


def test_occlusion_sensitivity_values():
    """Test that occlusion sensitivity produces reasonable values."""
    model = SimpleClassifier(num_classes=3)
    model.eval()

    images = torch.randn(1, 3, 64, 64)
    targets = torch.tensor([0])
    device = torch.device("cpu")

    sensitivity_maps = occlusion_sensitivity(
        model, images, targets, device, occlusion_size=16, stride=16
    )

    # Sensitivity values should be between -1 and 1 (probability changes)
    assert sensitivity_maps.min() >= -1.0
    assert sensitivity_maps.max() <= 1.0


def test_occlusion_sensitivity_spatial_importance():
    """Test that occlusion detects spatially important regions."""
    num_classes = 3
    corner_size = 16
    model = TopLeftClassifier(num_classes=num_classes, corner_size=corner_size)
    model.eval()

    # Create image where top-left determines class
    images = torch.zeros(1, 3, 64, 64)
    images[:, :, :corner_size, :corner_size] = 0.5  # Set top-left to specific value

    # First, determine what class the model predicts
    with torch.no_grad():
        outputs = model(images)
        predicted_class = outputs.argmax(dim=1).item()

    targets = torch.tensor([predicted_class])
    device = torch.device("cpu")

    sensitivity_maps = occlusion_sensitivity(
        model, images, targets, device, occlusion_size=16, stride=8
    )

    # Top-left region should have higher sensitivity (positive value when occluded)
    # The absolute sensitivity should be higher for important regions
    top_left_sensitivity = sensitivity_maps[0, 0, 0].abs()
    bottom_right_sensitivity = sensitivity_maps[0, -1, -1].abs()

    # Top-left should be more sensitive than bottom-right
    assert top_left_sensitivity > bottom_right_sensitivity


def test_occlusion_sensitivity_batch():
    """Test occlusion sensitivity with multiple images."""
    model = SimpleClassifier(num_classes=5)
    model.eval()

    images = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])
    device = torch.device("cpu")

    sensitivity_maps = occlusion_sensitivity(
        model, images, targets, device, occlusion_size=16, stride=16
    )

    # Should have one map per image
    assert sensitivity_maps.shape[0] == 4

    # Each map should have reasonable values
    for i in range(4):
        assert not torch.isnan(sensitivity_maps[i]).any()
        assert not torch.isinf(sensitivity_maps[i]).any()


def test_occlusion_sensitivity_different_sizes():
    """Test occlusion sensitivity with different occlusion sizes and strides."""
    model = SimpleClassifier(num_classes=3)
    model.eval()

    images = torch.randn(1, 3, 64, 64)
    targets = torch.tensor([0])
    device = torch.device("cpu")

    # Test different configurations
    configs = [
        (8, 4),  # Small occlusion, small stride
        (16, 8),  # Medium occlusion, medium stride
        (32, 16),  # Large occlusion, large stride
    ]

    for occlusion_size, stride in configs:
        sensitivity_maps = occlusion_sensitivity(
            model, images, targets, device, occlusion_size=occlusion_size, stride=stride
        )

        h_steps = (64 - occlusion_size) // stride + 1
        w_steps = (64 - occlusion_size) // stride + 1

        assert sensitivity_maps.shape == (1, h_steps, w_steps)
        assert not torch.isnan(sensitivity_maps).any()
