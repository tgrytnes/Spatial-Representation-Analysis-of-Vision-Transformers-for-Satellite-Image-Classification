"""Tests for model calibration and uncertainty quantification (Epic 3)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eurosat_vit_analysis.calibration import (
    CalibrationMetrics,
    TemperatureScaler,
    TemperatureScalingResult,
    apply_temperature_scaling,
    compute_ece,
    evaluate_calibration,
    generate_reliability_diagram_data,
)

# ============================================================================
# Tests for compute_ece
# ============================================================================


def test_compute_ece_perfect_calibration():
    """Test ECE computation with perfectly calibrated predictions."""
    # Perfect calibration: confidence matches accuracy
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5])
    predictions = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])  # Some correct, some wrong

    ece, metrics = compute_ece(confidences, predictions, targets, num_bins=5)

    # ECE should be very low (may not be exactly 0 due to binning)
    assert isinstance(ece, float)
    assert ece >= 0.0
    assert isinstance(metrics, CalibrationMetrics)
    assert metrics.ece == ece
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.avg_confidence <= 1.0


def test_compute_ece_overconfident():
    """Test ECE with overconfident predictions (high confidence, low accuracy)."""
    # Model is overconfident: high confidence but wrong predictions
    confidences = np.array([0.95] * 10)
    predictions = np.array([0] * 10)
    targets = np.array([1] * 10)  # All predictions are wrong

    ece, metrics = compute_ece(confidences, predictions, targets, num_bins=10)

    # ECE should be high (close to 0.95 since accuracy is 0 but confidence is 0.95)
    assert ece > 0.8  # Should be close to 0.95
    assert metrics.accuracy == 0.0
    assert metrics.avg_confidence > 0.9


def test_compute_ece_underconfident():
    """Test ECE with underconfident predictions (low confidence, high accuracy)."""
    # Model is underconfident: low confidence but correct predictions
    confidences = np.array([0.4] * 10)
    predictions = np.array([0] * 10)
    targets = np.array([0] * 10)  # All predictions are correct

    ece, metrics = compute_ece(confidences, predictions, targets, num_bins=10)

    # ECE should be high (close to 0.6 since accuracy is 1.0 but confidence is 0.4)
    assert ece > 0.5
    assert metrics.accuracy == 1.0
    assert metrics.avg_confidence < 0.5


def test_compute_ece_bin_structure():
    """Test that ECE computation produces correct bin structure."""
    confidences = np.linspace(0.1, 0.9, 100)
    predictions = np.random.randint(0, 2, 100)
    targets = np.random.randint(0, 2, 100)

    num_bins = 10
    ece, metrics = compute_ece(confidences, predictions, targets, num_bins=num_bins)

    # Check bin structure
    assert len(metrics.bin_accuracies) == num_bins
    assert len(metrics.bin_confidences) == num_bins
    assert len(metrics.bin_counts) == num_bins
    assert sum(metrics.bin_counts) == 100  # All samples accounted for


def test_compute_ece_empty_bins():
    """Test ECE computation when some bins are empty."""
    # All confidences in narrow range
    confidences = np.array([0.5, 0.51, 0.52, 0.53, 0.54])
    predictions = np.array([0, 0, 1, 1, 0])
    targets = np.array([0, 0, 1, 1, 1])

    num_bins = 10
    ece, metrics = compute_ece(confidences, predictions, targets, num_bins=num_bins)

    # Should have some empty bins
    assert any(count == 0 for count in metrics.bin_counts)
    # Non-empty bins should have valid values
    for i, count in enumerate(metrics.bin_counts):
        if count > 0:
            assert 0.0 <= metrics.bin_accuracies[i] <= 1.0
            assert 0.0 <= metrics.bin_confidences[i] <= 1.0


def test_compute_ece_mce():
    """Test that MCE (Maximum Calibration Error) is computed correctly."""
    # Create a case with varying calibration across bins
    confidences = np.array([0.1] * 5 + [0.9] * 5)
    predictions = np.array([0] * 10)
    targets = np.array([0] * 5 + [1] * 5)

    ece, metrics = compute_ece(confidences, predictions, targets, num_bins=10)

    # MCE should be the maximum error across all bins
    assert isinstance(metrics.mce, float)
    assert metrics.mce >= ece  # MCE should be at least as large as ECE
    assert 0.0 <= metrics.mce <= 1.0


# ============================================================================
# Tests for evaluate_calibration
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing that outputs deterministic logits."""

    def __init__(self, num_classes=3, logit_scale=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.logit_scale = logit_scale
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x):
        # Average pool and predict
        x = x.mean(dim=[1, 2, 3], keepdim=True)
        x = x.view(x.size(0), -1)
        return self.fc(x) * self.logit_scale


def test_evaluate_calibration_basic():
    """Test basic calibration evaluation on a model."""
    model = SimpleModel(num_classes=3)
    model.eval()

    # Create synthetic dataset
    images = torch.randn(20, 3, 32, 32)
    targets = torch.randint(0, 3, (20,))
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=5)

    device = torch.device("cpu")
    metrics = evaluate_calibration(model, loader, device, num_bins=5)

    # Check that metrics are valid
    assert isinstance(metrics, CalibrationMetrics)
    assert 0.0 <= metrics.ece <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.avg_confidence <= 1.0
    assert len(metrics.bin_accuracies) == 5
    assert len(metrics.bin_confidences) == 5
    assert len(metrics.bin_counts) == 5


def test_evaluate_calibration_perfect_model():
    """Test calibration of a model that predicts correctly with high confidence."""

    class PerfectModel(nn.Module):
        def forward(self, x):
            batch_size = x.size(0)
            # Return high confidence for class 0
            logits = torch.zeros(batch_size, 3)
            logits[:, 0] = 10.0
            return logits

    model = PerfectModel()
    model.eval()

    # All samples are class 0
    images = torch.randn(10, 3, 32, 32)
    targets = torch.zeros(10, dtype=torch.long)
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=5)

    device = torch.device("cpu")
    metrics = evaluate_calibration(model, loader, device)

    # Perfect accuracy
    assert metrics.accuracy == 1.0
    # High confidence
    assert metrics.avg_confidence > 0.95


# ============================================================================
# Tests for TemperatureScaler
# ============================================================================


def test_temperature_scaler_initialization():
    """Test that temperature scaler initializes with temperature=1.0."""
    scaler = TemperatureScaler()
    assert scaler.temperature.item() == 1.0


def test_temperature_scaler_forward():
    """Test that temperature scaling divides logits correctly."""
    scaler = TemperatureScaler()
    scaler.temperature.data = torch.tensor([2.0])

    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    scaled = scaler(logits)

    expected = logits / 2.0
    assert torch.allclose(scaled, expected)


def test_temperature_scaler_fit_basic():
    """Test that temperature scaler can fit on validation data."""

    # Create an overconfident model
    class OverconfidentModel(nn.Module):
        def forward(self, x):
            batch_size = x.size(0)
            # High logits lead to overconfidence
            logits = torch.randn(batch_size, 3) * 10
            return logits

    model = OverconfidentModel()
    model.eval()

    images = torch.randn(20, 3, 32, 32)
    targets = torch.randint(0, 3, (20,))
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=5)

    device = torch.device("cpu")
    scaler = TemperatureScaler().to(device)
    result = scaler.fit(model, loader, device, max_iter=20)

    # Check result structure
    assert isinstance(result, TemperatureScalingResult)
    assert result.temperature > 0.0
    assert 0.0 <= result.ece_before <= 1.0
    assert 0.0 <= result.ece_after <= 1.0
    assert result.nll_before > 0.0
    assert result.nll_after > 0.0


def test_temperature_scaler_reduces_ece():
    """Test that temperature scaling reduces ECE on overconfident model."""

    # Create a very overconfident model
    class VeryOverconfidentModel(nn.Module):
        def forward(self, x):
            batch_size = x.size(0)
            # Very high logits = very high confidence
            logits = torch.zeros(batch_size, 3)
            logits[:, 0] = 100.0  # Extremely confident about class 0
            return logits

    model = VeryOverconfidentModel()
    model.eval()

    # Mix of correct and incorrect predictions
    images = torch.randn(30, 3, 32, 32)
    targets = torch.cat(
        [torch.zeros(15, dtype=torch.long), torch.ones(15, dtype=torch.long)]
    )
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=10)

    device = torch.device("cpu")
    scaler = TemperatureScaler().to(device)
    result = scaler.fit(model, loader, device, max_iter=50)

    # Temperature should increase to reduce confidence
    assert result.temperature > 1.0
    # ECE should decrease (model is overconfident)
    assert result.ece_after < result.ece_before


def test_temperature_scaler_preserves_accuracy():
    """Test that temperature scaling doesn't change predictions (only confidence)."""

    class DeterministicClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 3, bias=False)
            # Set fixed weights
            with torch.no_grad():
                self.fc.weight = nn.Parameter(torch.tensor([[1.0], [2.0], [3.0]]))

        def forward(self, x):
            # Use mean of input for deterministic logits
            x = x.mean(dim=[1, 2, 3], keepdim=True)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    torch.manual_seed(42)
    model = DeterministicClassifier()
    model.eval()

    images = torch.randn(20, 3, 32, 32)
    targets = torch.randint(0, 3, (20,))

    # Get predictions before scaling
    with torch.no_grad():
        logits_before = model(images)
        preds_before = logits_before.argmax(dim=1)

    # Fit temperature scaler
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=5)
    device = torch.device("cpu")

    scaler = TemperatureScaler().to(device)
    scaler.fit(model, loader, device)

    # Get predictions after scaling
    with torch.no_grad():
        logits_after = scaler(model(images))
        preds_after = logits_after.argmax(dim=1)

    # Predictions should be identical (argmax is invariant to scaling)
    assert torch.equal(preds_before, preds_after)


# ============================================================================
# Tests for apply_temperature_scaling
# ============================================================================


def test_apply_temperature_scaling_integration():
    """Test full temperature scaling pipeline."""
    model = SimpleModel(num_classes=3, logit_scale=5.0)  # Overconfident
    model.eval()

    images = torch.randn(40, 3, 32, 32)
    targets = torch.randint(0, 3, (40,))
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=10)

    device = torch.device("cpu")
    result, metrics_before, metrics_after = apply_temperature_scaling(
        model, loader, device, num_bins=10
    )

    # Check all three returns
    assert isinstance(result, TemperatureScalingResult)
    assert isinstance(metrics_before, CalibrationMetrics)
    assert isinstance(metrics_after, CalibrationMetrics)

    # ECE values should match (with small tolerance for floating point differences)
    assert metrics_before.ece == pytest.approx(result.ece_before, abs=1e-6)
    assert metrics_after.ece == pytest.approx(result.ece_after, abs=1e-6)


# ============================================================================
# Tests for generate_reliability_diagram_data
# ============================================================================


def test_generate_reliability_diagram_data_structure():
    """Test that reliability diagram data has correct structure."""
    # Create sample metrics
    metrics = CalibrationMetrics(
        ece=0.05,
        mce=0.10,
        accuracy=0.85,
        avg_confidence=0.90,
        bin_accuracies=[0.1, 0.3, 0.5, 0.7, 0.9],
        bin_confidences=[0.1, 0.3, 0.5, 0.7, 0.9],
        bin_counts=[10, 20, 30, 25, 15],
    )

    data = generate_reliability_diagram_data(metrics)

    # Check structure
    assert "bin_centers" in data
    assert "bin_accuracies" in data
    assert "bin_confidences" in data
    assert "bin_counts" in data
    assert "gap" in data

    # Check lengths match
    assert len(data["bin_centers"]) == 5
    assert len(data["bin_accuracies"]) == 5
    assert len(data["bin_confidences"]) == 5
    assert len(data["bin_counts"]) == 5
    assert len(data["gap"]) == 5


def test_generate_reliability_diagram_data_values():
    """Test that reliability diagram data computes gaps correctly."""
    metrics = CalibrationMetrics(
        ece=0.05,
        mce=0.10,
        accuracy=0.85,
        avg_confidence=0.90,
        bin_accuracies=[0.2, 0.5, 0.8],
        bin_confidences=[0.3, 0.5, 0.7],
        bin_counts=[10, 20, 15],
    )

    data = generate_reliability_diagram_data(metrics)

    # Gap should be confidence - accuracy
    expected_gaps = [0.3 - 0.2, 0.5 - 0.5, 0.7 - 0.8]
    assert data["gap"] == pytest.approx(expected_gaps)

    # Bin centers should be at 0.5/num_bins, 1.5/num_bins, 2.5/num_bins
    num_bins = 3
    expected_centers = [0.5 / num_bins, 1.5 / num_bins, 2.5 / num_bins]
    assert data["bin_centers"] == pytest.approx(expected_centers)


def test_generate_reliability_diagram_data_consistency():
    """Test that diagram data matches original metrics."""
    metrics = CalibrationMetrics(
        ece=0.12,
        mce=0.25,
        accuracy=0.75,
        avg_confidence=0.82,
        bin_accuracies=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        bin_confidences=[0.15, 0.35, 0.55, 0.75, 0.85, 0.95],
        bin_counts=[5, 10, 15, 20, 8, 2],
    )

    data = generate_reliability_diagram_data(metrics)

    # Values should match metrics
    assert data["bin_accuracies"] == metrics.bin_accuracies
    assert data["bin_confidences"] == metrics.bin_confidences
    assert data["bin_counts"] == metrics.bin_counts
