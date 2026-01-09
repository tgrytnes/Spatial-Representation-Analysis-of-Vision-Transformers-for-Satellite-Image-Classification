"""Model calibration and uncertainty quantification for Epic 3.

This module provides tools for:
1. Computing Expected Calibration Error (ECE)
2. Temperature scaling for probability calibration
3. Generating reliability diagrams
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class CalibrationMetrics:
    """Metrics for model calibration evaluation."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    accuracy: float  # Overall accuracy
    avg_confidence: float  # Average confidence
    bin_accuracies: list[float]  # Accuracy per confidence bin
    bin_confidences: list[float]  # Average confidence per bin
    bin_counts: list[int]  # Number of samples per bin


@dataclass(frozen=True)
class TemperatureScalingResult:
    """Result of temperature scaling optimization."""

    temperature: float  # Optimal temperature value
    ece_before: float  # ECE before calibration
    ece_after: float  # ECE after calibration
    nll_before: float  # Negative log-likelihood before
    nll_after: float  # Negative log-likelihood after


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
) -> tuple[float, CalibrationMetrics]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between confidence and accuracy across bins.
    Lower ECE indicates better calibration.

    Args:
        confidences: Predicted confidence values (max probabilities), shape (N,)
        predictions: Predicted class labels, shape (N,)
        targets: True class labels, shape (N,)
        num_bins: Number of bins for calibration (default: 15)

    Returns:
        Tuple of (ECE value, CalibrationMetrics with detailed statistics)
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Initialize accumulators
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    mce = 0.0
    total = len(confidences)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum()

        if bin_count > 0:
            # Compute accuracy and confidence for this bin
            bin_accuracy = (predictions[in_bin] == targets[in_bin]).mean()
            bin_confidence = confidences[in_bin].mean()

            # Update ECE (weighted by bin size)
            bin_weight = bin_count / total
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

            # Update MCE (maximum error across bins)
            mce = max(mce, abs(bin_accuracy - bin_confidence))

            bin_accuracies.append(float(bin_accuracy))
            bin_confidences.append(float(bin_confidence))
            bin_counts.append(int(bin_count))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    # Overall metrics
    accuracy = (predictions == targets).mean()
    avg_confidence = confidences.mean()

    metrics = CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        accuracy=float(accuracy),
        avg_confidence=float(avg_confidence),
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )

    return float(ece), metrics


def evaluate_calibration(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_bins: int = 15,
) -> CalibrationMetrics:
    """
    Evaluate model calibration on a dataset.

    Args:
        model: Trained model to evaluate
        loader: DataLoader with evaluation data
        device: Device for computation
        num_bins: Number of bins for ECE computation

    Returns:
        CalibrationMetrics with ECE and detailed statistics
    """
    model.eval()

    all_confidences = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            # Get model predictions
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            # Get max confidence and predicted class
            confidences, predictions = probs.max(dim=1)

            all_confidences.append(confidences.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    confidences = np.concatenate(all_confidences)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Compute ECE
    _, metrics = compute_ece(confidences, predictions, targets, num_bins)

    return metrics


class TemperatureScaler(nn.Module):
    """
    Temperature scaling layer for probability calibration.

    Temperature scaling is a simple post-processing method that divides
    logits by a scalar temperature before softmax. The temperature is
    optimized to minimize negative log-likelihood on a validation set.

    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """

    def __init__(self):
        super().__init__()
        # Initialize temperature to 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def fit(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> TemperatureScalingResult:
        """
        Optimize temperature parameter on validation set.

        Args:
            model: Trained model (frozen)
            loader: Validation data loader
            device: Device for computation
            max_iter: Maximum optimization iterations
            lr: Learning rate for optimization

        Returns:
            TemperatureScalingResult with optimal temperature and metrics
        """
        # Store original model state
        model.eval()

        # Collect logits and targets from validation set
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                all_logits.append(logits)
                all_targets.append(targets)

        logits = torch.cat(all_logits).to(device)
        targets = torch.cat(all_targets).to(device)

        # Compute ECE before scaling
        with torch.no_grad():
            probs_before = F.softmax(logits, dim=1)
            confidences_before, predictions_before = probs_before.max(dim=1)
            ece_before, _ = compute_ece(
                confidences_before.cpu().numpy(),
                predictions_before.cpu().numpy(),
                targets.cpu().numpy(),
            )

            # Compute NLL before scaling
            nll_before = F.cross_entropy(logits, targets).item()

        # Optimize temperature
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Compute ECE after scaling
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            probs_after = F.softmax(scaled_logits, dim=1)
            confidences_after, predictions_after = probs_after.max(dim=1)
            ece_after, _ = compute_ece(
                confidences_after.cpu().numpy(),
                predictions_after.cpu().numpy(),
                targets.cpu().numpy(),
            )

            # Compute NLL after scaling
            nll_after = F.cross_entropy(scaled_logits, targets).item()

        return TemperatureScalingResult(
            temperature=float(self.temperature.item()),
            ece_before=float(ece_before),
            ece_after=float(ece_after),
            nll_before=float(nll_before),
            nll_after=float(nll_after),
        )


def apply_temperature_scaling(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_bins: int = 15,
) -> tuple[TemperatureScalingResult, CalibrationMetrics, CalibrationMetrics]:
    """
    Apply temperature scaling to a model and evaluate calibration.

    Args:
        model: Trained model to calibrate
        loader: Validation data loader for fitting temperature
        device: Device for computation
        num_bins: Number of bins for ECE computation

    Returns:
        Tuple of:
        - TemperatureScalingResult with optimal temperature
        - CalibrationMetrics before scaling
        - CalibrationMetrics after scaling
    """
    # Evaluate calibration before scaling
    metrics_before = evaluate_calibration(model, loader, device, num_bins)

    # Fit temperature scaler
    scaler = TemperatureScaler().to(device)
    scaling_result = scaler.fit(model, loader, device)

    # Create a wrapper model with temperature scaling
    class CalibratedModel(nn.Module):
        def __init__(self, base_model, temperature_scaler):
            super().__init__()
            self.base_model = base_model
            self.temperature_scaler = temperature_scaler

        def forward(self, x):
            logits = self.base_model(x)
            return self.temperature_scaler(logits)

    calibrated_model = CalibratedModel(model, scaler)

    # Evaluate calibration after scaling
    metrics_after = evaluate_calibration(calibrated_model, loader, device, num_bins)

    return scaling_result, metrics_before, metrics_after


def generate_reliability_diagram_data(
    metrics: CalibrationMetrics,
) -> dict[str, list[float]]:
    """
    Prepare data for reliability diagram visualization.

    Args:
        metrics: CalibrationMetrics from evaluation

    Returns:
        Dictionary with data for plotting:
        - bin_centers: Center of each confidence bin
        - bin_accuracies: Accuracy for each bin
        - bin_confidences: Average confidence for each bin
        - bin_counts: Number of samples per bin
        - gap: Calibration gap (confidence - accuracy) per bin
    """
    num_bins = len(metrics.bin_accuracies)
    bin_width = 1.0 / num_bins
    bin_centers = [(i + 0.5) * bin_width for i in range(num_bins)]

    gaps = [
        conf - acc for conf, acc in zip(metrics.bin_confidences, metrics.bin_accuracies)
    ]

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": metrics.bin_accuracies,
        "bin_confidences": metrics.bin_confidences,
        "bin_counts": metrics.bin_counts,
        "gap": gaps,
    }
