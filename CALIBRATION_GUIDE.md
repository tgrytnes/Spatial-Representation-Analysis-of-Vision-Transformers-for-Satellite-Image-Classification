# Calibration & Uncertainty Quantification Guide - Epic 3

This guide shows how to evaluate and improve model calibration using Expected Calibration Error (ECE) and temperature scaling.

## Overview

The calibration module provides tools for:

1. **ECE Computation**: Measure how well predicted probabilities match actual accuracy
2. **Temperature Scaling**: Post-processing method to calibrate probabilities
3. **Reliability Diagrams**: Visual assessment of calibration quality

## Table of Contents

- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Integration with Training Pipeline](#integration-with-training-pipeline)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)

---

## Quick Start

### Running the Demo

```bash
poetry run python demo_calibration.py
```

This creates visualizations and reports in `outputs/calibration/`:
- `reliability_diagram_demo.png` - Before/after calibration comparison
- `calibration_report_demo.txt` - Detailed metrics
- `model_comparison_demo.png` - ViT vs ResNet comparison

---

## API Reference

### 1. Computing ECE

```python
from eurosat_vit_analysis.calibration import compute_ece
import numpy as np

# Compute ECE from predictions
confidences = np.array([0.9, 0.8, 0.7, ...])  # Max probabilities
predictions = np.array([0, 1, 2, ...])         # Predicted classes
targets = np.array([0, 1, 1, ...])             # True classes

ece, metrics = compute_ece(
    confidences,
    predictions,
    targets,
    num_bins=15  # Number of confidence bins
)

print(f"Expected Calibration Error: {ece:.4f}")
print(f"Maximum Calibration Error: {metrics.mce:.4f}")
print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"Avg Confidence: {metrics.avg_confidence:.4f}")
```

**What it does**: Bins predictions by confidence level and computes the weighted average difference between confidence and accuracy.

---

### 2. Evaluating Model Calibration

```python
from eurosat_vit_analysis.calibration import evaluate_calibration

metrics = evaluate_calibration(
    model=model,        # Trained model
    loader=val_loader,  # Validation DataLoader
    device=device,      # torch.device
    num_bins=15         # Number of bins
)

print(f"ECE: {metrics.ece:.4f}")
print(f"Accuracy: {metrics.accuracy:.4f}")
```

---

### 3. Temperature Scaling

```python
from eurosat_vit_analysis.calibration import TemperatureScaler

# Create and fit temperature scaler
scaler = TemperatureScaler()
result = scaler.fit(
    model=model,
    loader=val_loader,
    device=device,
    max_iter=50,  # Optimization iterations
    lr=0.01       # Learning rate
)

print(f"Optimal Temperature: {result.temperature:.4f}")
print(f"ECE Before: {result.ece_before:.4f}")
print(f"ECE After:  {result.ece_after:.4f}")

# Use the calibrated model
calibrated_logits = scaler(model(images))
probs = F.softmax(calibrated_logits, dim=1)
```

**What it does**: Finds optimal temperature T that divides logits before softmax, minimizing negative log-likelihood on validation set.

---

### 4. Full Calibration Pipeline

```python
from eurosat_vit_analysis.calibration import apply_temperature_scaling

result, metrics_before, metrics_after = apply_temperature_scaling(
    model=model,
    loader=val_loader,
    device=device,
    num_bins=15
)

# Result contains:
# - result.temperature: Optimal T
# - result.ece_before: ECE before scaling
# - result.ece_after: ECE after scaling
# - result.nll_before: Negative log-likelihood before
# - result.nll_after: Negative log-likelihood after

# metrics_before: CalibrationMetrics before scaling
# metrics_after: CalibrationMetrics after scaling
```

---

### 5. Generating Reliability Diagrams

```python
from eurosat_vit_analysis.calibration import generate_reliability_diagram_data

# Get data for plotting
data = generate_reliability_diagram_data(metrics)

# data contains:
# - bin_centers: X-axis values (confidence bin centers)
# - bin_accuracies: Y-axis values (actual accuracy per bin)
# - bin_confidences: Mean confidence per bin
# - bin_counts: Number of samples per bin
# - gap: Calibration gap (confidence - accuracy)

# Use for custom plotting or use provided plot_reliability_diagram()
```

---

## Usage Examples

### Example 1: Evaluate Model Calibration

```python
import torch
from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.calibration import evaluate_calibration

# Load trained model
model = create_model("vit_base", num_classes=10)
model.load_state_dict(torch.load("checkpoints/vit_base_best.pt"))
model.eval()

# Load validation data
_, val_loader, _ = prepare_data(
    data_dir="data/EuroSAT",
    batch_size=32,
    augmentation="none"
)

# Evaluate calibration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

metrics = evaluate_calibration(model, val_loader, device)

print(f"Model Calibration Metrics:")
print(f"  ECE: {metrics.ece:.6f}")
print(f"  MCE: {metrics.mce:.6f}")
print(f"  Accuracy: {metrics.accuracy:.4f}")
print(f"  Avg Confidence: {metrics.avg_confidence:.4f}")
```

---

### Example 2: Apply Temperature Scaling

```python
from eurosat_vit_analysis.calibration import apply_temperature_scaling

# Apply temperature scaling
result, metrics_before, metrics_after = apply_temperature_scaling(
    model, val_loader, device, num_bins=15
)

print("\\nCalibration Results:")
print(f"  Optimal Temperature: {result.temperature:.4f}")
print(f"\\n  Before Scaling:")
print(f"    ECE: {result.ece_before:.6f}")
print(f"    NLL: {result.nll_before:.6f}")
print(f"\\n  After Scaling:")
print(f"    ECE: {result.ece_after:.6f}")
print(f"    NLL: {result.nll_after:.6f}")
print(f"\\n  Improvement:")
ece_improvement = result.ece_before - result.ece_after
print(f"    ECE Reduction: {ece_improvement:.6f}")
print(f"    Relative: {(ece_improvement / result.ece_before) * 100:.2f}%")
```

---

### Example 3: Compare Models

```python
models = {
    "ViT-Base": ("vit_base", "checkpoints/vit_base_best.pt"),
    "ResNet50": ("resnet50", "checkpoints/resnet50_best.pt"),
}

for model_name, (arch, checkpoint) in models.items():
    print(f"\\n{'='*60}")
    print(f"Evaluating {model_name}")
    print('='*60)

    model = create_model(arch, num_classes=10)
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(device).eval()

    result, _, _ = apply_temperature_scaling(model, val_loader, device)

    print(f"  ECE Before: {result.ece_before:.4f}")
    print(f"  ECE After:  {result.ece_after:.4f}")
    print(f"  Temperature: {result.temperature:.4f}")
    print(f"  Improvement: {(result.ece_before - result.ece_after):.4f}")
```

---

### Example 4: Visualize Calibration

```python
import matplotlib.pyplot as plt
from eurosat_vit_analysis.calibration import generate_reliability_diagram_data

result, metrics_before, metrics_after = apply_temperature_scaling(
    model, val_loader, device
)

# Get diagram data
data_after = generate_reliability_diagram_data(metrics_after)

# Create reliability diagram
fig, ax = plt.subplots(figsize=(8, 6))

# Bar chart: actual accuracy per bin
ax.bar(
    data_after["bin_centers"],
    data_after["bin_accuracies"],
    width=1.0/len(data_after["bin_centers"]),
    alpha=0.7,
    label="Actual Accuracy",
    color="steelblue",
    edgecolor="black"
)

# Line plot: mean confidence per bin
ax.plot(
    data_after["bin_centers"],
    data_after["bin_confidences"],
    "r-o",
    linewidth=2,
    label="Mean Confidence"
)

# Perfect calibration line
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect Calibration")

ax.set_xlabel("Confidence", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title(f"Reliability Diagram\\nECE: {metrics_after.ece:.4f}", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.savefig("reliability_diagram.png", dpi=150, bbox_inches="tight")
```

---

## Integration with Training Pipeline

### After Training Evaluation

Add calibration evaluation to your standard evaluation script:

```python
# evaluate_model.py

from eurosat_vit_analysis.calibration import apply_temperature_scaling
from pathlib import Path

def evaluate_trained_model(
    model_name: str,
    checkpoint_path: str,
    output_dir: str = "outputs/evaluation"
):
    """Evaluate model including calibration metrics."""

    # Load model and data (standard evaluation setup)
    model = create_model(model_name, num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device).eval()

    _, val_loader, _ = prepare_data("data/EuroSAT", batch_size=32)

    # Standard metrics: accuracy, loss, etc.
    # ... (your existing evaluation code)

    # Calibration metrics
    print("\\nEvaluating calibration...")
    result, metrics_before, metrics_after = apply_temperature_scaling(
        model, val_loader, device
    )

    # Save calibration report
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "calibration_report.txt", "w") as f:
        f.write(f"Model: {model_name}\\n")
        f.write(f"Checkpoint: {checkpoint_path}\\n\\n")

        f.write("CALIBRATION METRICS:\\n")
        f.write(f"  ECE Before Scaling: {result.ece_before:.6f}\\n")
        f.write(f"  ECE After Scaling:  {result.ece_after:.6f}\\n")
        f.write(f"  Optimal Temperature: {result.temperature:.6f}\\n")
        f.write(f"  ECE Improvement: {result.ece_before - result.ece_after:.6f}\\n")

    print(f"✓ Calibration report saved to {output_path}")

    return result
```

---

## Interpreting Results

### Expected Calibration Error (ECE)

**What it means**:
- ECE quantifies how far predicted probabilities are from actual accuracy
- Range: [0, 1], lower is better
- **ECE < 0.05**: Excellent calibration
- **ECE 0.05-0.15**: Good calibration
- **ECE 0.15-0.25**: Fair calibration
- **ECE > 0.25**: Poor calibration

**Example Interpretation**:
```
ECE Before: 0.18  →  Fair calibration (needs improvement)
ECE After:  0.04  →  Excellent calibration (well-calibrated)
Improvement: 0.14 →  Substantial improvement from temperature scaling
```

### Temperature Values

**What they mean**:
- **T = 1.0**: No scaling (original model)
- **T > 1.0**: Model is overconfident (needs softening)
- **T < 1.0**: Model is underconfident (needs sharpening)

**Typical Values**:
- Modern neural networks: T = 1.5 to 3.0
- Very overconfident models: T > 5.0
- Well-calibrated models: T ≈ 1.0

### Reliability Diagrams

**Reading the diagram**:

1. **Perfect Calibration** (diagonal line):
   - If points lie on this line, model is perfectly calibrated
   - Confidence matches actual accuracy at all levels

2. **Overconfidence** (points below diagonal):
   - Model predicts higher confidence than actual accuracy
   - Example: 90% confidence but only 70% accuracy
   - Common in modern neural networks

3. **Underconfidence** (points above diagonal):
   - Model predicts lower confidence than actual accuracy
   - Example: 60% confidence but 80% accuracy
   - Less common, sometimes occurs after aggressive regularization

**Example Analysis**:
```
If high-confidence bins (0.8-1.0) have bars much lower than the red line:
→ Model is overconfident in its top predictions
→ Temperature scaling will help by reducing extreme confidences
```

### Architecture Differences

**Vision Transformers (ViT)**:
- Often more overconfident than CNNs
- Typically need higher temperature (T > 2.0)
- Larger ECE before calibration
- Greater benefit from temperature scaling

**ResNets**:
- Better calibrated out-of-the-box
- Lower temperature needed (T ≈ 1.0-1.5)
- Smaller ECE before calibration
- Still benefits from temperature scaling

---

## Best Practices

### 1. When to Apply Temperature Scaling

✅ **Do apply when**:
- Deploying model in production
- Probability estimates are important for decision-making
- Model shows high ECE (> 0.10)
- Combining model predictions (ensembles)

❌ **Don't bother when**:
- Only care about argmax predictions (top-1 accuracy)
- Model already well-calibrated (ECE < 0.05)
- Using model for ranking (relative order matters, not absolute values)

### 2. Data Split for Temperature Scaling

- **Use validation set** (not test set!) to fit temperature
- Same set used for early stopping is fine
- Need at least ~1000 samples for reliable temperature estimation
- More samples → better temperature estimate

### 3. Number of Bins

- **Default: 15 bins** works well for most cases
- Too few bins (< 10): ECE may not capture calibration issues
- Too many bins (> 20): Some bins may be empty, noisy estimates
- More samples → can use more bins

### 4. Saving Calibrated Models

```python
# Option 1: Save temperature separately
torch.save({
    'model_state_dict': model.state_dict(),
    'temperature': scaler.temperature.item()
}, "model_calibrated.pt")

# Option 2: Save full calibrated model
class CalibratedModel(nn.Module):
    def __init__(self, base_model, temperature):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(self, x):
        logits = self.base_model(x)
        return logits / self.temperature

calibrated = CalibratedModel(model, scaler.temperature.item())
torch.save(calibrated.state_dict(), "model_calibrated.pt")
```

### 5. Monitoring Calibration During Training

```python
# Add to training loop
if epoch % 10 == 0:  # Every 10 epochs
    metrics = evaluate_calibration(model, val_loader, device)
    print(f"Epoch {epoch} - ECE: {metrics.ece:.4f}")
    # Log to tensorboard/wandb
```

---

## Troubleshooting

**Q: ECE doesn't improve after temperature scaling?**

A: This can happen if:
- Validation set is too small (< 500 samples)
- Model has very poor accuracy (< 30%)
- Try increasing `max_iter` parameter in `scaler.fit()`

**Q: Temperature is very high (> 10)?**

A: This indicates severe overconfidence. Check:
- Model architecture (some architectures are more confident)
- Training hyperparameters (high learning rate can cause overconfidence)
- Data distribution (train/val mismatch)

**Q: Can I use temperature scaling for regression?**

A: Temperature scaling is specifically for classification. For regression uncertainty, consider:
- Gaussian process regression
- Ensemble methods
- Monte Carlo dropout

**Q: Does temperature scaling hurt accuracy?**

A: No! Temperature scaling is **prediction-preserving**. The argmax is invariant to scaling, so top-1 accuracy remains identical.

---

## References

**Papers**:
- "On Calibration of Modern Neural Networks" (Guo et al., 2017)
- "Measuring Calibration in Deep Learning" (Nixon et al., 2019)
- "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021) - ViT calibration

**Epic 3 Acceptance Criteria**:
- ✅ Expected Calibration Error (ECE) is computed before/after temperature scaling
- ✅ Reliability diagrams are generated for ViT and ResNet

---

For more information, see:
- [eurosat_vit_analysis/calibration.py](eurosat_vit_analysis/calibration.py) - Implementation
- [tests/test_calibration.py](tests/test_calibration.py) - Test suite
- [demo_calibration.py](demo_calibration.py) - Demo script
- [evaluate_calibration.py](evaluate_calibration.py) - Production evaluation
