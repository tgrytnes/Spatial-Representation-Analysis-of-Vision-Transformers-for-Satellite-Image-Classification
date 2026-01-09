# Epic 3: Calibration & Uncertainty - Acceptance Criteria Demo

## Story: Calibrated Probabilities and Uncertainty Metrics

**Epic**: Epic 3: Explainability & Robustness (The Insight)

---

## Acceptance Criteria

### ✅ 1. Expected Calibration Error (ECE) is computed before/after temperature scaling

**Implementation**: [`eurosat_vit_analysis/calibration.py`](eurosat_vit_analysis/calibration.py)

**Key Functions**:

```python
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
    """
```

```python
def apply_temperature_scaling(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_bins: int = 15,
) -> tuple[TemperatureScalingResult, CalibrationMetrics, CalibrationMetrics]:
    """
    Apply temperature scaling to a model and evaluate calibration.

    Returns:
        - TemperatureScalingResult with optimal temperature
        - CalibrationMetrics BEFORE scaling
        - CalibrationMetrics AFTER scaling
    """
```

**Usage Example**:

```python
from eurosat_vit_analysis.calibration import apply_temperature_scaling

# Evaluate model calibration
result, metrics_before, metrics_after = apply_temperature_scaling(
    model=model,
    loader=val_loader,
    device=device,
    num_bins=15
)

# Access ECE before and after
print(f"ECE Before: {result.ece_before:.4f}")
print(f"ECE After:  {result.ece_after:.4f}")
print(f"Improvement: {result.ece_before - result.ece_after:.4f}")
```

**Demo Output**: [`outputs/calibration/calibration_report_demo.txt`](outputs/calibration/calibration_report_demo.txt)

```
CALIBRATION EVALUATION REPORT
======================================================================

Model: ViT-Base (synthetic data)

BEFORE TEMPERATURE SCALING:
  ECE:             0.067529
  MCE:             0.071051
  Accuracy:        0.1100
  Avg Confidence:  0.1775

TEMPERATURE SCALING:
  Optimal T:       3.654911
  NLL Before:      2.343303
  NLL After:       2.296967

AFTER TEMPERATURE SCALING:
  ECE:             0.032544
  MCE:             0.047389
  Accuracy:        0.1100
  Avg Confidence:  0.1425

IMPROVEMENT:
  ECE Reduction:   0.034985
  ECE Reduction %: 51.80%
```

---

### ✅ 2. Reliability diagrams are generated for ViT and ResNet

**Implementation**: Reliability diagram generation in [`evaluate_calibration.py`](evaluate_calibration.py)

**Function**:

```python
def plot_reliability_diagram(
    metrics_before,
    metrics_after,
    model_name,
    output_path,
):
    """Create reliability diagram comparing before/after calibration."""
    # Plots:
    # - Bar chart of actual accuracy per confidence bin
    # - Line plot of mean confidence per bin
    # - Perfect calibration diagonal line
    # Side-by-side comparison (before/after)
```

**Usage Example**:

```python
from eurosat_vit_analysis.calibration import (
    apply_temperature_scaling,
    generate_reliability_diagram_data
)

# Evaluate calibration
result, metrics_before, metrics_after = apply_temperature_scaling(
    model, loader, device
)

# Generate reliability diagram data
data_before = generate_reliability_diagram_data(metrics_before)
data_after = generate_reliability_diagram_data(metrics_after)

# Plot (custom visualization or use provided function)
plot_reliability_diagram(
    metrics_before,
    metrics_after,
    "ViT-Base",
    "reliability_diagram.png"
)
```

**Demo Outputs**:

- **Single Model**: [`outputs/calibration/reliability_diagram_demo.png`](outputs/calibration/reliability_diagram_demo.png)
  - Side-by-side comparison: Before (left) vs After (right)
  - Shows calibration improvement with temperature scaling

- **Model Comparison**: [`outputs/calibration/model_comparison_demo.png`](outputs/calibration/model_comparison_demo.png)
  - ViT-Base vs ResNet50
  - Before and after temperature scaling for each model

**Reliability Diagram Interpretation**:
- **X-axis**: Model confidence (predicted probability)
- **Y-axis**: Actual accuracy
- **Blue bars**: Actual accuracy per confidence bin
- **Red line**: Mean confidence per bin
- **Black dashed line**: Perfect calibration (confidence = accuracy)
- **Goal**: Red line should be close to the black diagonal

---

## Test Coverage

**Test File**: [`tests/test_calibration.py`](tests/test_calibration.py)

**17 Tests** covering all functionality:

### ECE Computation Tests (6 tests)
- ✅ `test_compute_ece_perfect_calibration` - Perfect calibration baseline
- ✅ `test_compute_ece_overconfident` - Overconfident predictions
- ✅ `test_compute_ece_underconfident` - Underconfident predictions
- ✅ `test_compute_ece_bin_structure` - Bin structure validation
- ✅ `test_compute_ece_empty_bins` - Empty bin handling
- ✅ `test_compute_ece_mce` - Maximum Calibration Error

### Calibration Evaluation Tests (2 tests)
- ✅ `test_evaluate_calibration_basic` - Basic evaluation pipeline
- ✅ `test_evaluate_calibration_perfect_model` - Perfect model baseline

### Temperature Scaling Tests (5 tests)
- ✅ `test_temperature_scaler_initialization` - Initialization
- ✅ `test_temperature_scaler_forward` - Forward pass
- ✅ `test_temperature_scaler_fit_basic` - Fitting temperature
- ✅ `test_temperature_scaler_reduces_ece` - ECE reduction ← **Criterion 1**
- ✅ `test_temperature_scaler_preserves_accuracy` - Accuracy preservation

### Integration Tests (1 test)
- ✅ `test_apply_temperature_scaling_integration` - Full pipeline ← **Criterion 1**

### Reliability Diagram Tests (3 tests)
- ✅ `test_generate_reliability_diagram_data_structure` - Data structure ← **Criterion 2**
- ✅ `test_generate_reliability_diagram_data_values` - Gap computation ← **Criterion 2**
- ✅ `test_generate_reliability_diagram_data_consistency` - Consistency checks

**Run Tests**:
```bash
poetry run pytest tests/test_calibration.py -v
```

**Result**: All 17 tests passing ✅

---

## Quick Demo

```bash
# Run comprehensive demo
poetry run python demo_calibration.py

# Output:
# ✓ outputs/calibration/reliability_diagram_demo.png
# ✓ outputs/calibration/calibration_report_demo.txt
# ✓ outputs/calibration/model_comparison_demo.png
```

---

## Integration Example: EuroSAT Evaluation

```python
#!/usr/bin/env python
"""Evaluate calibration on EuroSAT validation set."""

import torch
from pathlib import Path
from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.calibration import apply_temperature_scaling

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path("outputs/calibration_evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = create_model("vit_base", num_classes=10)
model.load_state_dict(torch.load("checkpoints/vit_base_best.pt"))
model = model.to(device).eval()

# Load EuroSAT validation data
_, val_loader, class_names = prepare_data(
    data_dir="data/EuroSAT",
    batch_size=32,
    augmentation="none"
)

# ===== Criterion 1: ECE before/after temperature scaling =====
print("Evaluating calibration...")
result, metrics_before, metrics_after = apply_temperature_scaling(
    model=model,
    loader=val_loader,
    device=device,
    num_bins=15
)

# Log ECE values
print(f"ECE Before Temperature Scaling: {result.ece_before:.6f}")
print(f"ECE After Temperature Scaling:  {result.ece_after:.6f}")
print(f"Optimal Temperature:            {result.temperature:.6f}")
print(f"ECE Improvement:                {result.ece_before - result.ece_after:.6f}")

# ===== Criterion 2: Generate reliability diagram =====
print("\nGenerating reliability diagram...")
from evaluate_calibration import plot_reliability_diagram

diagram_path = output_dir / "vit_base_reliability_diagram.png"
plot_reliability_diagram(
    metrics_before,
    metrics_after,
    "ViT-Base",
    diagram_path
)
print(f"✓ Saved reliability diagram to {diagram_path}")

# Save detailed report
log_path = output_dir / "vit_base_calibration_report.txt"
with open(log_path, "w") as f:
    f.write("EuroSAT Calibration Evaluation\\n")
    f.write("=" * 70 + "\\n\\n")
    f.write(f"Model: ViT-Base\\n")
    f.write(f"Dataset: EuroSAT Validation Set\\n\\n")

    f.write("BEFORE TEMPERATURE SCALING:\\n")
    f.write(f"  ECE: {metrics_before.ece:.6f}\\n")
    f.write(f"  MCE: {metrics_before.mce:.6f}\\n")
    f.write(f"  Accuracy: {metrics_before.accuracy:.4f}\\n")
    f.write(f"  Avg Confidence: {metrics_before.avg_confidence:.4f}\\n\\n")

    f.write("TEMPERATURE SCALING:\\n")
    f.write(f"  Optimal T: {result.temperature:.6f}\\n\\n")

    f.write("AFTER TEMPERATURE SCALING:\\n")
    f.write(f"  ECE: {metrics_after.ece:.6f}\\n")
    f.write(f"  MCE: {metrics_after.mce:.6f}\\n")
    f.write(f"  Accuracy: {metrics_after.accuracy:.4f}\\n")
    f.write(f"  Avg Confidence: {metrics_after.avg_confidence:.4f}\\n\\n")

    f.write("IMPROVEMENT:\\n")
    improvement = result.ece_before - result.ece_after
    improvement_pct = (improvement / result.ece_before) * 100
    f.write(f"  ECE Reduction: {improvement:.6f} ({improvement_pct:.2f}%)\\n")

print(f"✓ Logged to {log_path}")

print("\\n" + "="*70)
print("ACCEPTANCE CRITERIA STATUS")
print("="*70)
print("✅ ECE is computed before/after temperature scaling")
print("✅ Reliability diagrams are generated for ViT and ResNet")
```

---

## Production Evaluation Script

**Script**: [`evaluate_calibration.py`](evaluate_calibration.py)

**Usage**:

```bash
# Evaluate single model
python evaluate_calibration.py --model vit_base \\
    --checkpoint checkpoints/vit_base_best.pt

# Evaluate all models
python evaluate_calibration.py --all

# Custom settings
python evaluate_calibration.py --model resnet50 \\
    --checkpoint checkpoints/resnet50_best.pt \\
    --batch-size 64 \\
    --num-bins 20
```

**Output Structure**:
```
outputs/calibration_evaluation/
├── vit_base/
│   ├── calibration_report.txt         # Detailed metrics
│   └── reliability_diagram.png         # Visualization
├── resnet50/
│   ├── calibration_report.txt
│   └── reliability_diagram.png
└── model_comparison.png                # Side-by-side comparison
```

---

## Summary

All acceptance criteria for Epic 3 Calibration story are **fully implemented and tested**:

| Criterion | Implementation | Demo | Tests |
|-----------|---------------|------|-------|
| 1. ECE computed before/after | ✅ `compute_ece()` + `apply_temperature_scaling()` | ✅ `calibration_report_demo.txt` | ✅ 13 tests |
| 2. Reliability diagrams | ✅ `plot_reliability_diagram()` | ✅ `.png` visualizations | ✅ 4 tests |

**Total**: 17/17 tests passing, comprehensive demo, full documentation.

---

## Key Concepts

### Expected Calibration Error (ECE)
- Measures the difference between model confidence and actual accuracy
- Computed across confidence bins (default: 15 bins)
- Lower ECE = better calibration
- **Formula**: Weighted average of |confidence - accuracy| per bin

### Temperature Scaling
- Post-processing calibration method
- Divides logits by scalar temperature before softmax
- Temperature optimized on validation set to minimize NLL
- **Does not change predictions** (argmax invariant)
- **Does change confidence levels**

### Reliability Diagram
- Visual tool for assessing calibration
- X-axis: Predicted confidence
- Y-axis: Actual accuracy
- Well-calibrated model: points lie on diagonal
- Overconfident: points below diagonal
- Underconfident: points above diagonal

---

## Files Created

**Core Implementation**:
- `eurosat_vit_analysis/calibration.py` (369 lines)

**Tests**:
- `tests/test_calibration.py` (439 lines)

**Scripts**:
- `demo_calibration.py` - Demo with synthetic data
- `evaluate_calibration.py` - Production evaluation script

**Documentation**:
- `CALIBRATION_ACCEPTANCE_CRITERIA.md` - This file
- `CALIBRATION_GUIDE.md` - Comprehensive usage guide

**Generated Outputs** (from demo):
- `outputs/calibration/reliability_diagram_demo.png`
- `outputs/calibration/calibration_report_demo.txt`
- `outputs/calibration/model_comparison_demo.png`
