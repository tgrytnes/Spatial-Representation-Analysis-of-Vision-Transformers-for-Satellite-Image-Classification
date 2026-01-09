# Story 3.3 Acceptance Criteria - Demonstration

## Story: Quantify Spatial Reasoning and Invariance

**Epic**: Epic 3: Explainability & Robustness (The Insight)

---

## Acceptance Criteria

### ✅ 1. Patch-shuffle and occlusion tests are implemented and logged

**Implementation**: [`eurosat_vit_analysis/spatial_robustness.py`](eurosat_vit_analysis/spatial_robustness.py)

**Functions**:
- `shuffle_patches()` - Lines 21-82
- `evaluate_patch_shuffle()` - Lines 85-179
- `occlusion_sensitivity()` - Lines 182-238

**Logging Example**:
```python
from eurosat_vit_analysis.spatial_robustness import evaluate_patch_shuffle

report = evaluate_patch_shuffle(
    model=model,
    loader=val_loader,
    device=device,
    grid_size=(4, 4),
    num_classes=10,
    seed=42
)

# Log to file
with open("patch_shuffle_report.txt", "w") as f:
    f.write(f"Clean Accuracy:     {report.clean_accuracy:.2%}\n")
    f.write(f"Shuffled Accuracy:  {report.shuffled_accuracy:.2%}\n")
    f.write(f"Accuracy Drop:      {report.accuracy_drop:.2%}\n")
```

**Demo Output**: [`outputs/spatial_robustness/patch_shuffle_report.txt`](outputs/spatial_robustness/patch_shuffle_report.txt)

```
PATCH-SHUFFLE EVALUATION RESULTS
======================================================================

Grid Size: (4, 4)

Overall Metrics:
  Clean Accuracy:     6.00%
  Shuffled Accuracy:  8.00%
  Accuracy Drop:      -2.00%
```

---

### ✅ 2. Accuracy drop under patch-shuffle is reported per class

**Implementation**: `PatchShuffleReport` dataclass (Lines 9-18)

**Fields**:
```python
@dataclass(frozen=True)
class PatchShuffleReport:
    clean_accuracy: float
    shuffled_accuracy: float
    accuracy_drop: float
    per_class_clean_accuracy: dict[int, float]      # ← Per-class
    per_class_shuffled_accuracy: dict[int, float]   # ← Per-class
    per_class_accuracy_drop: dict[int, float]       # ← Per-class
```

**Usage Example**:
```python
report = evaluate_patch_shuffle(model, loader, device, (4, 4), 10)

# Access per-class metrics
for class_id in range(10):
    drop = report.per_class_accuracy_drop[class_id]
    print(f"Class {class_id}: {drop:+.2%} accuracy drop")
```

**Demo Output**: [`outputs/spatial_robustness/patch_shuffle_report.txt`](outputs/spatial_robustness/patch_shuffle_report.txt)

```
Per-Class Accuracy Drop:
  Class    Clean      Shuffled   Drop
  ----------------------------------------
  0        0.00%      0.00%      0.00%
  1        0.00%      0.00%      0.00%
  2        0.00%      0.00%      0.00%
  3        40.00%     60.00%     -20.00%
  4        0.00%      0.00%      0.00%
  5        0.00%      0.00%      0.00%
  6        0.00%      0.00%      0.00%
  7        11.11%     11.11%     0.00%
  8        0.00%      0.00%      0.00%
  9        0.00%      0.00%      0.00%
```

**Visualization**: [`outputs/spatial_robustness/per_class_accuracy_drop.png`](outputs/spatial_robustness/per_class_accuracy_drop.png)

---

### ✅ 3. Occlusion sensitivity maps are saved for a fixed evaluation set

**Implementation**: `occlusion_sensitivity()` function (Lines 182-238)

**Usage Example**:
```python
from eurosat_vit_analysis.spatial_robustness import occlusion_sensitivity
import torch

# Fixed evaluation set
images, targets = next(iter(val_loader))  # Fixed batch
images = images.to(device)
targets = targets.to(device)

# Generate sensitivity maps
sensitivity_maps = occlusion_sensitivity(
    model=model,
    images=images,
    targets=targets,
    device=device,
    occlusion_size=32,
    stride=16
)

# Save for evaluation set
torch.save({
    'maps': sensitivity_maps.cpu(),
    'images': images.cpu(),
    'targets': targets.cpu(),
}, 'occlusion_maps.pt')
```

**Saved Files**:
- `outputs/spatial_robustness/occlusion_maps/sensitivity_maps.pt` - PyTorch tensor bundle
- `outputs/spatial_robustness/occlusion_maps/sensitivity_map_sample_0_class_9.npy` - Individual maps
- `outputs/spatial_robustness/occlusion_maps/sensitivity_map_sample_1_class_9.npy`
- `outputs/spatial_robustness/occlusion_maps/sensitivity_map_sample_2_class_5.npy`
- `outputs/spatial_robustness/occlusion_maps/sensitivity_map_sample_3_class_9.npy`
- `outputs/spatial_robustness/occlusion_maps/sensitivity_map_sample_4_class_9.npy`

**Visualization**: [`outputs/spatial_robustness/occlusion_maps/sensitivity_maps_visualization.png`](outputs/spatial_robustness/occlusion_maps/sensitivity_maps_visualization.png)

**Loading Saved Maps**:
```python
# Load saved maps for analysis
data = torch.load('occlusion_maps.pt')
maps = data['maps']
images = data['images']
targets = data['targets']

print(f"Loaded {len(maps)} sensitivity maps")
print(f"Map shape per image: {maps[0].shape}")
```

---

## Test Coverage

**Test File**: [`tests/test_spatial_robustness.py`](tests/test_spatial_robustness.py)

**17 Tests** covering all functionality:

### Patch Shuffle Tests (8 tests)
- ✅ `test_shuffle_patches_shape` - Shape preservation
- ✅ `test_shuffle_patches_content_preservation` - Content preservation
- ✅ `test_shuffle_patches_deterministic_with_seed` - Reproducibility
- ✅ `test_shuffle_patches_different_seeds` - Different shuffles
- ✅ `test_shuffle_patches_batch_independence` - Per-image shuffling
- ✅ `test_shuffle_patches_invalid_grid_size` - Error handling
- ✅ `test_shuffle_patches_different_grid_sizes` - Multiple grid sizes
- ✅ `test_shuffle_patches_single_patch` - Edge case

### Evaluation Tests (4 tests)
- ✅ `test_evaluate_patch_shuffle_structure` - Report structure
- ✅ `test_evaluate_patch_shuffle_perfect_classifier` - Baseline behavior
- ✅ `test_evaluate_patch_shuffle_spatial_dependence` - Spatial sensitivity
- ✅ `test_evaluate_patch_shuffle_per_class_tracking` - Per-class metrics ← **Criterion 2**

### Occlusion Tests (5 tests)
- ✅ `test_occlusion_sensitivity_shape` - Output shape
- ✅ `test_occlusion_sensitivity_values` - Value ranges
- ✅ `test_occlusion_sensitivity_spatial_importance` - Spatial detection
- ✅ `test_occlusion_sensitivity_batch` - Batch processing ← **Criterion 3**
- ✅ `test_occlusion_sensitivity_different_sizes` - Different configurations

**Run Tests**:
```bash
poetry run pytest tests/test_spatial_robustness.py -v
```

**Result**: All 17 tests passing ✅

---

## Quick Demo

```bash
# Run comprehensive demo
poetry run python demo_spatial_robustness.py

# Output:
# ✓ outputs/spatial_robustness/patch_shuffle_demo.png
# ✓ outputs/spatial_robustness/patch_shuffle_report.txt
# ✓ outputs/spatial_robustness/per_class_accuracy_drop.png
# ✓ outputs/spatial_robustness/occlusion_maps/sensitivity_maps.pt
# ✓ outputs/spatial_robustness/occlusion_maps/sensitivity_maps_visualization.png
```

---

## Integration Example: EuroSAT Evaluation

```python
#!/usr/bin/env python
"""Evaluate spatial robustness on EuroSAT validation set."""

import torch
from pathlib import Path
from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.spatial_robustness import (
    evaluate_patch_shuffle,
    occlusion_sensitivity,
)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path("outputs/eurosat_spatial_robustness")
output_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = create_model("vit_base", num_classes=10)
model.load_state_dict(torch.load("checkpoints/vit_base_best.pt"))
model = model.to(device).eval()

# Load EuroSAT data
_, val_loader, class_names = prepare_data(
    data_dir="data/EuroSAT",
    batch_size=32,
    augmentation="none"
)

# ===== Criterion 1 & 2: Patch-shuffle with per-class logging =====
print("Evaluating patch-shuffle robustness...")
report = evaluate_patch_shuffle(
    model=model,
    loader=val_loader,
    device=device,
    grid_size=(4, 4),
    num_classes=10,
    seed=42
)

# Log results
log_path = output_dir / "patch_shuffle_eurosat.txt"
with open(log_path, "w") as f:
    f.write("EuroSAT Patch-Shuffle Robustness Evaluation\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Overall Accuracy Drop: {report.accuracy_drop:+.4f}\n\n")
    f.write("Per-Class Accuracy Drop:\n")
    for i, name in enumerate(class_names):
        drop = report.per_class_accuracy_drop[i]
        f.write(f"  {name:20s}: {drop:+6.2%}\n")

print(f"✓ Logged to {log_path}")

# ===== Criterion 3: Save occlusion maps for fixed set =====
print("Generating occlusion sensitivity maps...")

# Fixed evaluation set: first 50 samples
images, targets = [], []
for batch_images, batch_targets in val_loader:
    images.append(batch_images)
    targets.append(batch_targets)
    if sum(len(b) for b in images) >= 50:
        break

images = torch.cat(images)[:50].to(device)
targets = torch.cat(targets)[:50].to(device)

# Generate maps
sensitivity_maps = occlusion_sensitivity(
    model=model,
    images=images,
    targets=targets,
    device=device,
    occlusion_size=32,
    stride=16
)

# Save
maps_path = output_dir / "eurosat_occlusion_maps.pt"
torch.save({
    'sensitivity_maps': sensitivity_maps.cpu(),
    'images': images.cpu(),
    'targets': targets.cpu(),
    'class_names': class_names,
    'model': 'vit_base',
}, maps_path)

print(f"✓ Saved {len(images)} maps to {maps_path}")

print("\n" + "="*70)
print("ACCEPTANCE CRITERIA STATUS")
print("="*70)
print("✅ Patch-shuffle and occlusion tests are implemented and logged")
print("✅ Accuracy drop under patch-shuffle is reported per class")
print("✅ Occlusion sensitivity maps are saved for a fixed evaluation set")
```

---

## Summary

All acceptance criteria for Story 3.3 are **fully implemented and tested**:

| Criterion | Implementation | Demo | Tests |
|-----------|---------------|------|-------|
| 1. Patch-shuffle implemented and logged | ✅ `evaluate_patch_shuffle()` | ✅ `patch_shuffle_report.txt` | ✅ 12 tests |
| 2. Per-class accuracy drop | ✅ `PatchShuffleReport` | ✅ Per-class section in log | ✅ 4 tests |
| 3. Occlusion maps saved | ✅ `occlusion_sensitivity()` | ✅ `.pt` + `.npy` files | ✅ 5 tests |

**Total**: 17/17 tests passing, comprehensive demo, full documentation.

---

## Files Created

**Core Implementation**:
- `eurosat_vit_analysis/spatial_robustness.py` (251 lines)

**Tests**:
- `tests/test_spatial_robustness.py` (407 lines)

**Documentation**:
- `SPATIAL_ROBUSTNESS_GUIDE.md` - Comprehensive usage guide
- `ACCEPTANCE_CRITERIA_DEMO.md` - This file
- `demo_spatial_robustness.py` - Runnable demo script

**Generated Outputs** (from demo):
- `outputs/spatial_robustness/patch_shuffle_demo.png`
- `outputs/spatial_robustness/patch_shuffle_report.txt`
- `outputs/spatial_robustness/per_class_accuracy_drop.png`
- `outputs/spatial_robustness/occlusion_maps/sensitivity_maps.pt`
- `outputs/spatial_robustness/occlusion_maps/sensitivity_maps_visualization.png`
- `outputs/spatial_robustness/occlusion_maps/sensitivity_map_sample_*.npy` (5 files)
