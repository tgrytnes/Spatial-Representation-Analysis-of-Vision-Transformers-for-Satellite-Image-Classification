# Spatial Robustness Testing Guide - Epic 3.3

This guide shows how to use the spatial robustness functionality to quantify spatial reasoning and invariance in your models.

## Overview

The spatial robustness module provides three main capabilities:

1. **Patch Shuffling**: Tests if models rely on spatial structure by randomly shuffling image patches
2. **Per-Class Accuracy Tracking**: Reports accuracy drop for each class under spatial perturbations
3. **Occlusion Sensitivity Maps**: Identifies which spatial regions are most important for predictions

## Table of Contents

- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Integration with Training Pipeline](#integration-with-training-pipeline)
- [Interpreting Results](#interpreting-results)

---

## Quick Start

### Running the Demo

```bash
poetry run python demo_spatial_robustness.py
```

This creates visualizations and logs in `outputs/spatial_robustness/`:
- `patch_shuffle_demo.png` - Visual demonstration of patch shuffling
- `patch_shuffle_report.txt` - Logged evaluation results
- `per_class_accuracy_drop.png` - Per-class accuracy drop chart
- `occlusion_maps/` - Saved sensitivity maps and visualizations

---

## API Reference

### 1. Patch Shuffling

```python
from eurosat_vit_analysis.spatial_robustness import shuffle_patches

shuffled_images = shuffle_patches(
    images,       # torch.Tensor: (B, C, H, W)
    grid_size,    # tuple: (grid_h, grid_w) - e.g., (4, 4)
    seed=42       # int | None: for reproducibility
)
```

**What it does**: Divides each image into a grid of patches and randomly permutes them.

**Example**:
```python
import torch
from eurosat_vit_analysis.spatial_robustness import shuffle_patches

# Original image: 224x224
images = torch.randn(8, 3, 224, 224)

# Shuffle into 4x4 grid (56x56 patches)
shuffled = shuffle_patches(images, grid_size=(4, 4), seed=42)

# shuffled.shape == images.shape
# But spatial structure is disrupted
```

---

### 2. Patch-Shuffle Evaluation

```python
from eurosat_vit_analysis.spatial_robustness import evaluate_patch_shuffle

report = evaluate_patch_shuffle(
    model,        # nn.Module: your trained model
    loader,       # DataLoader: evaluation data
    device,       # torch.device: 'cuda' or 'cpu'
    grid_size,    # tuple: (grid_h, grid_w)
    num_classes,  # int: number of classes (10 for EuroSAT)
    seed=42       # int | None: for reproducibility
)

# Returns PatchShuffleReport with:
#   - clean_accuracy: float
#   - shuffled_accuracy: float
#   - accuracy_drop: float
#   - per_class_clean_accuracy: dict[int, float]
#   - per_class_shuffled_accuracy: dict[int, float]
#   - per_class_accuracy_drop: dict[int, float]
```

**Acceptance Criteria Coverage**:
- ✅ Patch-shuffle tests are implemented and logged
- ✅ Accuracy drop is reported per class

**Example with EuroSAT**:
```python
from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.spatial_robustness import evaluate_patch_shuffle
import torch

# Load trained model
model = create_model("vit_base", num_classes=10)
model.load_state_dict(torch.load("checkpoints/vit_base_best.pt"))
model.eval()

# Load EuroSAT validation data
_, val_loader, class_names = prepare_data(
    data_dir="data/EuroSAT",
    batch_size=32,
    augmentation="none"
)

# Evaluate with patch shuffle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

report = evaluate_patch_shuffle(
    model=model,
    loader=val_loader,
    device=device,
    grid_size=(4, 4),  # 224/4 = 56x56 patches
    num_classes=10,
    seed=42
)

# Log results
print(f"Clean Accuracy: {report.clean_accuracy:.2%}")
print(f"Shuffled Accuracy: {report.shuffled_accuracy:.2%}")
print(f"Accuracy Drop: {report.accuracy_drop:.2%}")

print("\nPer-Class Results:")
for class_id, class_name in enumerate(class_names):
    drop = report.per_class_accuracy_drop[class_id]
    print(f"  {class_name:20s}: {drop:+.2%}")
```

---

### 3. Occlusion Sensitivity Maps

```python
from eurosat_vit_analysis.spatial_robustness import occlusion_sensitivity

sensitivity_maps = occlusion_sensitivity(
    model,           # nn.Module: your trained model
    images,          # torch.Tensor: (B, C, H, W)
    targets,         # torch.Tensor: (B,) - target class indices
    device,          # torch.device
    occlusion_size,  # int: size of occluding square (e.g., 32)
    stride           # int: sliding window stride (e.g., 16)
)

# Returns: torch.Tensor of shape (B, H_steps, W_steps)
# where H_steps = (H - occlusion_size) // stride + 1
```

**What it does**: Slides an occluding square across the image and measures how much the prediction confidence drops when each region is occluded.

**Acceptance Criteria Coverage**:
- ✅ Occlusion sensitivity maps can be saved for evaluation sets

**Example - Generate and Save Maps**:
```python
import torch
from pathlib import Path
from eurosat_vit_analysis.spatial_robustness import occlusion_sensitivity

# Get a fixed evaluation batch
images, targets = next(iter(val_loader))
images = images.to(device)
targets = targets.to(device)

# Generate sensitivity maps
sensitivity_maps = occlusion_sensitivity(
    model=model,
    images=images,
    targets=targets,
    device=device,
    occlusion_size=32,  # 32x32 occluding square
    stride=16           # Slide by 16 pixels
)

# Save for later analysis
output_dir = Path("outputs/occlusion_sensitivity")
output_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'maps': sensitivity_maps.cpu(),
    'images': images.cpu(),
    'targets': targets.cpu(),
    'class_names': class_names
}, output_dir / "eurosat_sensitivity_maps.pt")

print(f"Saved {len(images)} sensitivity maps to {output_dir}")
```

---

## Usage Examples

### Example 1: Compare Models

Compare how ResNet50 and ViT respond to patch shuffling:

```python
from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.spatial_robustness import evaluate_patch_shuffle

models = {
    "ResNet50": create_model("resnet50", num_classes=10),
    "ViT-Base": create_model("vit_base", num_classes=10)
}

results = {}
for name, model in models.items():
    model.load_state_dict(torch.load(f"checkpoints/{name}_best.pt"))
    model = model.to(device).eval()

    report = evaluate_patch_shuffle(
        model, val_loader, device, (4, 4), 10, seed=42
    )

    results[name] = report.accuracy_drop
    print(f"{name}: {report.accuracy_drop:+.2%} accuracy drop")

# Hypothesis: CNNs rely more on local patterns,
# ViTs may be more robust to spatial shuffling
```

### Example 2: Test Different Grid Sizes

Determine how sensitive models are to different levels of spatial disruption:

```python
grid_sizes = [(2, 2), (4, 4), (8, 8), (14, 14)]

for grid_size in grid_sizes:
    patches_per_dim = grid_size[0]
    patch_size = 224 // patches_per_dim

    report = evaluate_patch_shuffle(
        model, val_loader, device, grid_size, 10, seed=42
    )

    print(f"Grid {grid_size} ({patch_size}px patches): "
          f"{report.accuracy_drop:+.2%} drop")
```

### Example 3: Identify Spatially-Sensitive Classes

Find which EuroSAT classes are most affected by spatial disruption:

```python
report = evaluate_patch_shuffle(
    model, val_loader, device, (4, 4), 10, seed=42
)

# Sort classes by accuracy drop
class_drops = [
    (class_names[i], report.per_class_accuracy_drop[i])
    for i in range(10)
]
class_drops.sort(key=lambda x: x[1], reverse=True)

print("Most affected classes:")
for class_name, drop in class_drops[:5]:
    print(f"  {class_name:20s}: {drop:+.2%}")

# Example interpretation:
# - "SeaLake" might show large drop (spatial context matters)
# - "Forest" might show small drop (texture-based)
```

### Example 4: Visualize Occlusion Sensitivity

Create heatmaps showing important regions:

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

# Get one image
images, targets = next(iter(val_loader))
image = images[0:1].to(device)
target = targets[0:1].to(device)

# Generate sensitivity map
sensitivity = occlusion_sensitivity(
    model, image, target, device, occlusion_size=32, stride=16
)[0].cpu().numpy()

# Resize to match image
scale = 224 / sensitivity.shape[0]
sensitivity_resized = zoom(sensitivity, (scale, scale), order=1)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
img = images[0].permute(1, 2, 0).numpy()
img = (img - img.min()) / (img.max() - img.min())
axes[0].imshow(img)
axes[0].set_title("Original Image")

# Sensitivity map
axes[1].imshow(sensitivity, cmap='hot')
axes[1].set_title("Sensitivity Map")
axes[1].colorbar()

# Overlay
axes[2].imshow(img)
axes[2].imshow(sensitivity_resized, cmap='hot', alpha=0.6)
axes[2].set_title("Overlay")

plt.savefig("outputs/sensitivity_example.png")
```

---

## Integration with Training Pipeline

### During Model Evaluation

Add spatial robustness evaluation to your standard evaluation script:

```python
# eval_spatial_robustness.py
import torch
from pathlib import Path
from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.spatial_robustness import (
    evaluate_patch_shuffle,
    occlusion_sensitivity
)

def evaluate_spatial_robustness(
    checkpoint_path: str,
    model_name: str,
    output_dir: str = "outputs/spatial_robustness"
):
    """Evaluate spatial robustness of a trained model."""

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    # Load data
    _, val_loader, class_names = prepare_data(
        "data/EuroSAT", batch_size=32, augmentation="none"
    )

    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Patch-shuffle evaluation
    print("Evaluating patch-shuffle robustness...")
    report = evaluate_patch_shuffle(
        model, val_loader, device, (4, 4), 10, seed=42
    )

    # Log results
    with open(output_path / "patch_shuffle_report.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write(f"Clean Accuracy: {report.clean_accuracy:.4f}\n")
        f.write(f"Shuffled Accuracy: {report.shuffled_accuracy:.4f}\n")
        f.write(f"Accuracy Drop: {report.accuracy_drop:.4f}\n\n")
        f.write("Per-Class Accuracy Drop:\n")
        for i, name in enumerate(class_names):
            drop = report.per_class_accuracy_drop[i]
            f.write(f"  {name:20s}: {drop:+.4f}\n")

    print(f"✓ Logged to {output_path / 'patch_shuffle_report.txt'}")

    # 2. Generate occlusion sensitivity maps for fixed set
    print("Generating occlusion sensitivity maps...")

    # Use first batch as fixed evaluation set
    images, targets = next(iter(val_loader))
    images, targets = images.to(device), targets.to(device)

    sensitivity_maps = occlusion_sensitivity(
        model, images, targets, device, occlusion_size=32, stride=16
    )

    # Save maps
    torch.save({
        'maps': sensitivity_maps.cpu(),
        'images': images.cpu(),
        'targets': targets.cpu(),
        'class_names': class_names,
        'model': model_name
    }, output_path / "occlusion_maps.pt")

    print(f"✓ Saved {len(images)} maps to {output_path / 'occlusion_maps.pt'}")

    return report

if __name__ == "__main__":
    # Evaluate all models
    models = [
        ("checkpoints/resnet50_best.pt", "resnet50"),
        ("checkpoints/vit_base_best.pt", "vit_base"),
    ]

    for checkpoint, model_name in models:
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name}")
        print(f"{'='*70}")
        evaluate_spatial_robustness(checkpoint, model_name)
```

---

## Interpreting Results

### Patch-Shuffle Results

**Accuracy Drop Interpretation:**

- **Large positive drop** (e.g., +20%): Model heavily relies on spatial structure
  - Example: "SeaLake" class might need spatial context to distinguish from ocean
  - Suggests model uses global spatial patterns

- **Small drop** (e.g., ±5%): Model is spatially invariant
  - Example: "Forest" might be recognized by local textures
  - Suggests model uses local features more than spatial arrangement

- **Negative drop** (accuracy improves): Unusual but possible
  - May indicate model overfits to specific spatial patterns
  - Random shuffling occasionally creates better patterns by chance

**Architecture Differences:**

- **CNNs (ResNet)**: Typically show larger drops
  - Strong inductive bias toward local spatial patterns
  - Hierarchical feature extraction depends on spatial arrangement

- **Vision Transformers (ViT)**: May show smaller drops
  - Patch-based processing with position embeddings
  - Self-attention can adapt to permuted patches
  - BUT: depends on how position embeddings are handled

### Occlusion Sensitivity Maps

**Reading the Maps:**

- **Bright/Hot regions** (high values): Critical for prediction
  - Occluding these areas significantly drops confidence
  - Model "looks here" to make decisions

- **Dark/Cold regions** (low values): Less important
  - Occluding has minimal effect
  - Model ignores these areas

**Use Cases:**

1. **Debug misclassifications**: See what the model focused on
2. **Verify learned features**: Check if model uses semantic regions
3. **Compare architectures**: CNNs vs. ViTs may focus on different areas
4. **Detect biases**: Model might use shortcuts (e.g., watermarks)

**Example Insights for EuroSAT:**

- "Highway": Should highlight road structures
- "River": Should focus on water boundaries
- "Forest": Might show uniform sensitivity (texture-based)
- "Residential": Should highlight building patterns

---

## Best Practices

1. **Grid Size Selection**:
   - Start with (4, 4) for 224x224 images (56x56 patches)
   - Try multiple sizes: (2,2), (4,4), (8,8), (14,14)
   - Smaller grid = less disruption, larger grid = more disruption

2. **Reproducibility**:
   - Always set `seed` parameter for consistent results
   - Use same seed across model comparisons

3. **Computational Cost**:
   - Occlusion sensitivity: O(H_steps × W_steps) forward passes
   - Use larger stride to speed up (e.g., stride=32 instead of 16)
   - Process in batches when possible

4. **Saving Results**:
   - Save both raw tensors (.pt) and visualizations (.png)
   - Include metadata (model name, hyperparameters)
   - Version control your evaluation scripts

---

## Troubleshooting

**Q: Accuracy drop is negative (model improves with shuffling)?**

A: This can happen with small evaluation sets due to randomness. Use larger sets or multiple seeds and average results.

**Q: Occlusion maps are all uniform?**

A: Model might be using global average pooling, making all regions equally important. Try different occlusion sizes.

**Q: CUDA out of memory with occlusion sensitivity?**

A: Reduce batch size or increase stride. Occlusion requires many forward passes.

**Q: How do I compare ViT vs. ResNet fairly?**

A: Ensure both models have similar capacity and are trained with same settings. Compare relative drops, not absolute accuracies.

---

## References

**Related Papers:**
- *Attention is Not Explanation* (Jain & Wallace, 2019)
- *Visualizing and Understanding Convolutional Networks* (Zeiler & Fergus, 2014)
- *An Image is Worth 16x16 Words* (Dosovitskiy et al., 2021)

**Epic 3.3 Acceptance Criteria:**
- ✅ Patch-shuffle and occlusion tests are implemented and logged
- ✅ Accuracy drop under patch-shuffle is reported per class
- ✅ Occlusion sensitivity maps are saved for a fixed evaluation set

---

For more information, see:
- [eurosat_vit_analysis/spatial_robustness.py](eurosat_vit_analysis/spatial_robustness.py) - Implementation
- [tests/test_spatial_robustness.py](tests/test_spatial_robustness.py) - Test suite
- [demo_spatial_robustness.py](demo_spatial_robustness.py) - Demo script
