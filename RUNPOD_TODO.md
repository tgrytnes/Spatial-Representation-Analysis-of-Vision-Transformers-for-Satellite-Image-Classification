# RunPod Evaluation TODO

## Epic 3: Explainability & Robustness - RunPod Evaluations

### ☐ Story 3.3: Spatial Robustness Evaluation

**Command:**
```bash
python evaluate_spatial_robustness.py --all
```

**What it does:**
- Evaluates patch-shuffle robustness for ViT and ResNet
- Tests how models respond to spatial structure disruption
- Generates occlusion sensitivity maps
- Creates per-class accuracy drop reports
- Saves visualizations and detailed reports

**Expected Output:**
```
outputs/spatial_robustness_evaluation/
├── vit_base/
│   ├── patch_shuffle_report.txt
│   ├── per_class_accuracy_drop.png
│   ├── grid_size_comparison.png
│   └── occlusion_maps/
│       ├── sensitivity_maps.pt
│       ├── sample_sensitivity_maps.png
│       └── map_*.npy
└── resnet50/
    └── (same structure)
```

**Acceptance Criteria:**
- ✅ Patch-shuffle tests are implemented and logged
- ✅ Accuracy drop under patch-shuffle is reported per class
- ✅ Occlusion sensitivity maps are saved for evaluation set

---

### ☐ Story 3.4: Calibration Evaluation

**Command:**
```bash
python evaluate_calibration.py --all
```

**What it does:**
- Computes Expected Calibration Error (ECE) before/after temperature scaling
- Applies temperature scaling to improve probability calibration
- Generates reliability diagrams for model comparison
- Saves detailed calibration reports

**Expected Output:**
```
outputs/calibration_evaluation/
├── vit_base/
│   ├── calibration_report.txt
│   └── reliability_diagram.png
├── resnet50/
│   ├── calibration_report.txt
│   └── reliability_diagram.png
└── model_comparison.png
```

**Acceptance Criteria:**
- ✅ Expected Calibration Error (ECE) is computed before/after temperature scaling
- ✅ Reliability diagrams are generated for ViT and ResNet

---

## Prerequisites

Before running on RunPod, ensure:

1. **Model checkpoints are available:**
   ```bash
   dvc pull checkpoints/vit_base_best.pt.dvc
   dvc pull checkpoints/resnet50_best.pt.dvc
   ```

2. **EuroSAT dataset is downloaded:**
   ```bash
   # Should be at: data/EuroSAT/
   ```

3. **Environment is set up:**
   ```bash
   poetry install
   ```

---

## Running Both Evaluations

You can run both evaluations sequentially:

```bash
# Run spatial robustness evaluation
echo "Running Spatial Robustness Evaluation..."
python evaluate_spatial_robustness.py --all

# Run calibration evaluation
echo "Running Calibration Evaluation..."
python evaluate_calibration.py --all

echo "All evaluations complete! Check outputs/ directory."
```

---

## Quick Reference

### Spatial Robustness Script Options
```bash
# Single model
python evaluate_spatial_robustness.py \
    --model vit_base \
    --checkpoint checkpoints/vit_base_best.pt

# Custom settings
python evaluate_spatial_robustness.py \
    --model resnet50 \
    --checkpoint checkpoints/resnet50_best.pt \
    --batch-size 64
```

### Calibration Script Options
```bash
# Single model
python evaluate_calibration.py \
    --model vit_base \
    --checkpoint checkpoints/vit_base_best.pt

# Custom bins
python evaluate_calibration.py \
    --model resnet50 \
    --checkpoint checkpoints/resnet50_best.pt \
    --num-bins 20
```

---

## Expected Results

### Spatial Robustness
- **ViT**: Expected to be more robust to patch shuffling (lower accuracy drop)
- **ResNet**: Expected to show larger accuracy drop (relies on spatial structure)
- **Occlusion maps**: Show which spatial regions are important for predictions

### Calibration
- **ViT**: Likely more overconfident (higher ECE, needs higher temperature)
- **ResNet**: Typically better calibrated out-of-the-box
- **Temperature scaling**: Should improve ECE for both models

---

## Notes

- Both evaluations use the validation set (20% of EuroSAT)
- Spatial robustness evaluation takes ~5-10 minutes per model
- Calibration evaluation takes ~2-3 minutes per model
- Total runtime: ~15-20 minutes for both evaluations on both models
- GPU recommended but not required (will use CPU if GPU unavailable)

---

## After Completion

Once evaluations are complete:

1. **Review reports:**
   - `outputs/spatial_robustness_evaluation/*/patch_shuffle_report.txt`
   - `outputs/calibration_evaluation/*/calibration_report.txt`

2. **Compare visualizations:**
   - Reliability diagrams
   - Occlusion sensitivity maps
   - Per-class accuracy drops

3. **Analyze results:**
   - How do ViT and ResNet differ in spatial reasoning?
   - Which model has better probability calibration?
   - What do occlusion maps reveal about learned features?

---

**Status:** Both implementations complete and tested. Ready to run on RunPod!
