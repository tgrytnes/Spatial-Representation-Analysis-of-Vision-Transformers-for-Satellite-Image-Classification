# Streamlit Dashboard User Guide

Interactive web interface for exploring Vision Transformer spatial analysis on satellite imagery.

---

## Quick Start

### 1. Start the Dashboard

```bash
# From project root
poetry run streamlit run app.py
```

The dashboard will open automatically at [http://localhost:8501](http://localhost:8501)

### 2. Upload an Image

- Click "Browse files" to upload a satellite image (PNG, JPG, JPEG)
- Or select from sample images (if available in `data/samples/`)
- Supported: EuroSAT imagery and similar satellite photos

### 3. Configure Models

**Single Model Mode:**
- Select one model from the dropdown
- View predictions and metrics for that model

**Compare Models Mode:**
- Select two different models
- View side-by-side comparison
- See which model is faster, more confident, and whether they agree

---

## Features

### Model Selection

Choose from available models:
- **ViT-Base**: Vision Transformer with 86M parameters
- **ResNet-50**: Residual CNN with 25M parameters
- **Swin-Tiny**: Swin Transformer (if checkpoint available)

### Model Information Display

For each model, the dashboard shows:
- **Architecture**: Model family (Transformer, CNN, etc.)
- **Parameters**: Total parameter count (formatted as M/K)
- **Inference Time**: Milliseconds to process the image

### Predictions

**Top-K Predictions:**
- Class names ranked by confidence
- Percentage confidence for each prediction
- Configurable via slider (1-10 predictions)

**Visualizations:**
- Horizontal bar chart of top predictions
- All 10 class probabilities (expandable)
- Color-coded bars highlighting top predictions

### Comparison Mode

When comparing two models, see:
- **Speed Winner**: Which model is faster
- **Confidence**: Which model is more certain
- **Agreement**: Do they predict the same class?

---

## EuroSAT Classes

The dashboard classifies images into 10 land use categories:

1. **AnnualCrop** - Annual cropland
2. **Forest** - Forest areas
3. **HerbaceousVegetation** - Herbaceous vegetation
4. **Highway** - Highway and major roads
5. **Industrial** - Industrial buildings
6. **Pasture** - Pasture land
7. **PermanentCrop** - Permanent cropland
8. **Residential** - Residential areas
9. **River** - Rivers and waterways
10. **SeaLake** - Seas and lakes

---

## Configuration Options

### Sidebar Settings

**Mode:**
- Single Model: Focus on one model
- Compare Models: Side-by-side comparison

**Top-K Predictions:**
- Slider to control how many predictions to show (1-10)

**Use Trained Checkpoints:**
- ✅ Checked: Load fine-tuned weights from `checkpoints/`
- ⬜ Unchecked: Use ImageNet pretrained weights

**Show Spatial Analysis:**
- Toggle to enable attention/occlusion overlays
- Select overlay type: attention or occlusion
- *(Full visualization coming in future update)*

---

## Using Trained Checkpoints

### With Checkpoints (Recommended)

If you've trained models or pulled checkpoints via DVC:

```bash
# Pull checkpoints from DVC remote
dvc pull checkpoints/vit_base_best.pt.dvc
dvc pull checkpoints/resnet50_best.pt.dvc

# Run dashboard
poetry run streamlit run app.py

# In dashboard: Keep "Use Trained Checkpoints" checked
```

### Without Checkpoints

The dashboard works with ImageNet pretrained weights:

```bash
# Just run the dashboard
poetry run streamlit run app.py

# In dashboard: Uncheck "Use Trained Checkpoints"
```

**Note:** ImageNet weights work but are not fine-tuned for EuroSAT, so predictions may be less accurate.

---

## Adding Sample Images

Create a samples directory for quick testing:

```bash
# Create samples directory
mkdir -p data/samples

# Add some EuroSAT images
cp data/EuroSAT/Forest/*.jpg data/samples/
cp data/EuroSAT/Highway/*.jpg data/samples/
# ... etc
```

Sample images will appear in a dropdown for quick selection.

---

## Troubleshooting

### "No module named 'eurosat_vit_analysis'"

**Solution:** Make sure you're in the project root and using Poetry:
```bash
cd /path/to/project
poetry run streamlit run app.py
```

### "Checkpoint not found"

**Solution:** Either pull checkpoints via DVC or disable checkpoint loading:
```bash
# Option 1: Pull checkpoints
dvc pull checkpoints/vit_base_best.pt.dvc

# Option 2: Use pretrained weights
# In dashboard sidebar, uncheck "Use Trained Checkpoints"
```

### Slow inference on CPU

**Expected behavior:** First inference takes ~5-10 seconds as models load. Subsequent inferences are faster (~0.5-2s per image) thanks to Streamlit caching.

**To speed up:**
- Use a GPU if available (auto-detected)
- Select smaller models (ResNet-50 is faster than ViT-Base)
- Close other applications to free up resources

### Dashboard won't start

**Check Streamlit installation:**
```bash
poetry show streamlit
# Should show: streamlit 1.52.2

# If not installed:
poetry install
```

**Check for port conflicts:**
```bash
# Streamlit uses port 8501 by default
# If occupied, specify a different port:
poetry run streamlit run app.py --server.port 8502
```

---

## Keyboard Shortcuts

When dashboard is focused:

- **R**: Rerun the app
- **C**: Clear cache
- **?**: Show all shortcuts

---

## Performance Tips

### Faster Inference

1. **Use GPU**: Dashboard auto-detects CUDA if available
2. **Cache models**: Streamlit caches loaded models - subsequent runs are faster
3. **Smaller images**: Dashboard resizes to 224x224, but smaller uploads transfer faster
4. **Close background apps**: Free up memory and CPU

### Better Predictions

1. **Use trained checkpoints**: Fine-tuned models perform much better than ImageNet pretrained
2. **Similar imagery**: Dashboard works best with EuroSAT-style satellite images
3. **Clear images**: Avoid heavily compressed or low-resolution uploads

---

## Example Workflows

### Workflow 1: Quick Testing

```bash
# Start dashboard
poetry run streamlit run app.py

# 1. Select "Single Model" mode
# 2. Choose "ViT-Base"
# 3. Upload a forest satellite image
# 4. Check if it predicts "Forest" with high confidence
```

### Workflow 2: Architecture Comparison

```bash
# Start dashboard
poetry run streamlit run app.py

# 1. Select "Compare Models" mode
# 2. Choose "ViT-Base" for Model 1
# 3. Choose "ResNet-50" for Model 2
# 4. Upload the same image
# 5. Compare:
#    - Which is faster?
#    - Which is more confident?
#    - Do they agree on the prediction?
```

### Workflow 3: Batch Analysis

```bash
# For multiple images, use the programmatic API:
poetry run python

>>> from PIL import Image
>>> import torch
>>> from eurosat_vit_analysis.inference import load_model_for_inference, predict_single_image
>>>
>>> device = torch.device("cpu")
>>> model, info = load_model_for_inference("vit_base")
>>>
>>> # Process multiple images
>>> for img_path in ["img1.jpg", "img2.jpg", "img3.jpg"]:
...     image = Image.open(img_path)
...     result = predict_single_image(model, image, info, device)
...     print(f"{img_path}: {result.predictions[0]} ({result.confidences[0]:.2%})")
```

---

## API Reference

For programmatic usage, see the inference module:

```python
from eurosat_vit_analysis.inference import (
    load_model_for_inference,
    predict_single_image,
    preprocess_image,
    EUROSAT_CLASSES,
)
```

**Functions:**
- `load_model_for_inference(model_name, checkpoint_path, device)` - Load a model
- `predict_single_image(model, image, model_info, device, top_k)` - Run inference
- `preprocess_image(image)` - Preprocess PIL/numpy image for model input

**Dataclasses:**
- `ModelInfo` - Model metadata (name, params, architecture)
- `InferenceResult` - Prediction results (predictions, confidences, timing)

See [eurosat_vit_analysis/inference.py](eurosat_vit_analysis/inference.py) for full documentation.

---

## Development

### Running Tests

```bash
# Test inference module
poetry run pytest tests/test_inference.py -v

# Expected: 22/22 tests passing
```

### Modifying the Dashboard

The dashboard code is in [app.py](app.py). Key sections:

- **Lines 30-47**: Configuration and model definitions
- **Lines 50-71**: Model loading with caching
- **Lines 74-157**: Visualization and display functions
- **Lines 160-356**: Main UI logic

After modifying, Streamlit auto-reloads when you save the file.

### Adding New Models

1. Add checkpoint to `AVAILABLE_MODELS` dict:
```python
AVAILABLE_MODELS = {
    "your_model": {
        "display_name": "Your Model",
        "checkpoint": "checkpoints/your_model_best.pt",
        "supports_attention": True,  # or False
    },
}
```

2. Ensure model is supported in `create_model()` function

3. Train and save checkpoint, or use pretrained weights

---

## Related Documentation

- [STORY_4_1_ACCEPTANCE_CRITERIA.md](STORY_4_1_ACCEPTANCE_CRITERIA.md) - Acceptance criteria details
- [tests/test_inference.py](tests/test_inference.py) - Test suite
- [eurosat_vit_analysis/inference.py](eurosat_vit_analysis/inference.py) - Backend logic
- [README.md](README.md) - Project overview

---

## Support

For issues or questions:
- Check the [troubleshooting section](#troubleshooting) above
- Review test cases in [tests/test_inference.py](tests/test_inference.py)
- Open an issue on [GitHub](https://github.com/tgrytnes/Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification)

---

Built with ❤️ for Vision Transformer spatial analysis research
