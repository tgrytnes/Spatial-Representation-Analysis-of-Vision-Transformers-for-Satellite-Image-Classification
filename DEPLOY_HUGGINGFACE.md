# Deploy to Hugging Face Spaces with GPU

This guide shows how to deploy the Streamlit dashboard to Hugging Face Spaces with FREE GPU acceleration (T4).

---

## Why Hugging Face Spaces?

- ✅ **Free GPU**: T4 GPU at no cost
- ✅ **Fast inference**: ~500ms vs 10-20s on CPU
- ✅ **Professional URL**: `https://huggingface.co/spaces/username/spatial-vit`
- ✅ **Auto-deployment**: Push to update
- ✅ **Zero maintenance**: Fully managed infrastructure

---

## Quick Start (5 minutes)

### Step 1: Create Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for a free account
3. Verify your email

### Step 2: Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Space name**: `spatial-vit-analysis` (or your choice)
   - **License**: `mit`
   - **SDK**: `Streamlit`
   - **Hardware**: `CPU basic` (we'll upgrade to GPU after)
   - **Visibility**: `Public`
3. Click **Create Space**

### Step 3: Push Your Code to the Space

```bash
# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/spatial-vit-analysis

# Rename README_HF.md to README.md for the Space
cp README_HF.md README.md

# Commit the README
git add README.md
git commit -m "Add Hugging Face Space README"

# Push to Hugging Face
git push hf main
```

**Note**: Replace `YOUR_USERNAME` with your actual Hugging Face username.

### Step 4: Upgrade to GPU

1. Go to your Space settings: `https://huggingface.co/spaces/YOUR_USERNAME/spatial-vit-analysis/settings`
2. Scroll to **Hardware** section
3. Click **Change hardware**
4. Select **T4 small** (FREE)
5. Click **Update**

The Space will restart with GPU enabled! 🎉

---

## Alternative: Deploy via Hugging Face CLI

### Install Hugging Face CLI

```bash
pip install huggingface_hub[cli]
```

### Login

```bash
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

### Create and Push

```bash
# Create a new Space
huggingface-cli repo create spatial-vit-analysis --type space --space_sdk streamlit

# Add remote and push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/spatial-vit-analysis
cp README_HF.md README.md
git add README.md
git commit -m "Add Hugging Face Space README"
git push hf main
```

---

## What Happens During Deployment

### 1. Hugging Face Detects Configuration

From `README_HF.md` (becomes `README.md` on the Space):
```yaml
---
title: Spatial ViT Analysis
emoji: 🛰️
sdk: streamlit
sdk_version: 1.52.2
app_file: app.py
hardware: t4-small  # GPU!
---
```

### 2. Dependencies Install

Hugging Face reads `requirements.txt`:
```
streamlit==1.52.2
torch==2.5.1          # With CUDA support
torchvision==0.20.1
timm==1.0.12
pillow==10.4.0
numpy==1.26.4
matplotlib==3.10.0
```

### 3. App Starts

- Streamlit runs `app.py`
- PyTorch detects GPU: `torch.cuda.is_available() == True`
- Green banner shows: "⚡ GPU Acceleration Enabled: Tesla T4"

---

## Verifying GPU is Working

Once deployed, check these indicators:

### 1. In the Dashboard
Look for the green success message at the top:
```
⚡ GPU Acceleration Enabled: Tesla T4
```

### 2. In the Logs
Click **Logs** in your Space, you should see:
```
CUDA available: True
Device: cuda
GPU: Tesla T4
```

### 3. Inference Speed
- **CPU**: 10-20 seconds per image
- **GPU**: 0.5-2 seconds per image ⚡

---

## Troubleshooting

### Issue 1: Space Stuck on "Building"

**Solution**: Check the logs for errors
```bash
# View logs
huggingface-cli repo logs YOUR_USERNAME/spatial-vit-analysis
```

Common causes:
- Missing `requirements.txt`
- Incompatible package versions
- Large model files not loading

### Issue 2: "No CUDA devices available"

**Cause**: Space is running on CPU hardware

**Solution**: Upgrade to GPU hardware
1. Go to Space settings
2. Hardware → Change hardware → T4 small
3. Click Update

### Issue 3: Out of Memory (OOM)

**Cause**: T4 has 16GB VRAM, but models + batch processing can exceed this

**Solution**: Add memory optimizations to `app.py`:
```python
@st.cache_resource
def load_model(model_name: str, use_checkpoint: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable memory efficient loading
    with torch.cuda.device(device):
        torch.cuda.empty_cache()  # Clear cache

    # Load model
    model, model_info = load_model_for_inference(...)

    return model, model_info, device
```

### Issue 4: Slow First Inference

**Cause**: Models download on first run (PyTorch ImageNet weights)

**Expected behavior**:
- First inference: ~30 seconds (downloading weights)
- Subsequent: ~0.5-2 seconds (cached)

**Solution**: Already handled by Streamlit's `@st.cache_resource`

---

## Configuration Options

### Hardware Tiers

| Hardware | GPU | VRAM | Cost | Best For |
|----------|-----|------|------|----------|
| **CPU basic** | None | - | Free | Testing |
| **T4 small** | T4 | 16GB | **Free** | Production ✅ |
| **T4 medium** | T4 | 16GB | $0.60/hr | High traffic |
| **A10G small** | A10G | 24GB | $1.05/hr | Large models |

For this project, **T4 small is perfect** - it's free and handles ViT/ResNet easily.

### Custom Domain (Optional)

You can set up a custom domain:
1. Go to Space settings
2. Scroll to **Custom domain**
3. Add your domain (e.g., `spatial-vit.yourdomain.com`)
4. Follow DNS instructions

---

## Performance Comparison

### CPU (Streamlit Cloud)
- Inference: 10-20 seconds
- Model loading: 30-60 seconds
- Total first request: ~90 seconds
- User experience: 😔 Slow

### GPU (Hugging Face Spaces T4)
- Inference: 0.5-2 seconds ⚡
- Model loading: 10-15 seconds
- Total first request: ~15 seconds
- User experience: 😊 Fast!

---

## Monitoring Your Space

### View Analytics

Go to: `https://huggingface.co/spaces/YOUR_USERNAME/spatial-vit-analysis/analytics`

You'll see:
- Total visitors
- Unique users
- Popular times
- Geographic distribution

### View Logs

Click **Logs** in your Space to see:
- Server startup
- Request handling
- Error messages
- Performance metrics

---

## Best Practices

### 1. Use Model Caching

Already implemented in `app.py`:
```python
@st.cache_resource
def load_model(model_name: str, use_checkpoint: bool = True):
    # Models cached after first load
    ...
```

### 2. Handle Checkpoint Loading Gracefully

```python
# In app.py - already configured
use_checkpoint = st.sidebar.checkbox(
    "Use Trained Checkpoints",
    value=False,  # Default to ImageNet for demo
    help="Load trained weights from DVC (if available)"
)
```

### 3. Show Loading Progress

Already implemented:
```python
with st.spinner(f"Running inference with {model_display}..."):
    result = predict_single_image(...)
```

### 4. Error Handling

Already implemented:
```python
try:
    model, model_info, device = load_model(model_1_name, use_checkpoint)
    result = predict_single_image(...)
except Exception as e:
    st.error(f"Error during inference: {str(e)}")
    st.exception(e)
```

---

## Updating Your Space

### Update Code

```bash
# Make changes locally
git add .
git commit -m "Update: your changes"

# Push to both GitHub and Hugging Face
git push origin main  # GitHub
git push hf main      # Hugging Face
```

The Space will automatically rebuild and redeploy! 🚀

### Update Configuration

Edit `README.md` on the Space directly or locally:
```yaml
---
title: Spatial ViT Analysis
hardware: a10g-small  # Upgrade to A10G
---
```

---

## Adding Your Trained Checkpoints

### Option A: Upload to Hugging Face Hub

```bash
# Install Hugging Face Hub
pip install huggingface_hub

# Login
huggingface-cli login

# Upload checkpoints
huggingface-cli upload YOUR_USERNAME/spatial-vit-checkpoints \
    checkpoints/vit_base_best.pt \
    checkpoints/resnet50_best.pt
```

Then update `app.py`:
```python
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model(model_name: str, use_checkpoint: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = None
    if use_checkpoint:
        # Download from Hugging Face
        checkpoint_path = hf_hub_download(
            repo_id="YOUR_USERNAME/spatial-vit-checkpoints",
            filename=f"{model_name}_best.pt"
        )

    model, model_info = load_model_for_inference(
        model_name, checkpoint_path=checkpoint_path, device=device
    )

    return model, model_info, device
```

### Option B: Use Git LFS

```bash
# Install Git LFS
git lfs install

# Track checkpoint files
git lfs track "checkpoints/*.pt"

# Commit and push
git add .gitattributes checkpoints/
git commit -m "Add checkpoints with Git LFS"
git push hf main
```

**Note**: Hugging Face Spaces supports Git LFS automatically.

---

## Example URLs

After deployment, your Space will be available at:

- **Main URL**: `https://huggingface.co/spaces/YOUR_USERNAME/spatial-vit-analysis`
- **Direct app**: `https://YOUR_USERNAME-spatial-vit-analysis.hf.space`
- **Embed**: `https://YOUR_USERNAME-spatial-vit-analysis.hf.space/?embed=true`

---

## Next Steps

After deploying:

1. ✅ Test the GPU-accelerated inference
2. ✅ Share the URL in your portfolio/resume
3. ✅ Add the URL to your GitHub README
4. ✅ Monitor analytics and logs
5. ✅ (Optional) Upload your trained checkpoints
6. ✅ (Optional) Add sample images to the Space

---

## Support

- **Hugging Face Docs**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Community Forum**: [discuss.huggingface.co](https://discuss.huggingface.co)
- **Discord**: [Hugging Face Discord](https://hf.co/join/discord)

---

## Summary

```bash
# Quick deployment commands
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/spatial-vit-analysis
cp README_HF.md README.md
git add README.md
git commit -m "Add Hugging Face Space README"
git push hf main

# Then go to Space settings and enable T4 GPU
```

**Result**: Your dashboard running on free GPU in ~5 minutes! 🚀

Built with ❤️ for Vision Transformer spatial analysis research
