# Quick Start: NiceGUI + RunPod Setup

Get your dashboard running in 10 minutes!

---

## What You'll Get

```
Your Machine                          RunPod GPU Cloud
┌────────────────┐                    ┌──────────────┐
│   NiceGUI      │  ────API────►     │   FastAPI    │
│   Frontend     │  ◄───JSON────     │   + PyTorch  │
│  (Docker)      │                    │   + GPU      │
└────────────────┘                    └──────────────┘
  localhost:8080                       your-pod-url:8000
```

---

## Step 1: Deploy API to RunPod (5 minutes)

### 1.1 Build Docker Image

```bash
# Build the API image
docker build -f Dockerfile.runpod -t spatial-vit-api:latest .

# Test locally (optional, if you have GPU)
docker run --gpus all -p 8000:8000 spatial-vit-api:latest
```

### 1.2 Push to Docker Hub

```bash
# Login
docker login

# Tag and push
docker tag spatial-vit-api:latest YOUR_DOCKERHUB_USERNAME/spatial-vit-api:latest
docker push YOUR_DOCKERHUB_USERNAME/spatial-vit-api:latest
```

### 1.3 Create RunPod Pod

1. Go to [runpod.io](https://runpod.io) and login
2. Click **+ New Pod**
3. Select GPU:
   - **Secure Cloud**: RTX 4090 ($0.39/hr)
   - **Community Cloud**: Even cheaper!
4. Configure:
   - **Docker Image**: `YOUR_DOCKERHUB_USERNAME/spatial-vit-api:latest`
   - **Expose HTTP Ports**: `8000`
   - **Container Disk**: 20 GB
5. Click **Deploy**
6. Wait ~2 minutes for pod to start
7. **Copy the Connect URL** (looks like: `https://abc123-8000.proxy.runpod.net`)

### 1.4 Test API

```bash
# Replace with your actual URL
curl https://abc123-8000.proxy.runpod.net/health

# Should return:
# {"status":"healthy","device":"cuda:0","gpu_name":"NVIDIA GeForce RTX 4090", ...}
```

---

## Step 2: Run NiceGUI Frontend (2 minutes)

### Option A: Use Your Existing NiceGUI Docker

If you already have a NiceGUI container:

```bash
# Copy the app file
docker cp nicegui_app.py your-nicegui-container:/app/

# Copy the inference module
docker cp eurosat_vit_analysis/ your-nicegui-container:/app/

# Install dependencies
docker exec your-nicegui-container pip install httpx pillow matplotlib numpy

# Run the app
docker exec -d your-nicegui-container python /app/nicegui_app.py

# Access at: http://localhost:8080
```

### Option B: Build New Container

```bash
# Build frontend image
docker build -f Dockerfile.nicegui -t spatial-vit-frontend .

# Run with your RunPod URL
docker run -p 8080:8080 \
  -e API_URL=https://abc123-8000.proxy.runpod.net \
  spatial-vit-frontend

# Access at: http://localhost:8080
```

### Option C: Run Directly with Python

```bash
# Install dependencies
pip install -r requirements-nicegui.txt

# Edit nicegui_app.py line 18 with your RunPod URL:
# API_URL = "https://abc123-8000.proxy.runpod.net"

# Run
python nicegui_app.py

# Access at: http://localhost:8080
```

---

## Step 3: Test the Dashboard (1 minute)

1. Open http://localhost:8080 in your browser
2. You should see "🛰️ Spatial ViT Analysis"
3. In the sidebar:
   - **RunPod API URL**: Paste your RunPod URL
   - **Mode**: Select "single" or "compare"
   - **Model 1**: Select "vit_base"
4. Upload a test image (any satellite image or photo)
5. Wait ~1-2 seconds for results
6. You should see:
   - Top predictions with confidence scores
   - Model info (parameters, architecture)
   - Inference time (~500ms on GPU)
   - Confidence chart

---

## Troubleshooting

### Issue: "Connection Refused"

**Check API is accessible:**
```bash
curl https://your-runpod-url-8000.proxy.runpod.net/health
```

**Fix**: Make sure:
- RunPod pod is running (check RunPod dashboard)
- Port 8000 is exposed in pod settings
- URL is correct in NiceGUI (check sidebar)

### Issue: "Slow First Request"

**Cause**: Models download on first inference

**Fix**: Preload models
```bash
curl -X POST https://your-runpod-url-8000.proxy.runpod.net/preload \
  -H "Content-Type: application/json" \
  -d '["vit_base", "resnet50"]'
```

After this, subsequent requests will be fast (~500ms).

### Issue: "Out of Memory"

**Cause**: GPU VRAM exhausted

**Fix**: Restart RunPod pod or upgrade to larger GPU (A40 with 48GB)

---

## Next Steps

### Add Trained Checkpoints

To use your fine-tuned models instead of ImageNet pretrained:

```bash
# Option 1: Include in Docker image
# Add to Dockerfile.runpod:
COPY checkpoints/ ./checkpoints/

# Rebuild and push
docker build -f Dockerfile.runpod -t spatial-vit-api:latest .
docker push YOUR_DOCKERHUB_USERNAME/spatial-vit-api:latest

# Recreate RunPod pod with new image
```

### Run Both Locally (if you have GPU)

```bash
# Start both services
docker-compose up

# Frontend: http://localhost:8080
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Stop RunPod When Not Using

**Important**: You're charged while pod is running!

```bash
# In RunPod dashboard:
# Click pod → Stop
```

You only pay for actual usage time.

---

## Cost Estimate

**RunPod Pricing:**
- RTX 4090: $0.39/hour
- Example: 10 hours/week = $3.90/week = ~$15/month

**Free Alternative for Testing:**
- Use CPU mode (slow but free)
- Edit `api_server.py` to force CPU:
  ```python
  device = torch.device("cpu")
  ```

---

## Commands Cheat Sheet

```bash
# Build images
docker build -f Dockerfile.runpod -t spatial-vit-api:latest .
docker build -f Dockerfile.nicegui -t spatial-vit-frontend .

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/spatial-vit-api:latest

# Run frontend
docker run -p 8080:8080 -e API_URL=https://your-url spatial-vit-frontend

# Test API
curl https://your-runpod-url-8000.proxy.runpod.net/health

# Stop all
docker-compose down
```

---

## Getting Help

- **Deployment Guide**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions
- **RunPod Docs**: [docs.runpod.io](https://docs.runpod.io)
- **NiceGUI Docs**: [nicegui.io](https://nicegui.io)

---

You're all set! 🎉 Your dashboard should now be running with GPU-accelerated inference.
