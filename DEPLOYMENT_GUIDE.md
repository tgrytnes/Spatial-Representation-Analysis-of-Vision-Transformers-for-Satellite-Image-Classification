# Deployment Guide: NiceGUI + RunPod Architecture

This guide explains how to deploy the Spatial ViT Analysis dashboard using NiceGUI for the frontend and RunPod for GPU-accelerated inference.

---

## Architecture Overview

```
┌─────────────────┐         HTTP API          ┌──────────────────┐
│                 │   ────────────────────►   │                  │
│  NiceGUI        │                            │  FastAPI Server  │
│  Frontend       │   ◄────────────────────   │  (RunPod GPU)    │
│  (Local Docker) │         JSON Response      │                  │
└─────────────────┘                            └──────────────────┘
     Port 8080                                      Port 8000
```

**Components:**
1. **NiceGUI Frontend**: User interface running in Docker locally
2. **FastAPI API**: Inference backend running on RunPod with GPU

**Benefits:**
- ⚡ Fast GPU inference on RunPod
- 🏠 Local UI with no deployment needed
- 💰 Pay only for GPU time you use
- 🔄 Easy to update and maintain

---

## Quick Start

### 1. Deploy API to RunPod

```bash
# Build and push Docker image
docker build -f Dockerfile.runpod -t your-dockerhub-username/spatial-vit-api:latest .
docker push your-dockerhub-username/spatial-vit-api:latest

# Or use the pre-built image (once available)
```

Then in RunPod:
1. Create new pod with GPU (RTX 4090, A100, etc.)
2. Use Docker image: `your-dockerhub-username/spatial-vit-api:latest`
3. Expose port 8000
4. Note the pod URL (e.g., `https://xyz123-8000.proxy.runpod.net`)

### 2. Run NiceGUI Frontend Locally

```bash
# Update API URL in nicegui_app.py or via environment variable
export API_URL=https://xyz123-8000.proxy.runpod.net

# Run with Docker
docker build -f Dockerfile.nicegui -t spatial-vit-frontend .
docker run -p 8080:8080 -e API_URL=$API_URL spatial-vit-frontend

# Or run directly with Python
pip install -r requirements-nicegui.txt
python nicegui_app.py
```

3. Open browser: http://localhost:8080

---

## Detailed Setup

### Part 1: RunPod API Deployment

#### Step 1: Prepare Docker Image

```bash
# Build the API server image
docker build -f Dockerfile.runpod -t spatial-vit-api:latest .

# Test locally (if you have GPU)
docker run --gpus all -p 8000:8000 spatial-vit-api:latest

# Check health
curl http://localhost:8000/health
```

#### Step 2: Push to Docker Registry

**Option A: Docker Hub**
```bash
docker login
docker tag spatial-vit-api:latest YOUR_USERNAME/spatial-vit-api:latest
docker push YOUR_USERNAME/spatial-vit-api:latest
```

**Option B: GitHub Container Registry**
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
docker tag spatial-vit-api:latest ghcr.io/YOUR_USERNAME/spatial-vit-api:latest
docker push ghcr.io/YOUR_USERNAME/spatial-vit-api:latest
```

#### Step 3: Create RunPod Pod

1. Go to [runpod.io](https://runpod.io) and sign in
2. Click **+ New Pod**
3. Configure:
   - **GPU**: RTX 4090 ($0.39/hr) or A40 ($0.79/hr)
   - **Template**: Select "Custom"
   - **Docker Image**: `YOUR_USERNAME/spatial-vit-api:latest`
   - **Expose HTTP Ports**: `8000`
   - **Container Disk**: 20 GB
   - **Volume**: Optional (for checkpoints)

4. Click **Deploy**
5. Wait for pod to start (~2-3 minutes)
6. Copy the **Connect** URL (e.g., `https://xyz123-8000.proxy.runpod.net`)

#### Step 4: Verify API is Running

```bash
# Replace with your actual RunPod URL
API_URL=https://xyz123-8000.proxy.runpod.net

# Check health
curl $API_URL/health

# Expected response:
{
  "status": "healthy",
  "device": "cuda:0",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "cuda_available": true,
  "models_loaded": []
}
```

---

### Part 2: NiceGUI Frontend Setup

#### Option 1: Run with Docker (Recommended)

```bash
# Build frontend image
docker build -f Dockerfile.nicegui -t spatial-vit-frontend .

# Run container
docker run -p 8080:8080 \
  -e API_URL=https://your-runpod-url-8000.proxy.runpod.net \
  spatial-vit-frontend

# Access at: http://localhost:8080
```

#### Option 2: Run with Python Directly

```bash
# Install dependencies
pip install -r requirements-nicegui.txt

# Set API URL
export API_URL=https://your-runpod-url-8000.proxy.runpod.net

# Run app
python nicegui_app.py

# Or edit nicegui_app.py line 18:
# API_URL = "https://your-runpod-url-8000.proxy.runpod.net"
```

#### Option 3: Use Existing NiceGUI Docker Container

If you already have a NiceGUI container running:

```bash
# Copy files to your container
docker cp nicegui_app.py your-nicegui-container:/app/
docker cp eurosat_vit_analysis/ your-nicegui-container:/app/

# Install dependencies in container
docker exec your-nicegui-container pip install httpx pillow matplotlib numpy

# Restart container or run the app
docker exec -d your-nicegui-container python /app/nicegui_app.py
```

---

## Configuration

### API URL Configuration

The frontend needs to know where the API is. Three ways to set this:

**1. Environment Variable (Recommended)**
```bash
export API_URL=https://your-runpod-url-8000.proxy.runpod.net
python nicegui_app.py
```

**2. Edit nicegui_app.py**
```python
# Line 18
API_URL = "https://your-runpod-url-8000.proxy.runpod.net"
```

**3. Configure in UI** (Already built-in)
- Open the app
- Enter RunPod URL in the "RunPod API URL" field in sidebar
- Changes persist in session state

### Using Trained Checkpoints

To use your trained models instead of ImageNet pretrained:

#### Option 1: Include in Docker Image
```dockerfile
# Add to Dockerfile.runpod
COPY checkpoints/ ./checkpoints/
```

Then rebuild and push:
```bash
docker build -f Dockerfile.runpod -t spatial-vit-api:latest .
docker push YOUR_USERNAME/spatial-vit-api:latest
```

#### Option 2: Mount RunPod Volume
1. Create a Network Volume in RunPod
2. Upload checkpoints to the volume
3. Attach volume to your pod at `/app/checkpoints`

```bash
# In RunPod pod settings:
# Volume Mount Path: /app/checkpoints
```

---

## Testing

### Test API Locally

```bash
# Start API server
python api_server.py

# In another terminal, test with curl
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -F "model_name=vit_base" \
  -F "top_k=3"
```

### Test Full Stack Locally

```bash
# Start both services
docker-compose up

# Access frontend at http://localhost:8080
# API available at http://localhost:8000
```

**Requirements:**
- Docker with GPU support (for API)
- NVIDIA Docker runtime installed

---

## Monitoring

### View API Logs

```bash
# In RunPod pod terminal
tail -f /var/log/api.log

# Or view in RunPod web interface
```

### Check GPU Usage

```bash
# In RunPod pod terminal
nvidia-smi

# Watch in real-time
watch -n 1 nvidia-smi
```

### API Metrics

Access FastAPI docs: `https://your-runpod-url-8000.proxy.runpod.net/docs`

Available endpoints:
- `GET /health` - Health check
- `POST /predict` - Run inference
- `GET /models` - List loaded models
- `POST /preload` - Preload models into memory

---

## Cost Optimization

### RunPod Pricing (as of 2024)

| GPU | VRAM | Price/hr | Best For |
|-----|------|----------|----------|
| RTX 4090 | 24GB | $0.39 | Development/Demo |
| A40 | 48GB | $0.79 | Production |
| A100 | 80GB | $1.89 | Heavy workloads |

### Tips to Save Money

1. **Use Spot Instances**: 50% cheaper but can be interrupted
2. **Stop When Not Using**: Pay only when pod is running
3. **Preload Models**: Reduce cold start time
4. **Batch Requests**: Process multiple images at once
5. **Use Smaller GPU**: RTX 4090 is sufficient for ViT/ResNet

### Auto-Shutdown

Add to `api_server.py`:
```python
import asyncio
from datetime import datetime, timedelta

last_request_time = datetime.now()

@app.middleware("http")
async def auto_shutdown_middleware(request, call_next):
    global last_request_time
    last_request_time = datetime.now()
    return await call_next(request)

async def check_idle_shutdown():
    """Shutdown if idle for 30 minutes"""
    while True:
        await asyncio.sleep(60)
        if datetime.now() - last_request_time > timedelta(minutes=30):
            print("Idle timeout, shutting down...")
            os.system("runpodctl stop pod")
```

---

## Troubleshooting

### Issue 1: "Connection Refused"

**Cause**: API not accessible from frontend

**Solution**:
```bash
# Check API is running
curl https://your-runpod-url-8000.proxy.runpod.net/health

# Check firewall/CORS settings
# Ensure RunPod port 8000 is exposed
```

### Issue 2: Slow Inference

**Cause**: Models downloading on first request

**Solution**: Preload models
```bash
curl -X POST https://your-runpod-url-8000.proxy.runpod.net/preload \
  -H "Content-Type: application/json" \
  -d '["vit_base", "resnet50"]'
```

### Issue 3: Out of Memory

**Cause**: GPU VRAM exhausted

**Solution**:
- Use smaller batch sizes
- Clear cache between requests
- Upgrade to GPU with more VRAM

### Issue 4: Docker Image Too Large

**Cause**: Including unnecessary files

**Solution**: Add `.dockerignore`
```
.git/
*.pyc
__pycache__/
tests/
docs/
*.md
outputs/
data/
```

---

## Advanced: Production Deployment

### Load Balancing Multiple RunPod Instances

```python
# In nicegui_app.py
import random

API_URLS = [
    "https://pod1-8000.proxy.runpod.net",
    "https://pod2-8000.proxy.runpod.net",
    "https://pod3-8000.proxy.runpod.net",
]

async def call_inference_api(image_bytes, model_name, top_k=3):
    api_url = random.choice(API_URLS)  # Simple load balancing
    # ... rest of code
```

### Add Authentication

```python
# In api_server.py
from fastapi import Header, HTTPException

API_KEY = "your-secret-key"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Header(..., alias="X-API-Key")
):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # ... rest of code
```

### Monitor with Prometheus

```python
# Add to api_server.py
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('inference_requests_total', 'Total inference requests')
inference_time = Histogram('inference_duration_seconds', 'Inference time')

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

---

## Next Steps

1. ✅ Deploy API to RunPod
2. ✅ Test API health endpoint
3. ✅ Run NiceGUI frontend locally
4. ✅ Upload test images and verify predictions
5. ✅ (Optional) Add your trained checkpoints
6. ✅ (Optional) Set up monitoring and alerts

---

## Support

- **RunPod Docs**: [docs.runpod.io](https://docs.runpod.io)
- **NiceGUI Docs**: [nicegui.io](https://nicegui.io)
- **FastAPI Docs**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

---

Built with ❤️ for Vision Transformer spatial analysis research
