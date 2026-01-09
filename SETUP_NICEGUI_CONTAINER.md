# Setup NiceGUI in Your Existing Docker Container

Quick guide to copy the NiceGUI app to your existing Docker container.

---

## Option 1: Automated Script (Recommended)

Run the setup script:

```bash
./setup_nicegui_container.sh
```

The script will:
1. List your running containers
2. Ask you to select which container to use
3. Copy all necessary files
4. Install dependencies
5. Provide next steps

---

## Option 2: Manual Setup

### Step 1: Find Your Container

```bash
# List running containers
docker ps

# Output example:
# CONTAINER ID   NAME           STATUS         PORTS
# abc123def456   nicegui-app    Up 2 hours     0.0.0.0:8080->8080/tcp
```

Note your container NAME or ID (e.g., `nicegui-app` or `abc123def456`)

### Step 2: Copy Files to Container

```bash
# Replace 'nicegui-app' with your actual container name

# Create app directory
docker exec nicegui-app mkdir -p /app/eurosat_vit_analysis

# Copy main app
docker cp nicegui_app.py nicegui-app:/app/

# Copy inference module
docker cp eurosat_vit_analysis/inference.py nicegui-app:/app/eurosat_vit_analysis/
docker cp eurosat_vit_analysis/__init__.py nicegui-app:/app/eurosat_vit_analysis/
docker cp eurosat_vit_analysis/models.py nicegui-app:/app/eurosat_vit_analysis/
```

### Step 3: Install Dependencies

```bash
# Install required packages
docker exec nicegui-app pip install httpx pillow matplotlib numpy
```

### Step 4: Configure API URL

You have two options:

**Option A: Edit the file (before copying)**
```bash
# Edit nicegui_app.py line 16
# API_URL = "http://localhost:8000"  # Change to your RunPod URL
```

**Option B: Configure in the UI (after starting)**
- Enter your RunPod URL in the sidebar when the app is running

### Step 5: Start the App

```bash
# Run in background
docker exec -d nicegui-app python /app/nicegui_app.py

# Or run interactively to see output
docker exec -it nicegui-app python /app/nicegui_app.py
```

### Step 6: Access the Dashboard

Open your browser:
- If container exposes port 8080: `http://localhost:8080`
- Check with: `docker port nicegui-app`

---

## Troubleshooting

### Issue: "No such file or directory"

**Cause**: Files not in the current directory

**Solution**: Run commands from project root
```bash
cd /Users/thomasfey-grytnes/Documents/Artificial\ Intelligence\ -\ Studying/Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification/
./setup_nicegui_container.sh
```

### Issue: "Container not found"

**Cause**: Container not running

**Solution**: Start your container
```bash
# If using docker-compose
docker-compose up -d

# Or start a stopped container
docker start your-container-name
```

### Issue: Port already in use

**Cause**: Another app using port 8080

**Solution**: Stop the other app or change port
```bash
# Edit nicegui_app.py line 367:
# ui.run(port=8081)  # Change to different port

# Or stop the conflicting process
lsof -ti:8080 | xargs kill
```

### Issue: "Module not found"

**Cause**: Python package not installed

**Solution**: Install dependencies again
```bash
docker exec nicegui-app pip install --upgrade httpx pillow matplotlib numpy
```

### Issue: App won't start

**Check logs:**
```bash
# View container logs
docker logs nicegui-app

# Or exec into container and check
docker exec -it nicegui-app bash
python /app/nicegui_app.py
```

---

## Verifying Setup

### Check Files Were Copied

```bash
# List files in container
docker exec nicegui-app ls -la /app/

# Should show:
# nicegui_app.py
# eurosat_vit_analysis/

# Check module files
docker exec nicegui-app ls -la /app/eurosat_vit_analysis/

# Should show:
# __init__.py
# inference.py
# models.py
```

### Check Dependencies

```bash
docker exec nicegui-app pip list | grep -E 'httpx|pillow|matplotlib|numpy'

# Should show versions:
# httpx          0.27.2
# matplotlib     3.10.0
# numpy          1.26.4
# pillow         10.4.0
```

### Check App is Running

```bash
# Check for Python process
docker exec nicegui-app pgrep -f nicegui_app.py

# If it returns a number, the app is running
```

### Test the Dashboard

```bash
# Check if port is accessible
curl http://localhost:8080

# Should return HTML content
```

---

## Updating Files

If you make changes to the code:

```bash
# Stop the app
docker exec nicegui-app pkill -f nicegui_app.py

# Copy updated files
docker cp nicegui_app.py nicegui-app:/app/
docker cp eurosat_vit_analysis/inference.py nicegui-app:/app/eurosat_vit_analysis/

# Restart the app
docker exec -d nicegui-app python /app/nicegui_app.py
```

---

## Stopping the App

```bash
# Stop the Python process
docker exec nicegui-app pkill -f nicegui_app.py

# Or stop the entire container
docker stop nicegui-app
```

---

## Alternative: Rebuild Container

If you prefer to rebuild your container with the files included:

### 1. Create a new Dockerfile

```dockerfile
# Dockerfile
FROM your-existing-nicegui-image

WORKDIR /app

# Copy files
COPY nicegui_app.py .
COPY eurosat_vit_analysis/ ./eurosat_vit_analysis/

# Install dependencies
RUN pip install httpx pillow matplotlib numpy

# Run app
CMD ["python", "nicegui_app.py"]
```

### 2. Build and run

```bash
docker build -t my-spatial-vit-nicegui .
docker run -p 8080:8080 my-spatial-vit-nicegui
```

---

## Next Steps After Setup

1. **Deploy API to RunPod** (see [QUICKSTART.md](QUICKSTART.md))
2. **Configure RunPod URL** in the NiceGUI sidebar
3. **Upload test images** to verify it works
4. **(Optional)** Add your trained checkpoints to RunPod

---

## Quick Command Reference

```bash
# List containers
docker ps

# Copy to container
docker cp <source> <container>:<dest>

# Execute command in container
docker exec <container> <command>

# View logs
docker logs <container>

# Start/stop container
docker start/stop <container>

# Enter container shell
docker exec -it <container> bash
```

---

Need help? Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for full details.
