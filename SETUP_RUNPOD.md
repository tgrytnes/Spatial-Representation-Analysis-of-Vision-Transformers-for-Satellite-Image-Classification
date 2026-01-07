# RunPod / GPU Instance Setup Guide

This guide explains how to set up a fresh Ubuntu GPU instance (like RunPod, Lambda Labs, or Paperspace) to run the EuroSAT benchmarks.

## 1. System Setup (Python 3.11)

RunPod usually comes with Python 3.10. We need Python 3.11. Run these commands block by block.

```bash
# Update package lists and install PPA tools
apt-get update && apt-get install -y software-properties-common git curl nano

# Add Deadsnakes PPA for newer Python versions (Press ENTER if prompted)
add-apt-repository -y ppa:deadsnakes/ppa

# Install Python 3.11 and dev tools
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev

# Set Python 3.11 as default (optional but helpful)
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
# Choose 2 (Python 3.11) if prompted, or it might auto-select.
```

## 2. Configure Git

Before cloning or committing, set up your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 3. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH (RunPod typically uses bash)
export PATH="/root/.local/bin:$PATH"
```

## 3. Clone Repository

```bash
git clone https://github.com/tgrytnes/Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification.git
cd Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification
```

## 4. Install Dependencies

```bash
# Configure poetry to use our new Python 3.11
poetry env use python3.11

# Install project dependencies
poetry install
```

## 5. Configure Secrets & APIs

You have two options: create a `.env` file or export variables manually.

### Option A: Create .env file (Recommended)
Create a file named `.env` in the project root:

```bash
nano .env
```

Paste the following content (replace placeholders with your real keys):

```ini
# Weights & Biases API Key (Found at https://wandb.ai/authorize)
WANDB_API_KEY=wandb_v1_YourLongKeyHere...

# Azure Storage Connection String (Only needed if using 'dvc pull')
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
```

Save and exit (`Ctrl+X`, `Y`, `Enter`).

Then, export them to your shell:
```bash
export $(grep -v '^#' .env | xargs)
```

### Option B: Interactive Login (W&B Only)
If you only need W&B and want to type it in:
```bash
poetry run wandb login
```

## 6. Get Data

### Path A: Fast Download (Recommended)
Download the raw dataset directly from the source. No Azure keys required.
```bash
poetry run python download.py
```

### Path B: DVC Pull (Requires Azure Key)
If you set up the Azure key in step 5:
```bash
poetry run dvc pull
```

## 7. Run Benchmarks

Run the full suite (ResNet, Swin, ViT, LoRA) sequentially:

```bash
for cfg in configs/benchmarks/*.yaml; do
  echo ">>> Starting benchmark for: $cfg"
  poetry run python eurosat_vit_analysis/experiment.py --config "$cfg"
done
```

## 8. View Results
Go to your W&B Dashboard to see live charts and comparisons.
