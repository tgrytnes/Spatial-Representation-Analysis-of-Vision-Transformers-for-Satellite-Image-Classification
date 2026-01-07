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

## 2. Configure Secrets (First Step)

To automate everything (Git, W&B, Azure), prepare a `.env` file on your **local machine** first.

### Local Preparation
Use the provided template to create your local secrets file:

```bash
# 1. Copy the template
cp .env.example .env

# 2. Fill in your real keys (W&B, GitHub, Azure)
nano .env
```

### Transfer to Pod
Transfer this file to the pod using `scp` (replace `[POD_IP]` and `[PORT]` with your specific RunPod details):

```bash
scp -P [PORT] .env root@[POD_IP]:/root/
```

### Apply Secrets
On the pod, export the variables:

```bash
export $(grep -v '^#' /root/.env | xargs)
```

## 3. Configure Git (Automated)

Once secrets are exported, configure Git in one line:

```bash
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_USERNAME@users.noreply.github.com"
git config --global credential.helper store

# Setup automatic authentication for cloning/pushing using the token
echo "https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com" > ~/.git-credentials
```

## 4. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH (RunPod typically uses bash)
export PATH="/root/.local/bin:$PATH"
```

## 5. Clone Repository

```bash
git clone https://github.com/tgrytnes/Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification.git
cd Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification
```

## 6. Install Dependencies

```bash
# Configure poetry to use our new Python 3.11
poetry env use python3.11

# Install project dependencies
poetry install
```

## 7. Login Services

If you didn't set variables in step 2, you can login manually:

```bash
poetry run wandb login
```

## 8. Get Data

###  DVC Pull (Requires Azure Key in .env)
```bash
poetry run dvc pull
```

## 9. Run Benchmarks 

Run the full suite (ResNet, Swin, ViT, LoRA) sequentially:

```bash
for cfg in configs/benchmarks/*.yaml; do
  echo ">>> Starting benchmark for: $cfg"
  poetry run python eurosat_vit_analysis/experiment.py --config "$cfg"
done
```

## 10. View Results
Go to your W&B Dashboard to see live charts and comparisons.
