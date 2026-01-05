# Spatial Representation Analysis of Vision Transformers

This repository contains the code and experiments for "Spatial Representation Analysis of Vision Transformers for Satellite Image Classification". The goal is to quantify how Vision Transformers (ViTs) encode spatial structure compared to CNNs on EuroSAT, and to surface robust, interpretable insights.

## 🔎 Project Highlights
- **Spatial probing:** attention rollout, patch shuffling, and occlusion sensitivity.
- **Comparative modeling:** ViT/Swin vs CNN baselines with controlled ablations.
- **Robustness & calibration:** adversarial testing and uncertainty metrics.
- **Reproducibility:** config-driven runs, tracked artifacts, and dataset versioning.
- **Showcase-ready:** interactive demo + portfolio landing page (in progress).

## 📋 Table of Contents
- [Spatial Representation Analysis of Vision Transformers](#spatial-representation-analysis-of-vision-transformers)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔎 Project Highlights](#-project-highlights)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Data Setup](#data-setup)
    - [Running Experiments](#running-experiments)
    - [Running the Demo](#running-the-demo)
  - [🌐 Demo & Portfolio](#-demo--portfolio)
  - [🧾 Technical Report](#-technical-report)
  - [🧪 Methods + Evaluation](#-methods--evaluation)
  - [📈 Results (WIP)](#-results-wip)
  - [🔁 Reproducibility & Tracking](#-reproducibility--tracking)
  - [🚢 Deployment (Vercel + FastAPI)](#-deployment-vercel--fastapi)
  - [🛠️ Development](#️-development)
    - [Linting and Formatting](#linting-and-formatting)
    - [Running Tests](#running-tests)
  - [📂 Project Structure](#-project-structure)
  - [🧭 Roadmap](#-roadmap)

## 🚀 Getting Started

### Prerequisites
- [Poetry](https://python-poetry.org/) for dependency management.
- [Git](https://git-scm.com/) for version control.
- [DVC](https://dvc.org/) for data versioning.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification
    ```

2.  **Install dependencies and hooks:**
    This command will create a virtual environment using Poetry, install all dependencies from `pyproject.toml`, and set up the pre-commit hooks.
    ```bash
    make install
    ```

## Usage

### Data Setup
The EuroSAT dataset is versioned using DVC. To pull the data from the remote storage (configured to use S3):
```bash
dvc pull
```
See [docs/dvc-remote.md](docs/dvc-remote.md) for configuring the Azure Blob remote (account/container selection, connection strings, rotation).

### Running Experiments
Experiments are managed via a training script that accepts arguments using Hydra. To launch a training run:
```bash
poetry run python eurosat_vit_analysis/train.py model=swin_t data=eurosat
```
*(Note: This is a placeholder command; the exact arguments will be defined by the Hydra configuration.)*

Metrics and model artifacts are tracked using Weights & Biases.

### Deterministic Experiment Runner
Run deterministic experiments with a single config file. The small runner writes a manifest (git SHA, params, and metrics) so you can compare two runs and see consistent metrics:

```bash
poetry run python -m eurosat_vit_analysis.experiment --config configs/experiment.yaml
```

Manifests land in the `manifests/` directory and include the dataset version, config, and computed metrics.

### Running the Demo
A Streamlit dashboard is available to interactively test models.
```bash
poetry run streamlit run eurosat_vit_analysis/app.py
```

## 🌐 Demo & Portfolio
- **Hosted demo:** Vercel + FastAPI (planned).
- **Custom domain:** `https://spatial-vit.fey-grytnes.com` (planned).
- **Portfolio page:** Project overview, architecture, results, and links (planned).
- **Local demo:** See [Running the Demo](#running-the-demo).

## 🧾 Technical Report
- **Link:** `https://fey-grytnes.com/spatial-vit-report` (planned).
- **Scope:** Methods, evaluation protocol, key findings, and limitations.

## 🧪 Methods + Evaluation
**Dataset**
- EuroSAT with DVC-managed versioning, fixed train/val/test splits, and class balance checks.

**Models**
- ViT-Base and Swin-Tiny via `timm`, plus CNN baselines (ResNet/ConvNeXt).
- Full fine-tuning vs parameter-efficient methods (LoRA/adapters) for cost/perf tradeoffs.

**Spatial Probes**
- Attention rollout, patch-shuffle degradation, and occlusion sensitivity to measure spatial reliance.

**Robustness & Calibration**
- FGSM at multiple epsilon levels and Expected Calibration Error (ECE) with temperature scaling.

**Metrics**
- Accuracy, Macro-F1, per-class F1, and 95% confidence intervals.

**Reproducibility**
- Config-driven runs with fixed seeds; each run logs git SHA, dataset version, and metrics.

## 📈 Results (WIP)
Planned reporting includes:
- **Metrics:** Accuracy, Macro-F1, per-class F1, and confidence intervals.
- **Robustness:** FGSM performance at multiple epsilon levels.
- **Spatial probes:** Patch-shuffle accuracy deltas and occlusion sensitivity maps.

## 🔁 Reproducibility & Tracking
- **Configuration:** Experiments launched via a single config entry point (Hydra/Argparse).
- **Artifacts:** Metrics, checkpoints, and configs tracked in Weights & Biases.
- **Data:** EuroSAT is versioned with DVC.
- **Data Remote:** Azure Blob (`spatialvit-eurosat` container) holds the dataset; see `docs/dvc-remote.md`.

## 🚢 Deployment (Vercel + FastAPI)
**Target**
- Frontend on Vercel with FastAPI serverless functions for inference.
- Custom domain: `spatial-vit.fey-grytnes.com`.

**Why this stack**
- Simple to ship and maintain, with fast global delivery.
- Sufficient for lightweight inference; keeps the demo snappy for recruiters.

**Notes**
- Prefer small model artifacts to avoid cold-start delays.
- Add a basic inference cache to keep responses under ~2 seconds.

## 🛠️ Development

### Linting and Formatting
The project uses `ruff` for linting and `black` for formatting.

- **To check for issues:**
  ```bash
  make lint
  ```
- **To automatically fix issues:**
  ```bash
  make format
  ```

### Running Tests
Tests are written with `pytest`. The following command will run the test suite and report code coverage.
```bash
make test
```

## 📂 Project Structure
```
.
├── .dvc/                   # DVC metadata
├── .github/                # GitHub Actions workflows
├── data/                   # Raw and processed data (tracked by DVC)
│   └── eurosat/
├── eurosat_vit_analysis/   # Main source code package
│   ├── __init__.py
│   ├── app.py              # Streamlit demo application
│   ├── train.py            # Main training script
│   ├── models/             # Model factory and definitions
│   └── vis/                # Visualization code (e.g., attention maps)
├── notebooks/              # Jupyter notebooks for exploratory analysis
├── pyproject.toml          # Poetry configuration and dependencies
├── Makefile                # Makefile for common commands
└── README.md
```

## 🧭 Roadmap
See `user stories.md` for the full roadmap and epic-level plan.
