# Spatial Representation Analysis of Vision Transformers

This repository contains the code and experiments for the project "Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification". The primary goal is to analyze how Vision Transformers (ViTs) build spatial understanding of satellite imagery compared to traditional CNNs, using the EuroSAT dataset.

## 📋 Table of Contents
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Data Setup](#data-setup)
  - [Running Experiments](#running-experiments)
  - [Running the Demo](#running-the-demo)
- [Development](#-development)
  - [Linting and Formatting](#linting-and-formatting)
  - [Running Tests](#running-tests)
- [Project Structure](#-project-structure)

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

### Running Experiments
Experiments are managed via a training script that accepts arguments using Hydra. To launch a training run:
```bash
poetry run python eurosat_vit_analysis/train.py model=swin_t data=eurosat
```
*(Note: This is a placeholder command; the exact arguments will be defined by the Hydra configuration.)*

Metrics and model artifacts are tracked using Weights & Biases.

### Running the Demo
A Streamlit dashboard is available to interactively test models.
```bash
poetry run streamlit run eurosat_vit_analysis/app.py
```

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
