# Project User Stories & Roadmap

This document outlines the development roadmap for the EuroSAT Vision Transformer Analysis project. It is organized by Epics representing key engineering and research phases.

## Epic 1: MLOps & Infrastructure (The Foundation)
*Goal: Establish a reproducible, robust, and automated research environment.*

- [x] **Story 1.1: Project Initialization**
  - **As a** DevOps Engineer, I want to set up the project using `Poetry` and `pre-commit` hooks.
  - **Acceptance Criteria:**
    - `pyproject.toml` configured for dependency management.
    - `pre-commit` hooks installed (Black, Flake8/Ruff, Isort).
    - `Makefile` created for common tasks (`make test`, `make lint`).

- [ ] **Story 1.2: Data Version Control (DVC)**
  - **As a** Data Engineer, I want to version control the EuroSAT dataset using DVC.
  - **Acceptance Criteria:**
    - `.dvc` files present in repo; data excluded from git.
    - `dvc pull` successfully retrieves data from remote storage.
    - Data loader tests pass after a fresh pull.

- [ ] **Story 1.3: CI/CD Pipeline**
  - **As a** Maintainer, I want a GitHub Actions workflow that runs on every Pull Request.
  - **Acceptance Criteria:**
    - Pipeline runs `pytest` and linter.
    - Pipeline fails if code coverage is < 80%.

## Epic 2: Advanced Modeling (The Core)
*Goal: Implement and train state-of-the-art Vision Transformers.*

- [ ] **Story 2.1: Model Factory (Swin & ViT)**
  - **As a** Researcher, I want a factory pattern to instantiate Swin-Tiny and ViT-Base models using `timm`.
  - **Acceptance Criteria:**
    - `create_model('swin_t')` returns correct architecture with 10 output classes.
    - Unit tests verify input/output shapes `(B, 10)`.
    - Support for freezing backbone layers.

- [ ] **Story 2.2: Experiment Tracking (W&B)**
  - **As a** Researcher, I want to log metrics and artifacts to Weights & Biases.
  - **Acceptance Criteria:**
    - Training script accepts CLI arguments (Hydra/Argparse).
    - Dashboard visualizes Loss, Accuracy, and F1-score live.
    - Best model checkpoints are uploaded to W&B.

## Epic 3: Explainability & Robustness (The Insight)
*Goal: Analyze spatial reasoning and model fragility.*

- [ ] **Story 3.1: Attention Rollout Visualization**
  - **As a** Data Scientist, I want to extract and visualize Attention Maps from ViT layers.
  - **Acceptance Criteria:**
    - Function returns normalized heatmaps overlaid on original images.
    - Unit tests verify heatmap dimensions match input image size.

- [ ] **Story 3.2: Adversarial Robustness (FGSM)**
  - **As a** Safety Engineer, I want to evaluate the model against gradient-based attacks.
  - **Acceptance Criteria:**
    - Script generates adversarial examples using FGSM.
    - Report generated comparing accuracy drop of ViT vs. ResNet.

## Epic 4: The Demo Application (The Showcase)
*Goal: A tangible, interactive artifact for demonstration.*

- [ ] **Story 4.1: Streamlit Dashboard**
  - **As a** End User, I want a web interface to upload satellite images and view predictions.
  - **Acceptance Criteria:**
    - User can toggle between ViT and CNN backends.
    - Displays prediction class, confidence score, and Attention Heatmap.
    - Runs locally via `streamlit run app.py`.

## Epic 5: Stretch Goals (Bleeding Edge)
*Goal: Explore emerging research trends (2024/2025).*

- [ ] **Story 5.1: Self-Supervised Learning (MAE)**
  - **As a** Researcher, I want to experiment with Masked Autoencoders (MAE) pre-training.
  - **Acceptance Criteria:** Compare MAE pre-training vs. ImageNet supervised pre-training on EuroSAT.

- [ ] **Story 5.2: Foundation Models**
  - **As a** Researcher, I want to fine-tune a Remote Sensing Foundation Model (e.g., IBM/NASA Prithvi).
  - **Acceptance Criteria:** Benchmark a generic ViT against a domain-specific Foundation Model.
