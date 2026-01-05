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

- [x] **Story 1.2: Data Version Control (DVC)**
  - **As a** Data Engineer, I want to version control the EuroSAT dataset using DVC.
  - **Acceptance Criteria:**
    - `.dvc` files present in repo; data excluded from git.
    - `dvc pull` successfully retrieves data from Azure Blob remote.
    - `dvc push`/`pull` validated for main dataset and run cache.

- [x] **Story 1.2b: Configure Azure Blob Remote for DVC**
  - **As a** Data Engineer, I want the EuroSAT dataset stored in Azure Blob so that any collaborator can fetch it.
  - **Acceptance Criteria:**
    - Remote `azure://spatialvit-eurosat/eurosat` configured with a local connection string override.
    - `dvc push --remote azure-remote` uploads the dataset; `dvc pull --remote azure-remote` rehydrates on a clean clone.
    - Documentation captures credential rotation and CI secret usage.

- [x] **Story 1.3: CI/CD Pipeline**
  - **As a** Maintainer, I want a GitHub Actions workflow that runs on every Pull Request.
  - **Acceptance Criteria:**
    - Pipeline runs `ruff`/`black` plus `pytest` with >= 80% coverage.
    - Environment installs dev tools so lint commands succeed.

- [x] **Story 1.4: Reproducible Experiments**
  - **As a** Research Engineer, I want deterministic, config-driven runs.
  - **Acceptance Criteria:**
    - All experiments launched via a single config (Hydra/Argparse).
    - Seeds fixed/logged; metrics match within ±1% across two runs.
    - Each run emits a manifest (git SHA, dataset version, params, metrics).

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
    - Dashboard visualizes Loss, Accuracy, and Macro-F1 live.
    - Best model checkpoints and config files are uploaded to W&B.

- [ ] **Story 2.3: Baselines & Ablations**
  - **As a** Researcher, I want a rigorous baseline and ablation suite.
  - **Acceptance Criteria:**
    - Benchmarks include ResNet and ConvNeXt baselines alongside ViT/Swin.
    - Report includes Macro-F1, per-class F1, and 95% confidence intervals.
    - Ablations cover patch size, augmentation strength, and freezing strategy.

- [ ] **Story 2.4: Parameter-Efficient Fine-Tuning**
  - **As a** Researcher, I want to compare full fine-tuning vs. LoRA/adapters.
  - **Acceptance Criteria:**
    - LoRA/adapters reduce trainable parameters by >80%.
    - Report compares accuracy and training time against full fine-tune.

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
    - Report compares accuracy drop of ViT vs. ResNet at epsilon {2/255, 4/255}.

- [ ] **Story 3.3: Spatial Probing & Invariance Tests**
  - **As a** Researcher, I want to quantify spatial reasoning and invariance.
  - **Acceptance Criteria:**
    - Patch-shuffle and occlusion tests are implemented and logged.
    - Accuracy drop under patch-shuffle is reported per class.
    - Occlusion sensitivity maps are saved for a fixed evaluation set.

- [ ] **Story 3.4: Calibration & Uncertainty**
  - **As a** Researcher, I want calibrated probabilities and uncertainty metrics.
  - **Acceptance Criteria:**
    - Expected Calibration Error (ECE) is computed before/after temperature scaling.
    - Reliability diagrams are generated for ViT and ResNet.

## Epic 4: The Demo Application (The Showcase)
*Goal: A tangible, interactive artifact for demonstration.*

- [ ] **Story 4.1: Streamlit Dashboard**
  - **As a** End User, I want a web interface to upload satellite images and view predictions.
  - **Acceptance Criteria:**
    - User can toggle between ViT and CNN backends.
    - Displays prediction class, confidence score, and Attention Heatmap.
    - Runs locally via `streamlit run app.py`.

- [ ] **Story 4.2: Hosted Demo App**
  - **As a** Recruiter, I want a public demo link I can try in under a minute.
  - **Acceptance Criteria:**
    - Frontend is deployed on Vercel with a FastAPI backend.
    - Custom domain `spatial-vit.fey-grytnes.com` is configured with HTTPS.
    - Includes sample gallery and inference cache for <2s response.
    - Public URL is added to README with screenshots.

- [ ] **Story 4.3: Portfolio Landing Page**
  - **As a** Hiring Manager, I want a concise page that explains the project impact.
  - **Acceptance Criteria:**
    - Page includes problem statement, architecture diagram, and key results.
    - Links to paper-style report, code, and demo.
    - Mobile-friendly and loads in <2s on 4G.

## Epic 5: Stretch Goals (Bleeding Edge)
*Goal: Explore emerging research trends (2024/2025).*

- [ ] **Story 5.1: Self-Supervised Learning (MAE)**
  - **As a** Researcher, I want to experiment with Masked Autoencoders (MAE) pre-training.
  - **Acceptance Criteria:** Compare MAE pre-training vs. ImageNet supervised pre-training on EuroSAT.

- [ ] **Story 5.2: Foundation Models**
  - **As a** Researcher, I want to fine-tune a Remote Sensing Foundation Model (e.g., IBM/NASA Prithvi).
  - **Acceptance Criteria:** Benchmark a generic ViT against a domain-specific Foundation Model.
