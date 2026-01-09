---
title: Spatial ViT Analysis
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.52.2
app_file: app.py
pinned: false
hardware: t4-small
---

# Spatial ViT Analysis Dashboard

Interactive web interface for comparing Vision Transformer and CNN models on satellite image classification.

## Features

- 🛰️ Upload satellite images for classification
- 🔄 Compare ViT-Base vs ResNet-50 predictions side-by-side
- 📊 Interactive visualizations with confidence scores
- ⚡ GPU-accelerated inference (T4)
- 📈 Model metrics (parameters, runtime, architecture)

## Models

- **ViT-Base**: Vision Transformer (86M params)
- **ResNet-50**: Residual CNN (25M params)
- **Swin-Tiny**: Swin Transformer (28M params)

## Dataset

EuroSAT - 10 land use/land cover classes:
- AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial
- Pasture, PermanentCrop, Residential, River, SeaLake

## Usage

1. Select single model or comparison mode
2. Upload a satellite image (or use samples)
3. View predictions with confidence scores
4. Compare model outputs and metrics

## Repository

[GitHub - Spatial ViT Analysis](https://github.com/tgrytnes/Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification)

Built with ❤️ for Vision Transformer spatial analysis research
