# EuroSAT Benchmark Comparison (20 Epochs)

## Summary
Across the latest 20-epoch runs, LoRA outperforms full fine-tuning for Swin-T, ViT-Base, and ConvNeXt-T, while ResNet50 performs better with full fine-tuning. Overall accuracy ranges from ~0.94 to ~0.997, with the strongest results from ViT-Base LoRA and Swin-T LoRA.

## Latest Runs (Full vs LoRA)
| Model | Variant | Accuracy | F1 (macro) | Loss | Run |
| --- | --- | --- | --- | --- | --- |
| resnet50 | full | 0.9852 | 0.9852 | 0.0533 | [stellar-disco-11](https://wandb.ai/thfe9252-university-of-colorado-boulder/eurosat-vit-analysis/runs/88loyvo9) |
| resnet50 | lora | 0.9420 | 0.9420 | 0.1761 | [misunderstood-cherry-16](https://wandb.ai/thfe9252-university-of-colorado-boulder/eurosat-vit-analysis/runs/xl6yqhfv) |
| swin_t | full | 0.9644 | 0.9644 | 0.1009 | [rural-sunset-12](https://wandb.ai/thfe9252-university-of-colorado-boulder/eurosat-vit-analysis/runs/yfo80dtn) |
| swin_t | lora | 0.9911 | 0.9911 | 0.0339 | [scarlet-donkey-13](https://wandb.ai/thfe9252-university-of-colorado-boulder/eurosat-vit-analysis/runs/rsnp73nd) |
| vit_base | full | 0.9906 | 0.9906 | 0.0334 | [swift-sea-14](https://wandb.ai/thfe9252-university-of-colorado-boulder/eurosat-vit-analysis/runs/3opeqm7h) |
| vit_base | lora | 0.9970 | 0.9970 | 0.0156 | [worldly-bird-15](https://wandb.ai/thfe9252-university-of-colorado-boulder/eurosat-vit-analysis/runs/ov8t58oj) |
| convnext_t | full | 0.9413 | 0.9413 | 0.1760 | [wandering-snow-17](https://wandb.ai/thfe9252-university-of-colorado-boulder/eurosat-vit-analysis/runs/jel7kof2) |
| convnext_t | lora | 0.9802 | 0.9802 | 0.0622 | [decent-field-19](https://wandb.ai/thfe9252-university-of-colorado-boulder/eurosat-vit-analysis/runs/9kim2q3j) |

## LoRA - Full Deltas
| Model | Accuracy Delta | F1 Delta | Loss Delta |
| --- | --- | --- | --- |
| resnet50 | -0.0431 | -0.0431 | +0.1228 |
| swin_t | +0.0267 | +0.0267 | -0.0670 |
| vit_base | +0.0065 | +0.0065 | -0.0178 |
| convnext_t | +0.0389 | +0.0389 | -0.1139 |

## Notes
- All runs are 20 epochs on the same dataset configuration (`v1.0`, augmentation `light`, batch size 64).
- Metrics come from W&B run summaries for the latest finished run per model and variant.
