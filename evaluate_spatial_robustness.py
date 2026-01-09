#!/usr/bin/env python
"""
Evaluate spatial robustness of trained models on EuroSAT.

This script analyzes how ViT and ResNet models respond to:
1. Patch shuffling (spatial structure disruption)
2. Occlusion (identifying important spatial regions)

Usage:
    python evaluate_spatial_robustness.py --model vit_base \\
        --checkpoint checkpoints/vit_base_best.pt
    python evaluate_spatial_robustness.py --all
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom

from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.spatial_robustness import (
    evaluate_patch_shuffle,
    occlusion_sensitivity,
)


def evaluate_model_spatial_robustness(
    model_name: str,
    checkpoint_path: str,
    data_dir: str = "data/EuroSAT",
    output_dir: str = "outputs/spatial_robustness_evaluation",
    batch_size: int = 32,
    grid_sizes: list[tuple[int, int]] = None,
    occlusion_config: dict = None,
):
    """
    Evaluate spatial robustness of a trained model.

    Args:
        model_name: Model architecture ('vit_base', 'resnet50', etc.)
        checkpoint_path: Path to trained checkpoint
        data_dir: Path to EuroSAT dataset
        output_dir: Where to save results
        batch_size: Batch size for evaluation
        grid_sizes: List of grid sizes for patch shuffle
        occlusion_config: Config for occlusion sensitivity
    """
    print("=" * 80)
    print(f"EVALUATING SPATIAL ROBUSTNESS: {model_name}")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n1. Loading model from {checkpoint_path}...")
    model = create_model(model_name, num_classes=10, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")

    # Load data
    print(f"\n2. Loading EuroSAT data from {data_dir}...")
    _, val_loader, class_names = prepare_data(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        val_split=0.2,
        seed=42,
        augmentation="none",
    )
    print(f"✓ Validation set: {len(val_loader.dataset)} samples")
    print(f"✓ Classes: {', '.join(class_names)}")

    # Default configurations
    if grid_sizes is None:
        grid_sizes = [(2, 2), (4, 4), (8, 8), (14, 14)]

    if occlusion_config is None:
        occlusion_config = {
            "occlusion_size": 32,
            "stride": 16,
            "num_samples": 50,  # Number of samples for occlusion maps
        }

    # ========================================================================
    # PART 1: Patch-Shuffle Evaluation
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: PATCH-SHUFFLE ROBUSTNESS")
    print("=" * 80)

    patch_shuffle_results = {}

    for grid_size in grid_sizes:
        patch_size = 224 // grid_size[0]
        print(f"\nEvaluating grid {grid_size} ({patch_size}x{patch_size} patches)...")

        report = evaluate_patch_shuffle(
            model=model,
            loader=val_loader,
            device=device,
            grid_size=grid_size,
            num_classes=10,
            seed=42,
        )

        patch_shuffle_results[grid_size] = report

        print(f"  Clean Accuracy:     {report.clean_accuracy:.2%}")
        print(f"  Shuffled Accuracy:  {report.shuffled_accuracy:.2%}")
        print(f"  Accuracy Drop:      {report.accuracy_drop:+.2%}")

    # Save detailed results for best grid size (4x4)
    best_grid = (4, 4)
    best_report = patch_shuffle_results[best_grid]

    log_path = output_path / "patch_shuffle_report.txt"
    with open(log_path, "w") as f:
        f.write("PATCH-SHUFFLE ROBUSTNESS EVALUATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {data_dir}\n\n")

        # Summary across grid sizes
        f.write("Summary Across Grid Sizes:\n")
        f.write(
            f"  {'Grid':<10} {'Patch Size':<12} "
            f"{'Clean Acc':<12} {'Shuffled Acc':<15} {'Drop':<10}\n"
        )
        f.write(f"  {'-'*65}\n")
        for grid_size in grid_sizes:
            patch_size = 224 // grid_size[0]
            r = patch_shuffle_results[grid_size]
            f.write(
                f"  {str(grid_size):<10} {patch_size}x{patch_size:<10} "
                f"{r.clean_accuracy:<12.2%} {r.shuffled_accuracy:<15.2%} "
                f"{r.accuracy_drop:+10.2%}\n"
            )

        # Detailed per-class results for 4x4 grid
        f.write(f"\n\nDetailed Results for {best_grid} Grid:\n")
        f.write(f"  Overall Accuracy Drop: {best_report.accuracy_drop:+.2%}\n\n")
        f.write("  Per-Class Accuracy Drop:\n")
        f.write(
            f"    {'Class Name':<20} "
            f"{'Clean Acc':<12} {'Shuffled Acc':<15} {'Drop':<10}\n"
        )
        f.write(f"    {'-'*60}\n")
        for i, name in enumerate(class_names):
            clean = best_report.per_class_clean_accuracy[i]
            shuffled = best_report.per_class_shuffled_accuracy[i]
            drop = best_report.per_class_accuracy_drop[i]
            f.write(f"    {name:<20} {clean:<12.2%} {shuffled:<15.2%} {drop:+10.2%}\n")

    print(f"\n✓ Results saved to {log_path}")

    # Visualize results
    visualize_patch_shuffle_results(
        model_name, class_names, patch_shuffle_results, best_report, output_path
    )

    # ========================================================================
    # PART 2: Occlusion Sensitivity Maps
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 2: OCCLUSION SENSITIVITY MAPS")
    print("=" * 80)

    # Get fixed evaluation set (first N samples from validation)
    print(
        f"\nGenerating occlusion maps for {occlusion_config['num_samples']} samples..."
    )

    images, targets = [], []
    for batch_images, batch_targets in val_loader:
        images.append(batch_images)
        targets.append(batch_targets)
        if sum(len(b) for b in images) >= occlusion_config["num_samples"]:
            break

    images = torch.cat(images)[: occlusion_config["num_samples"]].to(device)
    targets = torch.cat(targets)[: occlusion_config["num_samples"]].to(device)

    print(f"  Evaluation set: {len(images)} images")
    occ_size = occlusion_config["occlusion_size"]
    print(f"  Occlusion size: {occ_size}x{occ_size}")
    print(f"  Stride: {occlusion_config['stride']}")

    sensitivity_maps = occlusion_sensitivity(
        model=model,
        images=images,
        targets=targets,
        device=device,
        occlusion_size=occlusion_config["occlusion_size"],
        stride=occlusion_config["stride"],
    )

    print(f"  Generated sensitivity maps: {sensitivity_maps.shape}")

    # Save maps
    maps_dir = output_path / "occlusion_maps"
    maps_dir.mkdir(exist_ok=True)

    torch.save(
        {
            "sensitivity_maps": sensitivity_maps.cpu(),
            "images": images.cpu(),
            "targets": targets.cpu(),
            "class_names": class_names,
            "model": model_name,
            "config": occlusion_config,
        },
        maps_dir / "sensitivity_maps.pt",
    )

    # Save individual maps
    for i in range(len(images)):
        class_id = targets[i].item()
        class_name = class_names[class_id]
        map_path = maps_dir / f"map_{i:03d}_class_{class_id}_{class_name}.npy"
        np.save(map_path, sensitivity_maps[i].cpu().numpy())

    print(f"\n✓ Saved {len(images)} sensitivity maps to {maps_dir}")

    # Visualize sample maps
    visualize_occlusion_maps(
        images, targets, sensitivity_maps, class_names, maps_dir, num_samples=10
    )

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {model_name}")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print("\nSummary:")
    print(f"  - Patch-shuffle report: {log_path}")
    print(f"  - Visualizations: {output_path}/*.png")
    print(f"  - Occlusion maps: {maps_dir}/")


def visualize_patch_shuffle_results(
    model_name, class_names, all_results, best_report, output_path
):
    """Create visualizations for patch-shuffle results."""

    # 1. Per-class accuracy drop bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    classes = list(range(10))
    drops = [best_report.per_class_accuracy_drop[i] for i in classes]

    colors = ["red" if d > 0 else "green" for d in drops]
    bars = ax.bar(classes, drops, color=colors, alpha=0.7, edgecolor="black")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Accuracy Drop", fontsize=12)
    ax.set_title(
        f"{model_name}: Per-Class Accuracy Drop under Patch Shuffle (4x4 grid)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(classes)
    ax.set_xticklabels([name[:15] for name in class_names], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{drop:+.1%}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        output_path / "per_class_accuracy_drop.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # 2. Grid size comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    grid_labels = [f"{g[0]}x{g[1]}" for g in all_results.keys()]
    drops = [r.accuracy_drop for r in all_results.values()]

    ax.plot(grid_labels, drops, marker="o", linewidth=2, markersize=8)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Grid Size", fontsize=12)
    ax.set_ylabel("Accuracy Drop", fontsize=12)
    ax.set_title(
        f"{model_name}: Accuracy Drop vs. Grid Size",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add annotations
    for i, (label, drop) in enumerate(zip(grid_labels, drops)):
        ax.annotate(
            f"{drop:+.1%}",
            (i, drop),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path / "grid_size_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Visualizations saved to {output_path}")


def visualize_occlusion_maps(
    images, targets, sensitivity_maps, class_names, output_dir, num_samples=10
):
    """Visualize sample occlusion sensitivity maps."""

    num_samples = min(num_samples, len(images))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        class_id = targets[i].item()
        axes[i, 0].set_title(f"Sample {i}: {class_names[class_id]}")
        axes[i, 0].axis("off")

        # Sensitivity map
        sensitivity = sensitivity_maps[i].cpu().numpy()
        im = axes[i, 1].imshow(sensitivity, cmap="hot", interpolation="bilinear")
        axes[i, 1].set_title("Sensitivity Map")
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Overlay
        scale_h = 224 / sensitivity.shape[0]
        scale_w = 224 / sensitivity.shape[1]
        sensitivity_resized = zoom(sensitivity, (scale_h, scale_w), order=1)

        axes[i, 2].imshow(img)
        axes[i, 2].imshow(
            sensitivity_resized, cmap="hot", alpha=0.5, interpolation="bilinear"
        )
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        output_dir / "sample_sensitivity_maps.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Sample visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spatial robustness of trained models on EuroSAT"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vit_base", "resnet50", "swin_t", "convnext_t"],
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/EuroSAT",
        help="Path to EuroSAT dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/spatial_robustness_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all available models",
    )

    args = parser.parse_args()

    if args.all:
        # Evaluate all models
        models = [
            ("vit_base", "checkpoints/vit_base_best.pt"),
            ("resnet50", "checkpoints/resnet50_best.pt"),
        ]

        for model_name, checkpoint_path in models:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.exists():
                print(
                    f"⚠ Skipping {model_name}: "
                    f"checkpoint not found at {checkpoint_path}"
                )
                continue

            evaluate_model_spatial_robustness(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
            )
            print("\n")

    elif args.model and args.checkpoint:
        # Evaluate single model
        evaluate_model_spatial_robustness(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

    else:
        parser.print_help()
        print("\nError: Must specify either --all or both --model and --checkpoint")


if __name__ == "__main__":
    main()
