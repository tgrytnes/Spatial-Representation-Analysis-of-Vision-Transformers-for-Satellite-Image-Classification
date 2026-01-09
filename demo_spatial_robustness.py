"""
Demo script showing how to use spatial robustness functionality for Epic 3.3.

This demonstrates:
1. Patch-shuffle testing with logging
2. Per-class accuracy drop reporting
3. Saving occlusion sensitivity maps
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.spatial_robustness import (
    evaluate_patch_shuffle,
    occlusion_sensitivity,
    shuffle_patches,
)


def demo_patch_shuffle_visualization():
    """Demo 1: Visualize what patch shuffling does to an image."""
    print("=" * 70)
    print("DEMO 1: Patch Shuffle Visualization")
    print("=" * 70)

    # Create a simple pattern image to see the shuffling effect
    image = torch.zeros(1, 3, 64, 64)
    # Create a checkerboard pattern for visualization
    for i in range(0, 64, 16):
        for j in range(0, 64, 16):
            if (i // 16 + j // 16) % 2 == 0:
                image[:, :, i : i + 16, j : j + 16] = 1.0

    print(f"\nOriginal image shape: {image.shape}")

    # Shuffle with different grid sizes
    grid_sizes = [(2, 2), (4, 4), (8, 8)]

    fig, axes = plt.subplots(1, len(grid_sizes) + 1, figsize=(15, 4))
    axes[0].imshow(image[0, 0].cpu().numpy(), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for idx, grid_size in enumerate(grid_sizes):
        shuffled = shuffle_patches(image, grid_size, seed=42)
        axes[idx + 1].imshow(shuffled[0, 0].cpu().numpy(), cmap="gray")
        axes[idx + 1].set_title(f"Grid {grid_size[0]}x{grid_size[1]}")
        axes[idx + 1].axis("off")
        print(f"  Shuffled with grid {grid_size}: shape {shuffled.shape}")

    plt.tight_layout()
    output_dir = Path("outputs/spatial_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "patch_shuffle_demo.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to {output_dir / 'patch_shuffle_demo.png'}")
    plt.close()


def demo_patch_shuffle_evaluation_with_logging():
    """Demo 2: Evaluate model with patch shuffle and log results."""
    print("\n" + "=" * 70)
    print("DEMO 2: Patch-Shuffle Evaluation with Logging")
    print("=" * 70)

    # Create a simple model for demonstration
    print("\n1. Loading model...")
    model = create_model("vit_base", num_classes=10, pretrained=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Model: ViT-Base, Device: {device}")

    # Create synthetic evaluation dataset
    print("\n2. Creating synthetic evaluation dataset...")
    num_samples = 50
    images = torch.randn(num_samples, 3, 224, 224)
    targets = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    print(f"   Dataset: {num_samples} samples, 10 classes")

    # Evaluate with patch shuffle
    print("\n3. Evaluating with patch shuffle...")
    grid_size = (4, 4)  # Divide 224x224 into 4x4 grid of 56x56 patches
    report = evaluate_patch_shuffle(
        model=model,
        loader=loader,
        device=device,
        grid_size=grid_size,
        num_classes=10,
        seed=42,
    )

    # Log results - Acceptance Criterion 1: "Patch-shuffle tests are logged"
    print("\n" + "=" * 70)
    print("PATCH-SHUFFLE EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nGrid Size: {grid_size}")
    print("\nOverall Metrics:")
    print(f"  Clean Accuracy:     {report.clean_accuracy:.2%}")
    print(f"  Shuffled Accuracy:  {report.shuffled_accuracy:.2%}")
    print(f"  Accuracy Drop:      {report.accuracy_drop:.2%}")

    # Acceptance Criterion 2: "Accuracy drop per class is reported"
    print("\nPer-Class Accuracy Drop:")
    print(f"  {'Class':<8} {'Clean':<10} {'Shuffled':<10} {'Drop':<10}")
    print(f"  {'-'*40}")
    for class_id in range(10):
        clean_acc = report.per_class_clean_accuracy[class_id]
        shuffled_acc = report.per_class_shuffled_accuracy[class_id]
        drop = report.per_class_accuracy_drop[class_id]
        print(f"  {class_id:<8} {clean_acc:<10.2%} {shuffled_acc:<10.2%} {drop:<10.2%}")

    # Save logged results to file
    output_dir = Path("outputs/spatial_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "patch_shuffle_report.txt"
    with open(log_file, "w") as f:
        f.write("PATCH-SHUFFLE EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Grid Size: {grid_size}\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"  Clean Accuracy:     {report.clean_accuracy:.2%}\n")
        f.write(f"  Shuffled Accuracy:  {report.shuffled_accuracy:.2%}\n")
        f.write(f"  Accuracy Drop:      {report.accuracy_drop:.2%}\n\n")
        f.write("Per-Class Accuracy Drop:\n")
        f.write(f"  {'Class':<8} {'Clean':<10} {'Shuffled':<10} {'Drop':<10}\n")
        f.write(f"  {'-'*40}\n")
        for class_id in range(10):
            clean_acc = report.per_class_clean_accuracy[class_id]
            shuffled_acc = report.per_class_shuffled_accuracy[class_id]
            drop = report.per_class_accuracy_drop[class_id]
            f.write(
                f"  {class_id:<8} {clean_acc:<10.2%} "
                f"{shuffled_acc:<10.2%} {drop:<10.2%}\n"
            )

    print(f"\n✓ Results logged to {log_file}")

    # Visualize per-class accuracy drop
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(range(10))
    drops = [report.per_class_accuracy_drop[i] for i in classes]

    bars = ax.bar(classes, drops, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Class ID", fontsize=12)
    ax.set_ylabel("Accuracy Drop", fontsize=12)
    ax.set_title(
        "Per-Class Accuracy Drop under Patch Shuffle", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(classes)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{drop:.1%}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "per_class_accuracy_drop.png", dpi=150, bbox_inches="tight"
    )
    viz_path = output_dir / "per_class_accuracy_drop.png"
    print(f"✓ Per-class visualization saved to {viz_path}")
    plt.close()


def demo_occlusion_sensitivity_maps():
    """Demo 3: Generate and save occlusion sensitivity maps."""
    print("\n" + "=" * 70)
    print("DEMO 3: Occlusion Sensitivity Maps")
    print("=" * 70)

    # Create a simple model
    print("\n1. Loading model...")
    model = create_model("vit_base", num_classes=10, pretrained=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Model: ViT-Base, Device: {device}")

    # Create a fixed evaluation set
    print("\n2. Creating fixed evaluation set...")
    num_eval_samples = 5
    eval_images = torch.randn(num_eval_samples, 3, 224, 224).to(device)

    # Get predictions for these images
    with torch.no_grad():
        outputs = model(eval_images)
        predicted_classes = outputs.argmax(dim=1)

    print(f"   Evaluation set: {num_eval_samples} images")
    print(f"   Predicted classes: {predicted_classes.cpu().tolist()}")

    # Generate occlusion sensitivity maps
    print("\n3. Generating occlusion sensitivity maps...")
    occlusion_size = 32  # Size of occluding square
    stride = 16  # Stride for sliding window

    sensitivity_maps = occlusion_sensitivity(
        model=model,
        images=eval_images,
        targets=predicted_classes,
        device=device,
        occlusion_size=occlusion_size,
        stride=stride,
    )

    print(f"   Occlusion size: {occlusion_size}x{occlusion_size}")
    print(f"   Stride: {stride}")
    print(f"   Sensitivity maps shape: {sensitivity_maps.shape}")

    # Acceptance Criterion 3: "Occlusion sensitivity maps are saved for evaluation set"
    output_dir = Path("outputs/spatial_robustness/occlusion_maps")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n4. Saving sensitivity maps...")

    # Save as PyTorch tensor for later analysis
    torch.save(
        {
            "sensitivity_maps": sensitivity_maps.cpu(),
            "images": eval_images.cpu(),
            "targets": predicted_classes.cpu(),
            "occlusion_size": occlusion_size,
            "stride": stride,
        },
        output_dir / "sensitivity_maps.pt",
    )
    print(f"   ✓ Saved tensor to {output_dir / 'sensitivity_maps.pt'}")

    # Visualize and save as images
    fig, axes = plt.subplots(num_eval_samples, 3, figsize=(12, 4 * num_eval_samples))
    if num_eval_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_eval_samples):
        # Original image
        img = eval_images[i].cpu().permute(1, 2, 0).numpy()
        # Normalize to [0, 1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i} (Class {predicted_classes[i].item()})")
        axes[i, 0].axis("off")

        # Sensitivity map (heatmap)
        sensitivity = sensitivity_maps[i].cpu().numpy()
        im = axes[i, 1].imshow(sensitivity, cmap="hot", interpolation="bilinear")
        axes[i, 1].set_title(f"Sensitivity Map {i}")
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Overlay on original image
        from scipy.ndimage import zoom

        # Resize sensitivity map to match image size
        scale_h = 224 / sensitivity.shape[0]
        scale_w = 224 / sensitivity.shape[1]
        sensitivity_resized = zoom(sensitivity, (scale_h, scale_w), order=1)

        axes[i, 2].imshow(img)
        axes[i, 2].imshow(
            sensitivity_resized, cmap="hot", alpha=0.5, interpolation="bilinear"
        )
        axes[i, 2].set_title(f"Overlay {i}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    viz_path = output_dir / "sensitivity_maps_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Saved visualization to {viz_path}")
    plt.close()

    # Save individual maps as numpy arrays
    for i in range(num_eval_samples):
        map_path = (
            output_dir
            / f"sensitivity_map_sample_{i}_class_{predicted_classes[i].item()}.npy"
        )
        np.save(map_path, sensitivity_maps[i].cpu().numpy())
    print(f"   ✓ Saved {num_eval_samples} individual maps as .npy files")

    print("\n" + "=" * 70)
    print("OCCLUSION SENSITIVITY SUMMARY")
    print("=" * 70)
    print(f"Total samples processed: {num_eval_samples}")
    print(f"Sensitivity maps saved: {num_eval_samples}")
    print(f"Output directory: {output_dir}")
    print("\nInterpretation:")
    print("  - Bright regions (high values) = occluding this area drops confidence")
    print("  - Dark regions (low values) = occluding this area has little effect")
    print("  - These maps reveal which spatial regions the model relies on")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("SPATIAL ROBUSTNESS DEMO - Epic 3.3")
    print("=" * 70)
    print("\nThis demo shows how to use the spatial robustness functionality:")
    print("1. Patch-shuffle visualization")
    print("2. Patch-shuffle evaluation with per-class logging")
    print("3. Occlusion sensitivity maps for evaluation set")
    print()

    # Run demonstrations
    demo_patch_shuffle_visualization()
    demo_patch_shuffle_evaluation_with_logging()
    demo_occlusion_sensitivity_maps()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nAcceptance Criteria Status:")
    print("✓ Patch-shuffle tests are implemented and logged")
    print("✓ Accuracy drop under patch-shuffle is reported per class")
    print("✓ Occlusion sensitivity maps are saved for a fixed evaluation set")
    print("\nCheck the outputs/spatial_robustness/ directory for results!")


if __name__ == "__main__":
    main()
