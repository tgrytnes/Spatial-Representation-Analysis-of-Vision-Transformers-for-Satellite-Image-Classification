"""
Demo script showing how to use calibration functionality for Epic 3.

This demonstrates:
1. Computing ECE (Expected Calibration Error) before/after temperature scaling
2. Temperature scaling for probability calibration
3. Generating reliability diagrams for ViT and ResNet
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from eurosat_vit_analysis.calibration import (
    apply_temperature_scaling,
    evaluate_calibration,
    generate_reliability_diagram_data,
)
from eurosat_vit_analysis.models import create_model


def plot_reliability_diagram(
    metrics_before,
    metrics_after,
    model_name,
    output_path,
):
    """Plot reliability diagram comparing before/after calibration."""
    # Get data for plotting
    data_before = generate_reliability_diagram_data(metrics_before)
    data_after = generate_reliability_diagram_data(metrics_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot before calibration
    ax1.bar(
        data_before["bin_centers"],
        data_before["bin_accuracies"],
        width=1.0 / len(data_before["bin_centers"]),
        alpha=0.7,
        label="Accuracy",
        color="steelblue",
        edgecolor="black",
    )
    ax1.plot(
        data_before["bin_centers"],
        data_before["bin_confidences"],
        "r-o",
        linewidth=2,
        markersize=6,
        label="Confidence",
    )
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect Calibration")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(
        f"{model_name} - Before Calibration\nECE: {metrics_before.ece:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot after calibration
    ax2.bar(
        data_after["bin_centers"],
        data_after["bin_accuracies"],
        width=1.0 / len(data_after["bin_centers"]),
        alpha=0.7,
        label="Accuracy",
        color="steelblue",
        edgecolor="black",
    )
    ax2.plot(
        data_after["bin_centers"],
        data_after["bin_confidences"],
        "r-o",
        linewidth=2,
        markersize=6,
        label="Confidence",
    )
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect Calibration")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(
        f"{model_name} - After Calibration\nECE: {metrics_after.ece:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def demo_calibration_with_synthetic_data():
    """Demo: Calibration evaluation with synthetic data."""
    print("=" * 70)
    print("DEMO: Calibration Evaluation with Synthetic Data")
    print("=" * 70)

    # Create an overconfident model
    print("\n1. Creating overconfident model...")
    model = create_model("vit_base", num_classes=10, pretrained=False)
    model.eval()

    # Create synthetic dataset
    print("2. Creating synthetic evaluation dataset...")
    num_samples = 100
    images = torch.randn(num_samples, 3, 224, 224)
    targets = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    print(f"   Dataset: {num_samples} samples, 10 classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Device: {device}")

    # Evaluate calibration before scaling
    print("\n3. Evaluating calibration before temperature scaling...")
    metrics_before = evaluate_calibration(model, loader, device, num_bins=10)

    print("\n   Calibration Metrics (BEFORE scaling):")
    print(f"   ECE:                {metrics_before.ece:.4f}")
    print(f"   MCE:                {metrics_before.mce:.4f}")
    print(f"   Accuracy:           {metrics_before.accuracy:.4f}")
    print(f"   Avg Confidence:     {metrics_before.avg_confidence:.4f}")

    # Apply temperature scaling
    print("\n4. Applying temperature scaling...")
    result, metrics_before_full, metrics_after = apply_temperature_scaling(
        model, loader, device, num_bins=10
    )

    print(f"\n   Optimal Temperature: {result.temperature:.4f}")
    print(f"   ECE Before:          {result.ece_before:.4f}")
    print(f"   ECE After:           {result.ece_after:.4f}")
    print(f"   ECE Reduction:       {result.ece_before - result.ece_after:.4f}")
    print(f"   NLL Before:          {result.nll_before:.4f}")
    print(f"   NLL After:           {result.nll_after:.4f}")

    # Generate reliability diagram
    print("\n5. Generating reliability diagram...")
    output_dir = Path("outputs/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_reliability_diagram(
        metrics_before_full,
        metrics_after,
        "ViT-Base",
        output_dir / "reliability_diagram_demo.png",
    )
    print(f"   ✓ Saved to {output_dir / 'reliability_diagram_demo.png'}")

    # Save detailed report
    report_path = output_dir / "calibration_report_demo.txt"
    with open(report_path, "w") as f:
        f.write("CALIBRATION EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model: ViT-Base (synthetic data)\n\n")

        f.write("BEFORE TEMPERATURE SCALING:\n")
        f.write(f"  ECE:             {metrics_before_full.ece:.6f}\n")
        f.write(f"  MCE:             {metrics_before_full.mce:.6f}\n")
        f.write(f"  Accuracy:        {metrics_before_full.accuracy:.4f}\n")
        f.write(f"  Avg Confidence:  {metrics_before_full.avg_confidence:.4f}\n\n")

        f.write("TEMPERATURE SCALING:\n")
        f.write(f"  Optimal T:       {result.temperature:.6f}\n")
        f.write(f"  NLL Before:      {result.nll_before:.6f}\n")
        f.write(f"  NLL After:       {result.nll_after:.6f}\n\n")

        f.write("AFTER TEMPERATURE SCALING:\n")
        f.write(f"  ECE:             {metrics_after.ece:.6f}\n")
        f.write(f"  MCE:             {metrics_after.mce:.6f}\n")
        f.write(f"  Accuracy:        {metrics_after.accuracy:.4f}\n")
        f.write(f"  Avg Confidence:  {metrics_after.avg_confidence:.4f}\n\n")

        f.write("IMPROVEMENT:\n")
        f.write(f"  ECE Reduction:   {result.ece_before - result.ece_after:.6f}\n")
        f.write(
            f"  ECE Reduction %: "
            f"{(result.ece_before - result.ece_after) / result.ece_before * 100:.2f}%\n"
        )

    print(f"   ✓ Saved report to {report_path}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nAcceptance Criteria Status:")
    print("✓ ECE is computed before/after temperature scaling")
    print("✓ Reliability diagrams are generated")
    print(f"\nCheck {output_dir}/ for results!")


def compare_models_calibration():
    """Demo: Compare calibration of different model architectures."""
    print("\n" + "=" * 70)
    print("DEMO: Comparing Calibration Across Architectures")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create synthetic dataset
    num_samples = 100
    images = torch.randn(num_samples, 3, 224, 224)
    targets = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, targets)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)

    models = {
        "ViT-Base": "vit_base",
        "ResNet50": "resnet50",
    }

    results = {}

    for model_name, model_arch in models.items():
        print(f"\nEvaluating {model_name}...")
        model = create_model(model_arch, num_classes=10, pretrained=False)
        model = model.to(device)
        model.eval()

        # Evaluate calibration
        result, metrics_before, metrics_after = apply_temperature_scaling(
            model, loader, device, num_bins=10
        )

        results[model_name] = {
            "result": result,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        }

        print(f"  ECE Before: {result.ece_before:.4f}")
        print(f"  ECE After:  {result.ece_after:.4f}")
        print(f"  Temperature: {result.temperature:.4f}")

    # Generate comparison visualization
    print("\nGenerating comparison visualization...")
    output_dir = Path("outputs/calibration")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (model_name, data) in enumerate(results.items()):
        row = idx
        metrics_before = data["metrics_before"]
        metrics_after = data["metrics_after"]

        # Plot before
        ax = axes[row, 0]
        diagram_data = generate_reliability_diagram_data(metrics_before)
        ax.bar(
            diagram_data["bin_centers"],
            diagram_data["bin_accuracies"],
            width=1.0 / len(diagram_data["bin_centers"]),
            alpha=0.7,
            label="Accuracy",
            color="steelblue",
            edgecolor="black",
        )
        ax.plot(
            diagram_data["bin_centers"],
            diagram_data["bin_confidences"],
            "r-o",
            linewidth=2,
            label="Confidence",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_name} - Before (ECE: {metrics_before.ece:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Plot after
        ax = axes[row, 1]
        diagram_data = generate_reliability_diagram_data(metrics_after)
        ax.bar(
            diagram_data["bin_centers"],
            diagram_data["bin_accuracies"],
            width=1.0 / len(diagram_data["bin_centers"]),
            alpha=0.7,
            label="Accuracy",
            color="green",
            edgecolor="black",
        )
        ax.plot(
            diagram_data["bin_centers"],
            diagram_data["bin_confidences"],
            "r-o",
            linewidth=2,
            label="Confidence",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_name} - After (ECE: {metrics_after.ece:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_demo.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved to {output_dir / 'model_comparison_demo.png'}")
    plt.close()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("CALIBRATION & UNCERTAINTY QUANTIFICATION DEMO - Epic 3")
    print("=" * 70)
    print("\nThis demo shows how to use the calibration functionality:")
    print("1. Computing ECE before/after temperature scaling")
    print("2. Temperature scaling for probability calibration")
    print("3. Generating reliability diagrams")
    print()

    # Run demonstrations
    demo_calibration_with_synthetic_data()
    compare_models_calibration()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nAcceptance Criteria Status:")
    print("✓ ECE is computed before/after temperature scaling")
    print("✓ Reliability diagrams are generated for ViT and ResNet")
    print("\nCheck the outputs/calibration/ directory for results!")


if __name__ == "__main__":
    main()
