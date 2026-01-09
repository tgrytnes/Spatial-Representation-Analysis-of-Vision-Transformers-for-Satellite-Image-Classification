#!/usr/bin/env python
"""
Evaluate calibration and uncertainty quantification of trained models on EuroSAT.

This script evaluates:
1. Expected Calibration Error (ECE) before/after temperature scaling
2. Reliability diagrams for model comparison

Usage:
    python evaluate_calibration.py --model vit_base \\
        --checkpoint checkpoints/vit_base_best.pt
    python evaluate_calibration.py --all
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from eurosat_vit_analysis.calibration import (
    apply_temperature_scaling,
    evaluate_calibration,
    generate_reliability_diagram_data,
)
from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.models import create_model


def plot_reliability_diagram(
    metrics_before,
    metrics_after,
    model_name,
    output_path,
):
    """Create reliability diagram comparing before/after calibration."""
    data_before = generate_reliability_diagram_data(metrics_before)
    data_after = generate_reliability_diagram_data(metrics_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Before calibration
    ax1.bar(
        data_before["bin_centers"],
        data_before["bin_accuracies"],
        width=1.0 / len(data_before["bin_centers"]),
        alpha=0.7,
        label="Actual Accuracy",
        color="steelblue",
        edgecolor="black",
    )
    ax1.plot(
        data_before["bin_centers"],
        data_before["bin_confidences"],
        "r-o",
        linewidth=2,
        markersize=8,
        label="Mean Confidence",
    )
    ax1.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.5,
        alpha=0.5,
        label="Perfect Calibration",
    )
    ax1.set_xlabel("Confidence", fontsize=13)
    ax1.set_ylabel("Accuracy", fontsize=13)
    ax1.set_title(
        f"{model_name} - Before Temperature Scaling\nECE: {metrics_before.ece:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # After calibration
    ax2.bar(
        data_after["bin_centers"],
        data_after["bin_accuracies"],
        width=1.0 / len(data_after["bin_centers"]),
        alpha=0.7,
        label="Actual Accuracy",
        color="green",
        edgecolor="black",
    )
    ax2.plot(
        data_after["bin_centers"],
        data_after["bin_confidences"],
        "r-o",
        linewidth=2,
        markersize=8,
        label="Mean Confidence",
    )
    ax2.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.5,
        alpha=0.5,
        label="Perfect Calibration",
    )
    ax2.set_xlabel("Confidence", fontsize=13)
    ax2.set_ylabel("Accuracy", fontsize=13)
    ax2.set_title(
        f"{model_name} - After Temperature Scaling\nECE: {metrics_after.ece:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_model_calibration(
    model_name: str,
    checkpoint_path: str,
    data_dir: str = "data/EuroSAT",
    output_dir: str = "outputs/calibration_evaluation",
    batch_size: int = 32,
    num_bins: int = 15,
):
    """
    Evaluate calibration of a trained model.

    Args:
        model_name: Model architecture ('vit_base', 'resnet50', etc.)
        checkpoint_path: Path to trained checkpoint
        data_dir: Path to EuroSAT dataset
        output_dir: Where to save results
        batch_size: Batch size for evaluation
        num_bins: Number of bins for ECE computation
    """
    print("=" * 80)
    print(f"EVALUATING CALIBRATION: {model_name}")
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

    # Evaluate calibration
    print("\n3. Evaluating calibration before temperature scaling...")
    metrics_before = evaluate_calibration(model, val_loader, device, num_bins)

    print(f"  ECE:             {metrics_before.ece:.6f}")
    print(f"  MCE:             {metrics_before.mce:.6f}")
    print(f"  Accuracy:        {metrics_before.accuracy:.4f}")
    print(f"  Avg Confidence:  {metrics_before.avg_confidence:.4f}")

    # Apply temperature scaling
    print("\n4. Applying temperature scaling...")
    result, _, metrics_after = apply_temperature_scaling(
        model, val_loader, device, num_bins
    )

    print(f"  Optimal Temperature: {result.temperature:.6f}")
    print(f"  ECE Before:          {result.ece_before:.6f}")
    print(f"  ECE After:           {result.ece_after:.6f}")
    print(f"  ECE Improvement:     {result.ece_before - result.ece_after:.6f}")
    improvement_pct = (result.ece_before - result.ece_after) / result.ece_before * 100
    print(f"  Improvement:         {improvement_pct:.2f}%")

    # Generate reliability diagram
    print("\n5. Generating reliability diagram...")
    diagram_path = output_path / "reliability_diagram.png"
    plot_reliability_diagram(metrics_before, metrics_after, model_name, diagram_path)
    print(f"✓ Saved to {diagram_path}")

    # Save detailed report
    print("\n6. Saving detailed calibration report...")
    report_path = output_path / "calibration_report.txt"
    with open(report_path, "w") as f:
        f.write("CALIBRATION EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Validation Samples: {len(val_loader.dataset)}\n\n")

        f.write("BEFORE TEMPERATURE SCALING:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Expected Calibration Error (ECE): {metrics_before.ece:.6f}\n")
        f.write(f"  Maximum Calibration Error (MCE):  {metrics_before.mce:.6f}\n")
        f.write(f"  Accuracy:                          {metrics_before.accuracy:.6f}\n")
        avg_conf = metrics_before.avg_confidence
        f.write(f"  Average Confidence:                {avg_conf:.6f}\n")
        f.write(f"  Negative Log-Likelihood (NLL):     {result.nll_before:.6f}\n\n")

        f.write("TEMPERATURE SCALING:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Optimal Temperature:               {result.temperature:.6f}\n\n")

        f.write("AFTER TEMPERATURE SCALING:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Expected Calibration Error (ECE): {metrics_after.ece:.6f}\n")
        f.write(f"  Maximum Calibration Error (MCE):  {metrics_after.mce:.6f}\n")
        f.write(f"  Accuracy:                          {metrics_after.accuracy:.6f}\n")
        f.write(
            f"  Average Confidence:                {metrics_after.avg_confidence:.6f}\n"
        )
        f.write(f"  Negative Log-Likelihood (NLL):     {result.nll_after:.6f}\n\n")

        f.write("IMPROVEMENT:\n")
        f.write("-" * 40 + "\n")
        ece_reduction = result.ece_before - result.ece_after
        f.write(f"  ECE Reduction (absolute):          {ece_reduction:.6f}\n")
        f.write(f"  ECE Reduction (relative):          {improvement_pct:.2f}%\n")
        nll_reduction = result.nll_before - result.nll_after
        f.write(f"  NLL Reduction:                     {nll_reduction:.6f}\n\n")

        f.write("BIN-WISE ANALYSIS (AFTER CALIBRATION):\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"  {'Bin':<5} {'Confidence':<12} "
            f"{'Accuracy':<12} {'Count':<8} {'Gap':<10}\n"
        )
        f.write("  " + "-" * 55 + "\n")
        data = generate_reliability_diagram_data(metrics_after)
        for i in range(num_bins):
            if data["bin_counts"][i] > 0:
                f.write(
                    f"  {i+1:<5} {data['bin_confidences'][i]:<12.4f} "
                    f"{data['bin_accuracies'][i]:<12.4f} "
                    f"{data['bin_counts'][i]:<8} "
                    f"{data['gap'][i]:+10.4f}\n"
                )

    print(f"✓ Report saved to {report_path}")

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {model_name}")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print(f"  - Calibration report: {report_path.name}")
    print(f"  - Reliability diagram: {diagram_path.name}")

    return result, metrics_before, metrics_after


def compare_model_calibration(
    models: list[tuple[str, str]],
    data_dir: str,
    output_dir: str,
    batch_size: int,
):
    """Generate comparison visualization for multiple models."""
    print("\n" + "=" * 80)
    print("GENERATING MODEL COMPARISON")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data once
    _, val_loader, _ = prepare_data(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        val_split=0.2,
        seed=42,
        augmentation="none",
    )

    results = {}

    for model_name, checkpoint_path in models:
        print(f"\nEvaluating {model_name}...")
        model = create_model(model_name, num_classes=10, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        result, metrics_before, metrics_after = apply_temperature_scaling(
            model, val_loader, device, num_bins=15
        )

        results[model_name] = {
            "result": result,
            "before": metrics_before,
            "after": metrics_after,
        }

    # Create comparison visualization
    print("\nGenerating comparison plot...")
    num_models = len(results)
    fig, axes = plt.subplots(num_models, 2, figsize=(16, 6 * num_models))
    if num_models == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, data) in enumerate(results.items()):
        metrics_before = data["before"]
        metrics_after = data["after"]

        # Before (left column)
        ax = axes[idx, 0]
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
            markersize=6,
            label="Confidence",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            f"{model_name} - Before (ECE: {metrics_before.ece:.4f})",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # After (right column)
        ax = axes[idx, 1]
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
            markersize=6,
            label="Confidence",
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            f"{model_name} - After (ECE: {metrics_after.ece:.4f})",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    comparison_path = Path(output_dir) / "model_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"✓ Comparison saved to {comparison_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate calibration of trained models on EuroSAT"
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
        default="outputs/calibration_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of bins for ECE computation",
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

            evaluate_model_calibration(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_bins=args.num_bins,
            )
            print("\n")

        # Generate comparison
        available_models = [
            (name, path) for name, path in models if Path(path).exists()
        ]
        if len(available_models) > 1:
            compare_model_calibration(
                available_models,
                args.data_dir,
                args.output_dir,
                args.batch_size,
            )

    elif args.model and args.checkpoint:
        # Evaluate single model
        evaluate_model_calibration(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_bins=args.num_bins,
        )

    else:
        parser.print_help()
        print("\nError: Must specify either --all or both --model and --checkpoint")


if __name__ == "__main__":
    main()
