from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

import wandb
from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.models import create_model

CONFIG_DIR = Path("configs")
DEFAULT_CONFIG = CONFIG_DIR / "experiment.yaml"
MANIFEST_DIR = Path("manifests")


def load_config(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def current_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def emit_manifest(
    metrics: dict[str, Any],
    config: dict[str, Any],
    manifest_dir: Path = MANIFEST_DIR,
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "git_sha": current_git_sha(),
        "dataset_version": config.get("dataset_version"),
        "seed": config.get("seed"),
        "params": config,
        "metrics": metrics,
        "timestamp": timestamp,
    }
    manifest_path = manifest_dir / f"manifest-{timestamp}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    loss = total_loss / total
    # Placeholder for macro F1
    f1_macro = accuracy

    return loss, accuracy, f1_macro


def run_experiment(config: dict[str, Any], output_dir: Path | None = None) -> Path:
    # Initialize wandb
    wandb_config = config.get("wandb", {})
    wandb.init(
        project=wandb_config.get("project", "eurosat-vit-analysis"),
        config=config,
        job_type="experiment",
    )

    seed = config.get("seed", 42)
    set_deterministic_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # Data
    data_dir = config.get("dataset_path", "data/eurosat")
    batch_size = config.get("batch_size", 32)
    augmentation = config.get("augmentation", "none")

    # Check if data exists
    if not Path(data_dir).exists():
        print(f"Data directory {data_dir} not found. Skipping training.")
        metrics = {"accuracy": 0.0, "loss": 0.0, "f1_macro": 0.0}
        wandb.log(metrics)
        path = emit_manifest(metrics, config, manifest_dir=output_dir or MANIFEST_DIR)
        wandb.finish()
        return path

    train_loader, val_loader, class_names = prepare_data(
        data_dir=data_dir, batch_size=batch_size, seed=seed, augmentation=augmentation
    )

    # Model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "swin_t")
    freeze_backbone = model_config.get("freeze_backbone", False)
    use_lora = model_config.get("use_lora", False)
    lora_r = model_config.get("lora_r", 16)

    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        lora_r=lora_r,
    ).to(device)

    # Log parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction = 100 * (1 - trainable_params / total_params)

    print(f"Total Params: {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Reduction: {reduction:.2f}%")

    wandb.config.update(
        {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_reduction_pct": reduction,
        }
    )

    # Optimizer & Loss
    lr = float(model_config.get("lr", 1e-4))
    epochs = int(model_config.get("epochs", 5))

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    final_metrics = {}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "accuracy": val_acc,
            "f1_macro": val_f1,
            "loss": val_loss,
        }
        wandb.log(metrics)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            final_metrics = metrics

    # If no epochs run, ensure we have some metrics
    if not final_metrics:
        final_metrics = {"accuracy": 0.0, "loss": 0.0, "f1_macro": 0.0}

    manifest_path = emit_manifest(
        final_metrics, config, manifest_dir=output_dir or MANIFEST_DIR
    )
    print("Experiment run complete.")
    print("Best Metrics:", final_metrics)
    print("Manifest:", manifest_path)

    wandb.finish()
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=MANIFEST_DIR,
        help="Directory where manifests are written.",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found at {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    run_experiment(config, output_dir=args.manifest_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
