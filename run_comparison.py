import json
import shutil
import time
from pathlib import Path

from PIL import Image

from eurosat_vit_analysis.experiment import run_experiment


def create_dummy_dataset(root: Path, num_classes: int = 2, images_per_class: int = 10):
    """Create a dummy ImageFolder dataset structure."""
    if root.exists():
        shutil.rmtree(root)

    for i in range(num_classes):
        class_dir = root / f"class_{i}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for j in range(images_per_class):
            img_path = class_dir / f"img_{j}.jpg"
            # Create a simple 64x64 RGB image
            Image.new("RGB", (64, 64), color=(i * 10, j * 10, 0)).save(img_path)


def run_comparison():
    base_dir = Path("comparison_run")
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"

    print(f"Setting up dummy dataset at {data_dir}...")
    create_dummy_dataset(data_dir)

    # Common config
    base_config = {
        "seed": 42,
        "dataset_path": str(data_dir),
        "dataset_version": "comparison-test",
        "augmentation": "none",
        "batch_size": 4,
        "wandb": {"project": "peft-comparison"},
        "model": {
            "name": "swin_t",
            "lr": 0.001,
            "epochs": 1,
        },
    }

    results = []

    # 1. Full Fine-Tuning
    print("\n--- Running Full Fine-Tuning ---")
    config_full = base_config.copy()
    config_full["model"] = base_config["model"].copy()
    config_full["model"]["use_lora"] = False
    config_full["model"]["freeze_backbone"] = False

    start_time = time.time()
    manifest_full = run_experiment(config_full, output_dir=output_dir / "full")
    duration_full = time.time() - start_time

    with open(manifest_full) as f:
        data_full = json.load(f)
    results.append(
        {
            "Mode": "Full Fine-Tune",
            "Accuracy": data_full["metrics"]["accuracy"],
            "Loss": data_full["metrics"]["loss"],
            "Time (s)": duration_full,
        }
    )

    # 2. LoRA
    print("\n--- Running LoRA ---")
    config_lora = base_config.copy()
    config_lora["model"] = base_config["model"].copy()
    config_lora["model"]["use_lora"] = True
    config_lora["model"]["lora_r"] = 16

    start_time = time.time()
    manifest_lora = run_experiment(config_lora, output_dir=output_dir / "lora")
    duration_lora = time.time() - start_time

    with open(manifest_lora) as f:
        data_lora = json.load(f)
    results.append(
        {
            "Mode": "LoRA (r=16)",
            "Accuracy": data_lora["metrics"]["accuracy"],
            "Loss": data_lora["metrics"]["loss"],
            "Time (s)": duration_lora,
        }
    )

    # Report
    print("\n" + "=" * 60)
    print(f"{ 'Mode':<20} | { 'Accuracy':<10} | { 'Loss':<10} | { 'Time (s)':<10}")
    print("-" * 60)
    for res in results:
        print(
            f"{res['Mode']:<20} | {res['Accuracy']:<10.4f} | "
            f"{res['Loss']:<10.4f} | {res['Time (s)']:<10.2f}"
        )
    print("=" * 60)

    # Cleanup
    shutil.rmtree(base_dir)


if __name__ == "__main__":
    run_comparison()
