import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from eurosat_vit_analysis.experiment import run_experiment


def create_dummy_dataset(root: Path, num_classes: int = 2, images_per_class: int = 2):
    """Create a dummy ImageFolder dataset structure."""
    for i in range(num_classes):
        class_dir = root / f"class_{i}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for j in range(images_per_class):
            img_path = class_dir / f"img_{j}.jpg"
            # Create a simple 64x64 RGB image
            Image.new("RGB", (64, 64), color=(i * 10, j * 10, 0)).save(img_path)


def test_e2e_training_run(tmp_path: Path):
    """
    End-to-end test of the experiment runner.

    This test:
    1. Creates a dummy dataset on disk.
    2. Runs the full experiment pipeline (training + eval) using ResNet50.
    3. Verifies that a manifest is produced and metrics are logged.

    We patch wandb to avoid network calls, but we let the rest of the code run for real.
    """
    dataset_path = tmp_path / "data"
    output_dir = tmp_path / "output"

    create_dummy_dataset(dataset_path, num_classes=2, images_per_class=4)

    config = {
        "seed": 42,
        "dataset_path": str(dataset_path),
        "dataset_version": "e2e-test",
        "augmentation": "none",
        "batch_size": 2,
        "model": {
            "name": "resnet50",
            "lr": 0.001,
            "epochs": 1,
            "freeze_backbone": True,
        },
        "wandb": {"project": "test-project"},
    }

    # We patch wandb so we don't spam the real server during CI/testing
    with patch("eurosat_vit_analysis.experiment.wandb") as mock_wandb:
        manifest_path = run_experiment(config, output_dir=output_dir)

        # Verify W&B was initialized and used
        mock_wandb.init.assert_called_once()
        mock_wandb.log.assert_called()
        mock_wandb.finish.assert_called_once()

    # Verify manifest creation
    assert manifest_path.exists()

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Verify content
    assert manifest["dataset_version"] == "e2e-test"
    assert "accuracy" in manifest["metrics"]
    assert "loss" in manifest["metrics"]

    # Verify we actually trained (or at least ran the loop)
    # Since we have dummy data and 1 epoch, metrics might be garbage, but they
    # should exist.
    assert isinstance(manifest["metrics"]["loss"], float)
