from pathlib import Path
from unittest.mock import MagicMock, patch

from eurosat_vit_analysis.experiment import run_experiment


def base_config() -> dict:
    return {
        "seed": 123,
        "dataset_version": "unit-test",
        "model": {"name": "swin_t", "freeze_backbone": True},
        "batch_size": 16,
    }


def test_run_experiment_creates_manifest(tmp_path: Path) -> None:
    config = base_config()
    manifest_path = run_experiment(config, output_dir=tmp_path)
    assert manifest_path.exists()


def test_metrics_are_deterministic(tmp_path: Path) -> None:
    config = base_config()
    first = run_experiment(config, output_dir=tmp_path)
    second = run_experiment(config, output_dir=tmp_path)
    with open(first, encoding="utf-8") as handle:
        first_data = handle.read()
    with open(second, encoding="utf-8") as handle:
        second_data = handle.read()
    assert "metrics" in first_data
    assert first_data == second_data


@patch("eurosat_vit_analysis.experiment.create_model")
def test_run_experiment_initializes_model(
    mock_create_model: MagicMock, tmp_path: Path
) -> None:
    """Test that run_experiment calls create_model with correct config."""
    config = base_config()
    run_experiment(config, output_dir=tmp_path)

    mock_create_model.assert_called_once_with(
        model_name="swin_t",
        num_classes=10,  # Assuming default hardcoded for now or from data
        freeze_backbone=True,
    )
