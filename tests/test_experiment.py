import json
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


@patch("eurosat_vit_analysis.experiment.wandb")
def test_run_experiment_creates_manifest(mock_wandb: MagicMock, tmp_path: Path) -> None:
    config = base_config()
    manifest_path = run_experiment(config, output_dir=tmp_path)
    assert manifest_path.exists()


@patch("eurosat_vit_analysis.experiment.wandb")
def test_metrics_are_deterministic(mock_wandb: MagicMock, tmp_path: Path) -> None:
    config = base_config()
    first = run_experiment(config, output_dir=tmp_path)
    second = run_experiment(config, output_dir=tmp_path)

    with open(first, encoding="utf-8") as handle:
        first_data = json.load(handle)
    with open(second, encoding="utf-8") as handle:
        second_data = json.load(handle)

    # Compare deterministic fields
    for field in ["git_sha", "dataset_version", "seed", "params", "metrics"]:
        assert first_data[field] == second_data[field]


@patch("eurosat_vit_analysis.experiment.create_model")
@patch("eurosat_vit_analysis.experiment.wandb")
def test_run_experiment_logs_to_wandb(
    mock_wandb: MagicMock, mock_create_model: MagicMock, tmp_path: Path
) -> None:
    """Test that run_experiment initializes wandb and logs metrics."""
    config = base_config()
    run_experiment(config, output_dir=tmp_path)

    # Check wandb init
    mock_wandb.init.assert_called_once()
    init_kwargs = mock_wandb.init.call_args[1]
    assert init_kwargs["config"] == config
    assert init_kwargs["project"] == "eurosat-vit-analysis"

    # Check metrics logging
    mock_wandb.log.assert_called_once()
    logged_metrics = mock_wandb.log.call_args[0][0]
    assert "accuracy" in logged_metrics
    assert "precision" in logged_metrics
    assert "f1_macro" in logged_metrics
    assert "loss" in logged_metrics
