import json
from unittest.mock import MagicMock, patch

import torch

from eurosat_vit_analysis.experiment import run_experiment


def base_config() -> dict:
    return {
        "seed": 123,
        "dataset_version": "unit-test",
        "model": {"name": "swin_t", "freeze_backbone": True, "epochs": 1},
        "batch_size": 2,
        "dataset_path": "dummy_path",
    }


@patch("eurosat_vit_analysis.experiment.wandb")
@patch("eurosat_vit_analysis.experiment.prepare_data")
@patch("eurosat_vit_analysis.experiment.train_one_epoch")
@patch("eurosat_vit_analysis.experiment.evaluate")
@patch("pathlib.Path.exists", return_value=True)  # Mock data path exists
def test_run_experiment_creates_manifest(
    mock_exists, mock_evaluate, mock_train, mock_prepare_data, mock_wandb, tmp_path
) -> None:
    # Setup mocks
    mock_loader = MagicMock()
    mock_prepare_data.return_value = (mock_loader, mock_loader, ["c1", "c2"])
    mock_train.return_value = (0.5, 0.8)  # loss, acc
    mock_evaluate.return_value = (0.4, 0.85, 0.85)  # loss, acc, f1

    config = base_config()
    manifest_path = run_experiment(config, output_dir=tmp_path)
    assert manifest_path.exists()


@patch("eurosat_vit_analysis.experiment.wandb")
@patch("eurosat_vit_analysis.experiment.prepare_data")
@patch("eurosat_vit_analysis.experiment.train_one_epoch")
@patch("eurosat_vit_analysis.experiment.evaluate")
@patch("pathlib.Path.exists", return_value=True)
def test_metrics_are_deterministic(
    mock_exists, mock_evaluate, mock_train, mock_prepare_data, mock_wandb, tmp_path
) -> None:
    # Setup mocks to return SAME values for deterministic check
    mock_loader = MagicMock()
    mock_prepare_data.return_value = (mock_loader, mock_loader, ["c1", "c2"])
    mock_train.return_value = (0.5, 0.8)
    mock_evaluate.return_value = (0.4, 0.85, 0.85)

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
@patch("eurosat_vit_analysis.experiment.prepare_data")
@patch("eurosat_vit_analysis.experiment.train_one_epoch")
@patch("eurosat_vit_analysis.experiment.evaluate")
@patch("pathlib.Path.exists", return_value=True)
def test_run_experiment_logs_to_wandb(
    mock_exists,
    mock_evaluate,
    mock_train,
    mock_prepare_data,
    mock_wandb,
    mock_create_model,
    tmp_path,
) -> None:
    """Test that run_experiment initializes wandb and logs metrics."""
    # Setup mocks
    mock_loader = MagicMock()
    mock_prepare_data.return_value = (mock_loader, mock_loader, ["c1", "c2"])
    mock_train.return_value = (0.5, 0.8)
    mock_evaluate.return_value = (0.4, 0.85, 0.85)

    mock_model = MagicMock()
    # Create a mock parameter that acts like a tensor with numel() returning an int
    mock_param = MagicMock(spec=torch.Tensor)
    mock_param.numel.return_value = 1000
    mock_param.requires_grad = True

    mock_model.parameters.return_value = [mock_param]
    mock_create_model.return_value = mock_model
    # Ensure .to(device) returns the same mock with parameters
    mock_model.to.return_value = mock_model

    config = base_config()
    run_experiment(config, output_dir=tmp_path)

    # Check wandb init
    mock_wandb.init.assert_called_once()

    # Check metrics logging (should be called at least once per epoch)
    assert mock_wandb.log.called
    logged_metrics = mock_wandb.log.call_args[0][0]
    assert "accuracy" in logged_metrics
    assert "f1_macro" in logged_metrics
    assert "loss" in logged_metrics
