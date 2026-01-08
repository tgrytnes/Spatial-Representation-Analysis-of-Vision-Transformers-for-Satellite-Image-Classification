from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import torch

from eurosat_vit_analysis.robustness import FgsmReport


def _report_for_eps(eps_values: list[float]) -> FgsmReport:
    adv_accuracy = {value: 0.7 for value in eps_values}
    accuracy_drop = {value: 0.2 for value in eps_values}
    return FgsmReport(
        clean_accuracy=0.9,
        adv_accuracy=adv_accuracy,
        accuracy_drop=accuracy_drop,
    )


@patch("eurosat_vit_analysis.cli.fgsm_eval.evaluate_fgsm")
@patch("eurosat_vit_analysis.cli.fgsm_eval.build_normalization_tensors")
@patch("eurosat_vit_analysis.cli.fgsm_eval.create_model")
@patch("eurosat_vit_analysis.cli.fgsm_eval.prepare_data")
@patch("eurosat_vit_analysis.cli.fgsm_eval._select_device")
@patch("eurosat_vit_analysis.cli.fgsm_eval.torch.load")
def test_main_writes_report(
    mock_torch_load,
    mock_select_device,
    mock_prepare_data,
    mock_create_model,
    mock_build_norm,
    mock_evaluate_fgsm,
    tmp_path,
) -> None:
    from eurosat_vit_analysis.cli import fgsm_eval

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    report_path = tmp_path / "report.json"

    mock_select_device.return_value = torch.device("cpu")
    mock_prepare_data.return_value = (None, "loader", ["c1", "c2"])

    vit_model = MagicMock()
    vit_model.to.return_value = vit_model
    resnet_model = MagicMock()
    resnet_model.to.return_value = resnet_model
    mock_create_model.side_effect = [vit_model, resnet_model]

    mock_build_norm.return_value = (
        torch.tensor([1.0]),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
    )

    eps_values = [2 / 255, 4 / 255]
    mock_evaluate_fgsm.side_effect = [
        _report_for_eps(eps_values),
        _report_for_eps(eps_values),
    ]

    mock_torch_load.return_value = {
        "state_dict": {"module.head.weight": torch.randn(1)}
    }

    args = [
        "--data-dir",
        str(data_dir),
        "--output",
        str(report_path),
        "--vit-checkpoint",
        str(tmp_path / "vit.pt"),
        "--resnet-checkpoint",
        str(tmp_path / "resnet.pt"),
    ]
    exit_code = fgsm_eval.main(args)

    assert exit_code == 0
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["models"]["vit"]["adv_accuracy"]["2/255"] == 0.7
    assert payload["models"]["resnet"]["accuracy_drop"]["4/255"] == 0.2
    assert vit_model.load_state_dict.called
    assert resnet_model.load_state_dict.called


def test_main_returns_error_when_data_missing(tmp_path) -> None:
    from eurosat_vit_analysis.cli import fgsm_eval

    data_dir = tmp_path / "missing"
    exit_code = fgsm_eval.main(["--data-dir", str(data_dir)])

    assert exit_code == 1
