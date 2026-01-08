from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eurosat_vit_analysis.robustness import evaluate_fgsm, fgsm_attack


def test_fgsm_attack_applies_sign_and_clamps() -> None:
    inputs = torch.tensor([[[[0.0, 0.5], [1.0, -0.5]]]])
    grads = torch.tensor([[[[1.0, -2.0], [0.0, 3.0]]]])
    epsilon = torch.tensor(0.1)

    perturbed = fgsm_attack(inputs, grads, epsilon, clamp_min=-0.2, clamp_max=0.6)

    expected = torch.clamp(inputs + epsilon * grads.sign(), -0.2, 0.6)
    assert torch.allclose(perturbed, expected)


class SumModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        totals = inputs.view(inputs.size(0), -1).sum(dim=1, keepdim=True)
        return torch.cat([totals, -totals], dim=1)


def test_evaluate_fgsm_reports_accuracy_drop() -> None:
    inputs = torch.tensor(
        [
            [[[-0.25, -0.25], [-0.25, -0.25]]],
            [[[0.25, 0.25], [0.25, 0.25]]],
        ]
    )
    targets = torch.tensor([1, 0])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = SumModel()
    criterion = nn.CrossEntropyLoss()

    def deterministic_attack(
        batch: torch.Tensor,
        _grad: torch.Tensor,
        epsilon: torch.Tensor,
        clamp_min: float | torch.Tensor | None = None,
        clamp_max: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        epsilon_value = torch.as_tensor(epsilon)
        perturbed = batch + epsilon_value
        if clamp_min is not None and clamp_max is not None:
            perturbed = torch.clamp(perturbed, clamp_min, clamp_max)
        return perturbed

    with patch(
        "eurosat_vit_analysis.robustness.fgsm_attack", side_effect=deterministic_attack
    ):
        report = evaluate_fgsm(
            model,
            loader,
            device=torch.device("cpu"),
            epsilons=[0.6],
            criterion=criterion,
            clamp_min=-1.0,
            clamp_max=1.0,
            std=None,
        )

    assert report.clean_accuracy == 1.0
    assert report.adv_accuracy[0.6] == 0.5
    assert report.accuracy_drop[0.6] == 0.5
