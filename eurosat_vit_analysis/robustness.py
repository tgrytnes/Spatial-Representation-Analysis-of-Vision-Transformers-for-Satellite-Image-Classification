from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FgsmReport:
    clean_accuracy: float
    adv_accuracy: dict[float, float]
    accuracy_drop: dict[float, float]


def parse_epsilons(spec: str) -> tuple[list[float], list[str]]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    eps_values: list[float] = []
    labels: list[str] = []
    for part in parts:
        if "/" in part:
            eps = float(Fraction(part))
        else:
            eps = float(part)
        eps_values.append(eps)
        labels.append(part)
    return eps_values, labels


def build_normalization_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, -1, 1, 1)
    clamp_min = (0 - mean) / std
    clamp_max = (1 - mean) / std
    return std, clamp_min, clamp_max


def _as_broadcast_tensor(
    value: float | torch.Tensor, device: torch.device
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        return tensor.view(1, 1, 1, 1)
    if tensor.ndim == 1:
        return tensor.view(1, -1, 1, 1)
    return tensor


def fgsm_attack(
    inputs: torch.Tensor,
    data_grad: torch.Tensor,
    epsilon: torch.Tensor,
    clamp_min: torch.Tensor | float | None = None,
    clamp_max: torch.Tensor | float | None = None,
) -> torch.Tensor:
    sign_data_grad = data_grad.sign()
    perturbed = inputs + epsilon * sign_data_grad

    if clamp_min is not None and clamp_max is not None:
        min_tensor = _as_broadcast_tensor(clamp_min, perturbed.device)
        max_tensor = _as_broadcast_tensor(clamp_max, perturbed.device)
        perturbed = torch.max(torch.min(perturbed, max_tensor), min_tensor)

    return perturbed


def evaluate_fgsm(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epsilons: Iterable[float],
    criterion: nn.Module,
    clamp_min: torch.Tensor | float | None = None,
    clamp_max: torch.Tensor | float | None = None,
    std: torch.Tensor | None = None,
) -> FgsmReport:
    model.eval()

    eps_list = list(epsilons)
    adv_correct = {float(epsilon): 0 for epsilon in eps_list}
    total = 0
    clean_correct = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs.clone().detach().requires_grad_(True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        model.zero_grad()
        loss.backward()
        data_grad = inputs.grad.detach()

        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            clean_correct += preds.eq(targets).sum().item()

        for epsilon in eps_list:
            epsilon_tensor = _as_broadcast_tensor(epsilon, device)
            if std is not None:
                epsilon_tensor = epsilon_tensor / std

            perturbed = fgsm_attack(
                inputs,
                data_grad,
                epsilon_tensor,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            ).detach()

            with torch.no_grad():
                adv_outputs = model(perturbed)
                adv_preds = adv_outputs.argmax(dim=1)
                adv_correct[float(epsilon)] += adv_preds.eq(targets).sum().item()

        total += targets.size(0)

    clean_accuracy = clean_correct / total if total else 0.0
    adv_accuracy = {
        float(epsilon): adv_correct[float(epsilon)] / total if total else 0.0
        for epsilon in eps_list
    }
    accuracy_drop = {
        float(epsilon): clean_accuracy - adv_accuracy[float(epsilon)]
        for epsilon in eps_list
    }

    return FgsmReport(
        clean_accuracy=clean_accuracy,
        adv_accuracy=adv_accuracy,
        accuracy_drop=accuracy_drop,
    )


__all__ = [
    "FgsmReport",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_normalization_tensors",
    "evaluate_fgsm",
    "fgsm_attack",
    "parse_epsilons",
]
