from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.experiment import set_deterministic_seed
from eurosat_vit_analysis.models import create_model
from eurosat_vit_analysis.robustness import (
    FgsmReport,
    build_normalization_tensors,
    evaluate_fgsm,
    parse_epsilons,
)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
        )
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=False)


def _build_report_payload(
    report: FgsmReport,
    eps_labels: list[str],
    eps_values: list[float],
) -> dict[str, float | dict[str, float]]:
    adv_accuracy = {
        label: report.adv_accuracy[value]
        for label, value in zip(eps_labels, eps_values, strict=True)
    }
    accuracy_drop = {
        label: report.accuracy_drop[value]
        for label, value in zip(eps_labels, eps_values, strict=True)
    }
    return {
        "clean_accuracy": report.clean_accuracy,
        "adv_accuracy": adv_accuracy,
        "accuracy_drop": accuracy_drop,
    }


def _print_report(
    title: str, report: FgsmReport, eps_labels: list[str], eps_values: list[float]
) -> None:
    print(f"\n{title}")
    print(f"{'Epsilon':<10} | {'Clean':<8} | {'Adversarial':<11} | {'Drop':<8}")
    print("-" * 48)
    for label, value in zip(eps_labels, eps_values, strict=True):
        adv_acc = report.adv_accuracy[value]
        drop = report.accuracy_drop[value]
        print(
            f"{label:<10} | {report.clean_accuracy:<8.4f} | "
            f"{adv_acc:<11.4f} | {drop:<8.4f}"
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FGSM robustness for ViT vs ResNet on EuroSAT."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/eurosat"),
        help="Path to EuroSAT dataset directory.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker count."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for data splits.")
    parser.add_argument(
        "--epsilons",
        type=str,
        default="2/255,4/255",
        help="Comma-separated list of epsilon values (e.g., '2/255,4/255').",
    )
    parser.add_argument(
        "--vit-model",
        type=str,
        default="vit_base",
        help="ViT model name (create_model key).",
    )
    parser.add_argument(
        "--resnet-model",
        type=str,
        default="resnet50",
        help="ResNet model name (create_model key).",
    )
    parser.add_argument(
        "--vit-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path for the ViT model.",
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path for the ResNet model.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
        return 1

    set_deterministic_seed(args.seed)
    device = _select_device()

    _, val_loader, class_names = prepare_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augmentation="none",
    )

    eps_values, eps_labels = parse_epsilons(args.epsilons)
    std, clamp_min, clamp_max = build_normalization_tensors(device)

    criterion = nn.CrossEntropyLoss()
    pretrained = not args.no_pretrained

    vit_model = create_model(
        model_name=args.vit_model,
        num_classes=len(class_names),
        pretrained=pretrained,
    ).to(device)
    if args.vit_checkpoint:
        _load_checkpoint(vit_model, args.vit_checkpoint)

    resnet_model = create_model(
        model_name=args.resnet_model,
        num_classes=len(class_names),
        pretrained=pretrained,
    ).to(device)
    if args.resnet_checkpoint:
        _load_checkpoint(resnet_model, args.resnet_checkpoint)

    vit_report = evaluate_fgsm(
        vit_model,
        val_loader,
        device=device,
        epsilons=eps_values,
        criterion=criterion,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        std=std,
    )
    resnet_report = evaluate_fgsm(
        resnet_model,
        val_loader,
        device=device,
        epsilons=eps_values,
        criterion=criterion,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        std=std,
    )

    _print_report(f"ViT ({args.vit_model})", vit_report, eps_labels, eps_values)
    _print_report(
        f"ResNet ({args.resnet_model})", resnet_report, eps_labels, eps_values
    )

    if args.output:
        payload = {
            "dataset": str(args.data_dir),
            "seed": args.seed,
            "epsilons": [
                {"label": label, "value": value}
                for label, value in zip(eps_labels, eps_values, strict=True)
            ],
            "models": {
                "vit": {
                    "name": args.vit_model,
                    "checkpoint": str(args.vit_checkpoint)
                    if args.vit_checkpoint
                    else None,
                    **_build_report_payload(vit_report, eps_labels, eps_values),
                },
                "resnet": {
                    "name": args.resnet_model,
                    "checkpoint": str(args.resnet_checkpoint)
                    if args.resnet_checkpoint
                    else None,
                    **_build_report_payload(resnet_report, eps_labels, eps_values),
                },
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
