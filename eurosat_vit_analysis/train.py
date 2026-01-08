# train.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from eurosat_vit_analysis.data import prepare_data
from eurosat_vit_analysis.experiment import (
    evaluate,
    set_deterministic_seed,
    train_one_epoch,
)
from eurosat_vit_analysis.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save best checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/eurosat"))
    parser.add_argument("--model", type=str, default="vit_base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation", type=str, default="light")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_deterministic_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = prepare_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augmentation=args.augmentation,
    )

    model = create_model(
        model_name=args.model,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)
    print(f"Saved best checkpoint to {args.out} (val_acc={best_acc:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
