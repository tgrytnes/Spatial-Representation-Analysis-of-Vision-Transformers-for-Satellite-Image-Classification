from pathlib import Path

import pytest
from PIL import Image
from torch.utils.data import DataLoader

from eurosat_vit_analysis.data import get_transforms, prepare_data


def test_get_transforms_returns_compose() -> None:
    transforms = get_transforms(image_size=16)
    assert hasattr(transforms, "transforms")


def test_prepare_data_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_data(tmp_path / "missing")


def test_prepare_data_creates_dataloaders(tmp_path: Path) -> None:
    class_dir = tmp_path / "class_a"
    class_dir.mkdir(parents=True, exist_ok=True)
    image_path = class_dir / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    train_loader, val_loader, classes = prepare_data(
        tmp_path, batch_size=1, num_workers=0, val_split=0.0, seed=123
    )
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert classes == ["class_a"]
