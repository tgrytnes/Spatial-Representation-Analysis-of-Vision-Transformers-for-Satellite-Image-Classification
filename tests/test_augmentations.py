import pytest
from torchvision import transforms

from eurosat_vit_analysis.data import get_transforms


@pytest.mark.parametrize("strength", ["none", "light", "medium", "strong"])
def test_get_transforms_returns_compose(strength):
    """Test that get_transforms returns a Compose object."""
    transform = get_transforms(strength=strength)
    assert isinstance(transform, transforms.Compose)


def test_augmentation_none_structure():
    """Test 'none' augmentation structure (Resize -> ToTensor -> Normalize)."""
    t = get_transforms(strength="none")
    ops = t.transforms
    assert isinstance(ops[0], transforms.Resize)
    assert isinstance(ops[1], transforms.ToTensor)
    assert isinstance(ops[2], transforms.Normalize)


def test_augmentation_light_structure():
    """Test 'light' includes HorizontalFlip."""
    t = get_transforms(strength="light")
    ops = t.transforms
    # [Resize, RandomHorizontalFlip, ToTensor, Normalize]
    assert any(isinstance(op, transforms.RandomHorizontalFlip) for op in ops)


def test_augmentation_strong_structure():
    """Test 'strong' includes RandAugment."""
    t = get_transforms(strength="strong")
    ops = t.transforms
    # [Resize, RandAugment, ToTensor, Normalize]
    assert any(isinstance(op, transforms.RandAugment) for op in ops)


def test_invalid_strength_raises_error():
    """Test that invalid augmentation strength raises ValueError."""
    with pytest.raises(ValueError):
        get_transforms(strength="super_saiyan")
