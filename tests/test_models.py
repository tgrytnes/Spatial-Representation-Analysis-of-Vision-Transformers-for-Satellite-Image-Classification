import pytest
import torch

from eurosat_vit_analysis.models import create_model


@pytest.mark.parametrize("model_name", ["swin_t", "vit_base"])
def test_create_model_returns_model(model_name):
    """Test that create_model returns a torch.nn.Module."""
    model = create_model(model_name, num_classes=10)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("model_name", ["swin_t", "vit_base"])
def test_model_output_shape(model_name):
    """Test that the model produces the correct output shape (B, 10)."""
    batch_size = 2
    # EuroSAT images are 64x64, but generic timm models often default to 224x224.
    # However, for these specific models, we verify they can accept an input
    # and return the correct class count.
    # We'll use 224x224 as a safe default for standard ViT/Swin inputs unless specified
    # otherwise.
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    model = create_model(model_name, num_classes=10)
    output = model(input_tensor)

    assert output.shape == (batch_size, 10)


def test_create_model_invalid_name():
    """Test that an invalid model name raises a ValueError."""
    with pytest.raises(ValueError):
        create_model("invalid_model", num_classes=10)


@pytest.mark.parametrize("model_name", ["swin_t", "vit_base"])
def test_freeze_backbone(model_name):
    """Test that freeze_backbone=True freezes the backbone but not the head."""
    model = create_model(model_name, num_classes=10, freeze_backbone=True)

    # We need to identify which parameters are head and which are backbone.
    # In timm, the head is usually 'head' or 'fc'.
    # We will check that at least some parameters are frozen and the head is not.

    frozen_params = []
    active_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            active_params.append(name)
        else:
            frozen_params.append(name)

    assert len(frozen_params) > 0, "Backbone should be frozen"
    assert len(active_params) > 0, "Head should be active"

    # Check that the classifier head is in the active params
    # timm uses 'head' for Swin and ViT usually
    head_found = any("head" in name for name in active_params)
    assert (
        head_found
    ), f"Classifier head should be trainable. Active params: {active_params[:5]}..."
