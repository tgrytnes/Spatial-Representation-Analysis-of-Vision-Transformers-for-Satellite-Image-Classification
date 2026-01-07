import pytest
import torch
from peft import PeftModel

from eurosat_vit_analysis.models import create_model


@pytest.mark.parametrize("model_name", ["swin_t", "vit_base", "resnet50", "convnext_t"])
def test_create_model_returns_model(model_name):
    """Test that create_model returns a torch.nn.Module."""
    model = create_model(model_name, num_classes=10)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize("model_name", ["swin_t", "vit_base", "resnet50", "convnext_t"])
def test_model_output_shape(model_name):
    """Test that the model produces the correct output shape (B, 10)."""
    batch_size = 2
    # EuroSAT images are 64x64, but generic timm models often default to 224x224.
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    model = create_model(model_name, num_classes=10)
    output = model(input_tensor)

    assert output.shape == (batch_size, 10)


def test_create_model_invalid_name():
    """Test that an invalid model name raises a ValueError."""
    with pytest.raises(ValueError):
        create_model("invalid_model", num_classes=10)


@pytest.mark.parametrize("model_name", ["swin_t", "vit_base", "resnet50", "convnext_t"])
def test_freeze_backbone(model_name):
    """Test that freeze_backbone=True freezes the backbone but not the head."""
    model = create_model(model_name, num_classes=10, freeze_backbone=True)

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
    # timm uses 'head' for Swin, ViT, and ConvNeXt, and 'fc' for ResNet
    head_found = any("head" in name or "fc" in name for name in active_params)
    assert (
        head_found
    ), f"Classifier head should be trainable. Active params: {active_params[:5]}..."


@pytest.mark.parametrize("model_name", ["swin_t", "resnet50"])
def test_create_model_lora(model_name):
    """Test that use_lora=True returns a PeftModel with reduced trainable params."""
    # Create full model
    full_model = create_model(model_name, num_classes=10, use_lora=False)
    full_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)

    # Create LoRA model
    lora_model = create_model(model_name, num_classes=10, use_lora=True, lora_r=4)
    lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    # Verify it is a PEFT model
    # Note: Depending on implementation, it might wrap it or be it.
    # peft.get_peft_model returns a PeftModel
    assert isinstance(lora_model, PeftModel) or hasattr(lora_model, "peft_config")

    # Verify trainable parameters are significantly reduced
    # LoRA typically reduces params by >90% compared to full fine-tuning
    assert lora_params < full_params
    assert lora_params > 0  # Should still have SOME trainable params (adapters + head)

    # Check that the head is trainable
    # In PEFT, "modules_to_save" are trainable.
    head_trainable = False
    for n, p in lora_model.named_parameters():
        if ("head" in n or "fc" in n) and p.requires_grad:
            head_trainable = True
            break
    assert head_trainable, "Classifier head should be trainable in LoRA mode"
