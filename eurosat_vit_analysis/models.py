import timm
import torch.nn as nn


def create_model(
    model_name: str, num_classes: int = 10, freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to create models using timm.

    Args:
        model_name (str): Name of the model ('swin_t' or 'vit_base').
        num_classes (int): Number of output classes.
        freeze_backbone (bool): Whether to freeze backbone parameters.

    Returns:
        nn.Module: The created model.
    """
    timm_model_names = {
        "swin_t": "swin_tiny_patch4_window7_224",
        "vit_base": "vit_base_patch16_224",
    }

    if model_name not in timm_model_names:
        raise ValueError(
            f"Model {model_name} not supported. "
            f"Available: {list(timm_model_names.keys())}"
        )

    timm_name = timm_model_names[model_name]

    # Create the model
    # We explicitly set num_classes to ensure the head is correctly sized
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Standard timm naming for classifier head is usually 'head' or 'fc'
            # For Swin and ViT it is typically 'head'.
            if "head" not in name and "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model
