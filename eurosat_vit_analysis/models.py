import timm
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def create_model(
    model_name: str,
    num_classes: int = 10,
    freeze_backbone: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
) -> nn.Module:
    """
    Factory function to create models using timm, optionally with LoRA.

    Args:
        model_name (str): Name of the model ('swin_t', 'vit_base', 'resnet50',
            'convnext_t').
        num_classes (int): Number of output classes.
        freeze_backbone (bool): Whether to freeze backbone parameters.
        use_lora (bool): Whether to use LoRA (Low-Rank Adaptation).
        lora_r (int): LoRA rank.

    Returns:
        nn.Module: The created model.
    """
    timm_model_names = {
        "swin_t": "swin_tiny_patch4_window7_224",
        "vit_base": "vit_base_patch16_224",
        "resnet50": "resnet50",
        "convnext_t": "convnext_tiny",
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

    if use_lora:
        # Define target modules based on architecture
        target_modules = []
        if "swin" in model_name or "vit" in model_name:
            # Common attention targets for Transformers
            target_modules = ["qkv", "q_proj", "v_proj", "fc1", "fc2"]
        elif "resnet" in model_name:
            # ResNet targets (Conv2d is also supported by recent PEFT, or Linear layers)
            # PEFT LoRA supports Conv2d now.
            target_modules = ["conv1", "conv2", "fc"]
        elif "convnext" in model_name:
            target_modules = ["conv_dw", "mlp.fc1", "mlp.fc2"]

        # If we can't guess, let PEFT try to find standard names or user provides config
        # For this MVP, we stick to simple rules.

        # NOTE: timm layer naming varies.
        # Swin: layers.0.blocks.0.attn.qkv
        # ViT: blocks.0.attn.qkv

        # We'll use a more generic list that covers most timm transformer patterns
        if not target_modules and ("swin" in model_name or "vit" in model_name):
            target_modules = ["qkv", "proj"]

        peft_config = LoraConfig(
            inference_mode=False,
            r=lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
            modules_to_save=["head", "fc", "classifier"],  # Train the head!
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    if freeze_backbone:
        for name, param in model.named_parameters():
            # Standard timm naming for classifier head is usually 'head' or 'fc'
            # For Swin and ViT it is typically 'head'.
            # For ResNet it is 'fc'.
            # For ConvNeXt it is 'head'.
            if "head" not in name and "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    return model
