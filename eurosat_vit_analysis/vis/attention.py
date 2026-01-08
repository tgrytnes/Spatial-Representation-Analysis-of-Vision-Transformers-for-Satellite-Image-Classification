from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_attention_rollout(
    attentions: torch.Tensor,
    image_size: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    """
    Compute attention rollout heatmap for a ViT-style model.

    Args:
        attentions: Tensor shaped (layers, heads, tokens, tokens).
        image_size: (height, width) of the input image.
        patch_size: Patch size used by the model.

    Returns:
        Heatmap tensor shaped (height, width), normalized to [0, 1].
    """
    if attentions.ndim != 4:
        raise ValueError("attentions must be shaped (layers, heads, tokens, tokens)")

    height, width = image_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")

    num_patches = (height // patch_size) * (width // patch_size)
    tokens = num_patches + 1  # class token
    if attentions.shape[-1] != tokens or attentions.shape[-2] != tokens:
        raise ValueError(
            "attention token dimensions do not match image_size/patch_size"
        )

    # Average over heads -> (layers, tokens, tokens)
    attn = attentions.mean(dim=1)

    # Add residual connection, normalize rows
    identity = torch.eye(tokens, device=attn.device, dtype=attn.dtype)
    attn = attn + identity
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # Rollout: multiply attention matrices from first to last layer
    joint_attn = attn[0]
    for layer_attn in attn[1:]:
        joint_attn = layer_attn @ joint_attn

    # CLS token attention to patches
    cls_attn = joint_attn[0, 1:]
    grid_size = (height // patch_size, width // patch_size)
    heatmap = cls_attn.reshape(1, 1, grid_size[0], grid_size[1])
    heatmap = F.interpolate(
        heatmap, size=image_size, mode="bilinear", align_corners=False
    )
    heatmap = heatmap.squeeze(0).squeeze(0)

    # Normalize to [0, 1]
    min_val = heatmap.min()
    max_val = heatmap.max()
    if (max_val - min_val) > 0:
        heatmap = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap = torch.zeros_like(heatmap)

    return heatmap
