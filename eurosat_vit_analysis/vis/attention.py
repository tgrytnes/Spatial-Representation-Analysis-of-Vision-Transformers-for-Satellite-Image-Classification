from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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


def capture_attention_maps(
    model: torch.nn.Module, inputs: torch.Tensor
) -> torch.Tensor:
    """
    Capture attention maps from modules that expose an attn_drop submodule.

    Returns a tensor shaped (layers, heads, tokens, tokens). If batch > 1,
    attentions are averaged across the batch dimension.
    """
    captured: list[torch.Tensor] = []
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(store: list[torch.Tensor]):
        def _hook(_module, _inputs, output):
            store.append(output)

        return _hook

    for module in model.modules():
        attn_drop = getattr(module, "attn_drop", None)
        if isinstance(attn_drop, torch.nn.Module):
            hooks.append(attn_drop.register_forward_hook(_make_hook(captured)))

    if not hooks:
        raise ValueError("no attention modules with attn_drop found on model")

    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    for hook in hooks:
        hook.remove()

    if not captured:
        raise ValueError("no attention maps captured from model")

    # Stack -> (layers, batch, heads, tokens, tokens)
    stacked = torch.stack(captured, dim=0)
    if stacked.ndim != 5:
        raise ValueError("captured attention maps have unexpected shape")

    # Average across batch if needed -> (layers, heads, tokens, tokens)
    return stacked.mean(dim=1)


def overlay_heatmap(
    image: Image.Image, heatmap: np.ndarray | torch.Tensor, alpha: float = 0.4
) -> Image.Image:
    """
    Overlay a heatmap onto an RGB image.

    Args:
        image: PIL image.
        heatmap: 2D heatmap array (H, W), values in [0, 1] or unnormalized.
        alpha: Blend factor for the heatmap overlay.

    Returns:
        A new PIL image with the overlay applied.
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array")

    base = np.array(image.convert("RGB"), dtype=np.float32)
    if base.shape[:2] != heatmap.shape:
        raise ValueError("heatmap size must match image size")

    # Normalize heatmap to [0, 1]
    h_min = float(np.min(heatmap))
    h_max = float(np.max(heatmap))
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap = np.clip(heatmap, 0.0, 1.0)
    overlay = np.zeros_like(base)
    overlay[..., 0] = 255.0  # red channel

    blend = (1.0 - alpha) * base + alpha * overlay * heatmap[..., None]
    blend = np.clip(blend, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blend)
