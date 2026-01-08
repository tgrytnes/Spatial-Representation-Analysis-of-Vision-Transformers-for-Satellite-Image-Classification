from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from eurosat_vit_analysis.vis.attention import (
    capture_attention_maps,
    compute_attention_rollout,
    overlay_heatmap,
)


def _dummy_attention(
    image_size: tuple[int, int], patch_size: int, layers: int = 2, heads: int = 2
) -> torch.Tensor:
    height, width = image_size
    num_patches = (height // patch_size) * (width // patch_size)
    tokens = num_patches + 1
    return torch.rand(layers, heads, tokens, tokens)


def _load_model(name: str) -> torch.nn.Module | None:
    if name == "dummy":
        return None
    import timm

    return timm.create_model(name, pretrained=True).eval()


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention rollout demo.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output overlay image path."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name or 'dummy' for random attention.",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv or [])

    image = Image.open(args.image).convert("RGB")
    image = image.resize((args.image_size, args.image_size))

    model = _load_model(args.model)
    if model is None:
        attentions = _dummy_attention(
            (args.image_size, args.image_size), args.patch_size
        )
    else:
        tensor = transforms.ToTensor()(image).unsqueeze(0)
        attentions = capture_attention_maps(model, tensor)

    heatmap = compute_attention_rollout(
        attentions,
        image_size=(args.image_size, args.image_size),
        patch_size=args.patch_size,
    )
    overlay = overlay_heatmap(image, np.array(heatmap), alpha=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
