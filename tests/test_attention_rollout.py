import torch

from eurosat_vit_analysis.vis.attention import compute_attention_rollout


def test_attention_rollout_shape_matches_image_size() -> None:
    image_size = (32, 32)
    patch_size = 16
    num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    tokens = num_patches + 1  # cls token

    attentions = torch.rand(2, 4, tokens, tokens)
    heatmap = compute_attention_rollout(
        attentions=attentions, image_size=image_size, patch_size=patch_size
    )

    assert heatmap.shape == image_size


def test_attention_rollout_is_normalized() -> None:
    image_size = (32, 32)
    patch_size = 16
    num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    tokens = num_patches + 1

    attentions = torch.rand(3, 2, tokens, tokens)
    heatmap = compute_attention_rollout(
        attentions=attentions, image_size=image_size, patch_size=patch_size
    )

    assert torch.isfinite(heatmap).all()
    assert heatmap.min().item() >= 0.0
    assert heatmap.max().item() <= 1.0
