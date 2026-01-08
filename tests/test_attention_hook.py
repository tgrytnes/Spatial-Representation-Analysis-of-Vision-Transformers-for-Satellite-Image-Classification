import torch
from torch import nn

from eurosat_vit_analysis.vis.attention import capture_attention_maps


class DummyAttention(nn.Module):
    def __init__(self, heads: int, tokens: int) -> None:
        super().__init__()
        self.heads = heads
        self.tokens = tokens
        self.attn_drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.rand(x.size(0), self.heads, self.tokens, self.tokens)
        _ = self.attn_drop(attn)
        return x


class DummyModel(nn.Module):
    def __init__(self, heads: int, tokens: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DummyAttention(heads, tokens), DummyAttention(heads, tokens)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def test_capture_attention_maps_returns_layers() -> None:
    model = DummyModel(heads=3, tokens=5)
    inputs = torch.randn(1, 8)

    attn = capture_attention_maps(model, inputs)

    assert attn.shape == (2, 3, 5, 5)
