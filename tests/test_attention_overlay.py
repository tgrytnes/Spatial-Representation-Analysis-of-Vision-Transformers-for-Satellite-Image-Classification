import numpy as np
from PIL import Image

from eurosat_vit_analysis.vis.attention import overlay_heatmap


def test_overlay_heatmap_returns_image_same_size() -> None:
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    heatmap = np.random.rand(32, 32)

    overlay = overlay_heatmap(image, heatmap, alpha=0.4)

    assert overlay.size == image.size
