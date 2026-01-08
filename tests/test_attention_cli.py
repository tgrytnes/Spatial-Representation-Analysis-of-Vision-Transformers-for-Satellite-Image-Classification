from pathlib import Path

from PIL import Image

from eurosat_vit_analysis.cli.attention_demo import main


def test_attention_cli_writes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(input_path)

    exit_code = main(
        [
            "--image",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            "dummy",
            "--patch-size",
            "16",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
