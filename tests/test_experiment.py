from pathlib import Path

from eurosat_vit_analysis.experiment import run_experiment


def base_config() -> dict:
    return {
        "seed": 123,
        "dataset_version": "unit-test",
        "model": {"name": "test-model"},
        "batch_size": 16,
    }


def test_run_experiment_creates_manifest(tmp_path: Path) -> None:
    config = base_config()
    manifest_path = run_experiment(config, output_dir=tmp_path)
    assert manifest_path.exists()


def test_metrics_are_deterministic(tmp_path: Path) -> None:
    config = base_config()
    first = run_experiment(config, output_dir=tmp_path)
    second = run_experiment(config, output_dir=tmp_path)
    with open(first, encoding="utf-8") as handle:
        first_data = handle.read()
    with open(second, encoding="utf-8") as handle:
        second_data = handle.read()
    assert "metrics" in first_data
    assert first_data == second_data
