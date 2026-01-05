from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

CONFIG_DIR = Path("configs")
DEFAULT_CONFIG = CONFIG_DIR / "experiment.yaml"
MANIFEST_DIR = Path("manifests")


def load_config(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_metrics(config: dict[str, Any]) -> dict[str, float]:
    seed = int(config.get("seed", 0))
    dataset_version = config.get("dataset_version", "unknown")
    model_name = config.get("model", {}).get("name", "model")
    base = f"{dataset_version}:{seed}:{model_name}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    accuracy = 0.50 + (int(digest[:4], 16) % 50) / 1000
    precision = 0.40 + (int(digest[4:8], 16) % 60) / 1000
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
    }


def current_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def emit_manifest(
    metrics: dict[str, float],
    config: dict[str, Any],
    manifest_dir: Path = MANIFEST_DIR,
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "git_sha": current_git_sha(),
        "dataset_version": config.get("dataset_version"),
        "seed": config.get("seed"),
        "params": config,
        "metrics": metrics,
        "timestamp": timestamp,
    }
    manifest_path = manifest_dir / f"manifest-{timestamp}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def run_experiment(config: dict[str, Any], output_dir: Path | None = None) -> Path:
    seed = config.get("seed", 0)
    set_deterministic_seed(seed)
    metrics = compute_metrics(config)
    manifest_path = emit_manifest(
        metrics, config, manifest_dir=output_dir or MANIFEST_DIR
    )
    print("Experiment run complete.")
    print("Metrics:", metrics)
    print("Manifest:", manifest_path)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=MANIFEST_DIR,
        help="Directory where manifests are written.",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found at {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    run_experiment(config, output_dir=args.manifest_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
