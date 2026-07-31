"""Evaluate a checkpoint under Gaussian input noise.

Example:
    python -m scripts.evaluate_robustness \
        --config configs/experiment/cosine_crop_flip.yaml
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from src.data.cifar import CIFARDataConfig, build_dataloaders
from src.evaluation.robustness import evaluate_gaussian_noise
from src.models.factory import build_model_from_yaml
from src.training.checkpoint import load_checkpoint
from src.training.experiment import ExperimentSpec
from src.utils.runtime import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ExperimentSpec.from_yaml(args.config)
    seed_everything(spec.seed)
    device = resolve_device(spec.training.device)
    checkpoint_path = args.checkpoint or spec.output_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_config = CIFARDataConfig.from_yaml(spec.data_config)
    data = build_dataloaders(data_config)
    model = build_model_from_yaml(spec.model_config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)

    results = []
    for noise_std in args.noise_stds:
        seed_everything(spec.seed)
        test_loader = (
            islice(data.test_loader, args.max_test_batches)
            if args.max_test_batches is not None
            else data.test_loader
        )
        accuracy = evaluate_gaussian_noise(model, test_loader, device, noise_std)
        results.append({"noise_std": noise_std, "accuracy": accuracy})
        print(f"noise_std={noise_std:.2f} accuracy={accuracy:.4f}")

    output_path = args.output or spec.output_dir / "robustness_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint": str(checkpoint_path),
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
