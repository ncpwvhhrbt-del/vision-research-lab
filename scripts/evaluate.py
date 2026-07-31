"""Evaluate a saved checkpoint on the configured CIFAR-10 test set.

Example:
    python -m scripts.evaluate --config configs/experiment/baseline_simple_cnn.yaml
"""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

from src.data.cifar import CIFARDataConfig, build_dataloaders
from src.evaluation.classification import evaluate_classifier
from src.evaluation.export import save_report_json
from src.models.factory import build_model_from_yaml
from src.training.checkpoint import load_checkpoint
from src.training.experiment import ExperimentSpec
from src.utils.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/baseline_simple_cnn.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <output_dir>/best.pt.",
    )
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ExperimentSpec.from_yaml(args.config)
    device = resolve_device(spec.training.device)
    checkpoint_path = args.checkpoint or spec.output_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_config = CIFARDataConfig.from_yaml(spec.data_config)
    data = build_dataloaders(data_config)
    model = build_model_from_yaml(spec.model_config).to(device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        map_location=device,
    )

    test_loader = (
        islice(data.test_loader, args.max_test_batches)
        if args.max_test_batches is not None
        else data.test_loader
    )
    report = evaluate_classifier(
        model,
        test_loader,
        device,
        data.class_names,
    )
    report_dir = args.output_dir or spec.output_dir
    report_path = save_report_json(report, report_dir / "test_report.json")
    print(
        f"checkpoint_epoch={checkpoint['epoch']} "
        f"test_accuracy={report.overall_accuracy:.4f} "
        f"samples={report.samples}"
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
