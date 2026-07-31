"""Evaluate accuracy and confidence calibration for a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import torch

from src.data.cifar import CIFARDataConfig, build_dataloaders
from src.evaluation.brier import multiclass_brier_score
from src.evaluation.calibration import (
    calibration_bin_stats,
    expected_calibration_error,
)
from src.models.factory import build_model_from_yaml
from src.training.checkpoint import load_checkpoint
from src.training.experiment import ExperimentSpec
from src.utils.runtime import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.inference_mode()
def collect_confidences_and_correctness(model, loader, device):
    """Collect one confidence and correctness value per test sample."""
    model.eval()
    confidence_batches = []
    correctness_batches = []
    probability_batches = []
    target_batches = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        probabilities = torch.softmax(model(images), dim=1)
        confidences, predictions = probabilities.max(dim=1)
        probability_batches.append(probabilities.cpu())
        target_batches.append(targets.cpu())
        confidence_batches.append(confidences.cpu())
        correctness_batches.append((predictions == targets).cpu())
    if not confidence_batches:
        raise ValueError("Cannot evaluate calibration on an empty loader.")
    return (
        torch.cat(probability_batches),
        torch.cat(target_batches),
        torch.cat(confidence_batches),
        torch.cat(correctness_batches),
    )


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
    test_loader = (
        islice(data.test_loader, args.max_test_batches)
        if args.max_test_batches is not None
        else data.test_loader
    )
    probabilities, targets, confidences, correctness = collect_confidences_and_correctness(
        model,
        test_loader,
        device,
    )
    accuracy = correctness.float().mean().item()
    ece = expected_calibration_error(
        confidences,
        correctness,
        num_bins=args.num_bins,
    )
    brier = multiclass_brier_score(probabilities, targets)
    bins = calibration_bin_stats(
        confidences,
        correctness,
        num_bins=args.num_bins,
    )
    payload = {
        "checkpoint_epoch": checkpoint["epoch"],
        "samples": int(correctness.numel()),
        "accuracy": accuracy,
        "ece": ece,
        "brier_score": brier,
        "num_bins": args.num_bins,
        "bins": bins,
    }
    output_path = args.output or spec.output_dir / "calibration_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(
        f"checkpoint_epoch={checkpoint['epoch']} samples={payload['samples']} "
        f"accuracy={accuracy:.4f} ece={ece:.4f} brier={brier:.4f}"
    )
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
