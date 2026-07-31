"""Fit temperature on validation logits and compare test calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.cifar import CIFARDataConfig, build_dataloaders
from src.evaluation.brier import multiclass_brier_score
from src.evaluation.calibration import expected_calibration_error
from src.evaluation.temperature_scaling import (
    calibrated_probabilities,
    fit_temperature,
)
from src.models.factory import build_model_from_yaml
from src.training.checkpoint import load_checkpoint
from src.training.experiment import ExperimentSpec
from src.utils.runtime import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def collect_logits(model, loader, device):
    model.eval()
    logits_batches = []
    target_batches = []
    for images, targets in loader:
        logits_batches.append(model(images.to(device)).cpu())
        target_batches.append(targets.cpu())
    if not logits_batches:
        raise ValueError("Cannot collect logits from an empty loader.")
    return torch.cat(logits_batches), torch.cat(target_batches)


def calculate_metrics(logits, targets, num_bins):
    probabilities = torch.softmax(logits, dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correctness = predictions == targets
    return {
        "accuracy": correctness.float().mean().item(),
        "ece": expected_calibration_error(
            confidences,
            correctness,
            num_bins=num_bins,
        ),
        "brier_score": multiclass_brier_score(probabilities, targets),
    }


def main() -> None:
    args = parse_args()
    spec = ExperimentSpec.from_yaml(args.config)
    seed_everything(spec.seed)
    device = resolve_device(spec.training.device)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    data = build_dataloaders(CIFARDataConfig.from_yaml(spec.data_config))
    model = build_model_from_yaml(spec.model_config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device)
    validation_logits, validation_targets = collect_logits(
        model,
        data.val_loader,
        device,
    )
    test_logits, test_targets = collect_logits(model, data.test_loader, device)
    temperature = fit_temperature(validation_logits, validation_targets)

    before = calculate_metrics(test_logits, test_targets, args.num_bins)
    after = calculate_metrics(
        test_logits / temperature,
        test_targets,
        args.num_bins,
    )
    payload = {
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_samples": int(validation_targets.numel()),
        "test_samples": int(test_targets.numel()),
        "temperature": temperature,
        "before": before,
        "after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(
        f"temperature={temperature:.4f} "
        f"before_ece={before['ece']:.4f} after_ece={after['ece']:.4f} "
        f"before_brier={before['brier_score']:.4f} "
        f"after_brier={after['brier_score']:.4f}"
    )
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
