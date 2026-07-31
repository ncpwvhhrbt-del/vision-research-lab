"""Export the most confident incorrect predictions from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.cifar import CIFARDataConfig, build_dataloaders
from src.evaluation.high_confidence import rank_high_confidence_errors
from src.models.factory import build_model_from_yaml
from src.training.checkpoint import load_checkpoint
from src.training.experiment import ExperimentSpec
from src.utils.runtime import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def collect_test_logits(model, loader, device):
    model.eval()
    logits_batches = []
    target_batches = []
    for images, targets in loader:
        logits_batches.append(model(images.to(device)).cpu())
        target_batches.append(targets.cpu())
    if not logits_batches:
        raise ValueError("Cannot analyze an empty test loader.")
    return torch.cat(logits_batches), torch.cat(target_batches)


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
    logits, targets = collect_test_logits(model, data.test_loader, device)
    errors = rank_high_confidence_errors(
        logits,
        targets,
        data.class_names,
        top_k=args.top_k,
    )
    payload = {
        "checkpoint_epoch": checkpoint["epoch"],
        "samples": int(targets.numel()),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"checkpoint_epoch={checkpoint['epoch']} samples={payload['samples']}")
    for rank, error in enumerate(errors[:10], start=1):
        print(
            f"{rank:02d}. {error['true_class']} -> {error['predicted_class']} "
            f"confidence={error['confidence']:.4f} index={error['index']}"
        )
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
