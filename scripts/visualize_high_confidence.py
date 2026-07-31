"""Visualize the most confident but incorrect test predictions as an image grid.

Runs inference on the pinned checkpoint, ranks the incorrect predictions by
their softmax confidence, and renders the misclassified test images with the
true / predicted labels and confidence so failure modes are easy to inspect.

Example:
    python -m scripts.visualize_high_confidence \
        --config configs/experiment/cosine_crop_flip.yaml \
        --checkpoint outputs/cosine_crop_flip/best.pt \
        --top-k 18 \
        --output reports/figures/high_confidence_errors_cosine_crop_flip.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
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
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


@torch.inference_mode()
def collect_test(model, loader, device):
    model.eval()
    logits_batches, target_batches, image_batches = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits_batches.append(model(images).cpu())
        target_batches.append(targets.cpu())
        image_batches.append(images.cpu())
    if not logits_batches:
        raise ValueError("Cannot visualize an empty test loader.")
    return (
        torch.cat(logits_batches),
        torch.cat(target_batches),
        torch.cat(image_batches),
    )


def unnormalize(images: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return images * std_t + mean_t


def main() -> None:
    args = parse_args()
    spec = ExperimentSpec.from_yaml(args.config)
    seed_everything(spec.seed)
    device = resolve_device(spec.training.device)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    data_config = CIFARDataConfig.from_yaml(spec.data_config)
    data = build_dataloaders(data_config)
    model = build_model_from_yaml(spec.model_config).to(device)
    loaded = load_checkpoint(args.checkpoint, model=model, map_location=device)

    logits, targets, images = collect_test(model, data.test_loader, device)
    errors = rank_high_confidence_errors(
        logits, targets, data.class_names, top_k=args.top_k
    )

    cols = max(1, args.cols)
    rows = math.ceil(len(errors) / cols)
    figure, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    axes = axes.ravel() if rows * cols > 1 else [axes]

    shown = unnormalize(images, data_config.mean, data_config.std)
    for axis in axes:
        axis.axis("off")
    for position, error in enumerate(errors):
        index = int(error["index"])
        image = shown[index].clamp(0, 1).permute(1, 2, 0).numpy()
        axis = axes[position]
        axis.imshow(image, interpolation="nearest")
        axis.set_title(
            f"{error['true_class']} → {error['predicted_class']}\n"
            f"conf={error['confidence']:.4f}",
            fontsize=9,
            color="#b30000",
        )

    figure.suptitle(
        "High-confidence misclassifications "
        f"(checkpoint_epoch={loaded['epoch']})",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)

    print(
        f"samples={targets.numel()} errors_ranked={len(errors)} "
        f"top_confidence={errors[0]['confidence']:.4f} ({errors[0]['true_class']}→{errors[0]['predicted_class']})"
    )
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()