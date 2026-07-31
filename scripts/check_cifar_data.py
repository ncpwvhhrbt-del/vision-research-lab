"""Validate the configured local CIFAR-10 dataset without downloading files.

Run from the repository root:
    python -m scripts.check_cifar_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torchvision.datasets import CIFAR10

from src.data.cifar import CIFARDataConfig, inspect_cifar10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/cifar10.yaml"),
        help="Path to a CIFAR-10 YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CIFARDataConfig.from_yaml(args.config)
    report = inspect_cifar10(config.root)

    if not report["complete"]:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit("CIFAR-10 is missing or incomplete.")

    train_dataset = CIFAR10(root=config.root, train=True, download=False)
    test_dataset = CIFAR10(root=config.root, train=False, download=False)
    image, label = train_dataset[0]
    report.update(
        {
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "classes": train_dataset.classes,
            "sample_image_size": list(image.size),
            "sample_label": label,
            "sample_class": train_dataset.classes[label],
        }
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
