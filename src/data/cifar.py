"""CIFAR-10 data configuration, validation, transforms, and loaders.

Tensor flow:
    PIL image [32, 32, 3]
    -> float tensor [3, 32, 32]
    -> batch [B, 3, 32, 32]
    -> labels [B]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .augmentations import build_train_augmentation

CIFAR10_REQUIRED_FILES = (
    "batches.meta",
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
)


@dataclass(frozen=True)
class CIFARDataConfig:
    """All data-related choices for one CIFAR-10 experiment."""

    root: Path
    download: bool = False
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    val_ratio: float = 0.1
    split_seed: int = 42
    image_size: int = 32
    num_classes: int = 10
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    random_crop_padding: int = 4
    horizontal_flip_probability: float = 0.5
    augmentation_name: str = "crop_flip"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CIFARDataConfig":
        """Load a config file while keeping YAML details out of training code."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw: Mapping[str, Any] = yaml.safe_load(file)

        if raw.get("dataset") != "cifar10":
            raise ValueError("This module currently supports dataset: cifar10 only.")

        augmentations = raw.get("train_augmentations", {})
        config = cls(
            root=Path(raw["root"]).expanduser(),
            download=bool(raw.get("download", False)),
            batch_size=int(raw.get("batch_size", 128)),
            num_workers=int(raw.get("num_workers", 2)),
            pin_memory=bool(raw.get("pin_memory", True)),
            persistent_workers=bool(raw.get("persistent_workers", True)),
            val_ratio=float(raw.get("val_ratio", 0.1)),
            split_seed=int(raw.get("split_seed", 42)),
            image_size=int(raw.get("image_size", 32)),
            num_classes=int(raw.get("num_classes", 10)),
            mean=tuple(float(value) for value in raw["mean"]),
            std=tuple(float(value) for value in raw["std"]),
            random_crop_padding=int(augmentations.get("random_crop_padding", 4)),
            horizontal_flip_probability=float(
                augmentations.get("horizontal_flip_probability", 0.5)
            ),
            augmentation_name=str(augmentations.get("name", "crop_flip")).lower(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Fail early when an experiment configuration is internally invalid."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1.")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("CIFAR RGB normalization needs three means and stds.")
        if any(value <= 0 for value in self.std):
            raise ValueError("Every normalization std must be positive.")
        if self.random_crop_padding < 0:
            raise ValueError("random_crop_padding cannot be negative.")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")
        if self.augmentation_name not in {"none", "crop", "crop_flip"}:
            raise ValueError("augmentation name must be none, crop, or crop_flip.")


@dataclass(frozen=True)
class DataBundle:
    """Stable return type consumed later by training and evaluation modules."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: tuple[str, ...]


def cifar10_directory(root: str | Path) -> Path:
    """Return torchvision's expected extracted CIFAR-10 directory."""
    return Path(root) / "cifar-10-batches-py"


def inspect_cifar10(root: str | Path) -> dict[str, Any]:
    """Inspect local files without downloading or changing the dataset."""
    dataset_dir = cifar10_directory(root)
    missing = [
        filename
        for filename in CIFAR10_REQUIRED_FILES
        if not (dataset_dir / filename).is_file()
    ]
    return {
        "root": str(Path(root).resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "exists": dataset_dir.is_dir(),
        "complete": not missing,
        "missing_files": missing,
    }


def build_transforms(config: CIFARDataConfig) -> tuple[Any, Any]:
    """Build stochastic train and deterministic evaluation transforms.

    Responsibility:
        Convert a PIL image into a normalized ``float32`` tensor. Apply random
        augmentation only to training samples.
    Input:
        A validated ``CIFARDataConfig``.
    Output:
        ``(train_transform, eval_transform)``.
    Key calls:
        ``transforms.Compose``, ``RandomCrop``, ``RandomHorizontalFlip``,
        ``ToTensor``, and ``Normalize``.
    """
    normalize=transforms.Normalize(mean=config.mean,std=config.std)
    train_steps = build_train_augmentation(
        name=config.augmentation_name,
        image_size=config.image_size,
        random_crop_padding=config.random_crop_padding,
        horizontal_flip_probability=config.horizontal_flip_probability,
    )
    train_transform=transforms.Compose([
        *train_steps,
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform=transforms.Compose([transforms.ToTensor(),normalize])
    return train_transform,eval_transform



def build_dataloaders(config: CIFARDataConfig) -> DataBundle:
    """Create reproducible train/validation/test DataLoaders.

    Responsibility:
        Load CIFAR-10, create a deterministic train/validation index split,
        keep augmentation off validation/test data, and configure batching.
    Input:
        A validated ``CIFARDataConfig``.
    Output:
        A ``DataBundle`` with three loaders and class names.
    Key calls:
        ``CIFAR10``, ``torch.randperm``, ``Subset``, and ``DataLoader``.

    Important invariant:
        Train and validation subsets must wrap separate dataset objects because
        their transforms differ, even though their underlying images match.
    """
    config.validate()
    train_transform, eval_transform = build_transforms(config)
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.root, train=True, transform=train_transform, download=config.download
    )
    validation_dataset = torchvision.datasets.CIFAR10(
        root=config.root, train=True, transform=eval_transform, download=False
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.root, train=False, transform=eval_transform, download=config.download
    )
    generator = torch.Generator().manual_seed(config.split_seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    validation_size = int(len(indices) * config.val_ratio)
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers and config.num_workers > 0,
    }
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(validation_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=tuple(train_dataset.classes),
    )
