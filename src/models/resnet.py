"""ResNet-18 baseline adapted from ImageNet input to CIFAR input.

The baseline adapts torchvision's standard ResNet-18 instead of copying its
full source.

Why adaptation is required:
    ImageNet images are commonly 224x224, while CIFAR images are 32x32. The
    original 7x7 stride-2 stem and max-pool would shrink CIFAR feature maps too
    aggressively before the residual stages can learn useful details.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from torch import nn
from torchvision.models import ResNet, resnet18
import yaml


@dataclass(frozen=True)
class ResNet18Config:
    """Configuration for a from-scratch CIFAR ResNet-18 baseline."""

    in_channels: int = 3
    num_classes: int = 10
    weights: str = "none"
    zero_init_residual: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResNet18Config":
        with Path(path).open("r", encoding="utf-8") as file:
            raw: Mapping[str, Any] = yaml.safe_load(file)

        if raw.get("model") != "resnet18":
            raise ValueError("Expected model: resnet18.")

        config = cls(
            in_channels=int(raw.get("in_channels", 3)),
            num_classes=int(raw.get("num_classes", 10)),
            weights=str(raw.get("weights", "none")).lower(),
            zero_init_residual=bool(raw.get("zero_init_residual", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.weights != "none":
            raise ValueError(
                "The first reproducible baseline trains from scratch; set weights: none."
            )


def adapt_resnet18_for_cifar(model: ResNet, config: ResNet18Config) -> ResNet:
    """Replace the ImageNet-specific input stem and classification head.

    Responsibility:
        Preserve all residual stages while adapting input downsampling and the
        number of output classes for 32x32 CIFAR images.
    Input:
        A standard torchvision ResNet-18 and validated configuration.
    Output:
        The same model object with three modules replaced.
    Required replacements:
        1. ``model.conv1``: 3x3, stride 1, padding 1, no bias.
        2. ``model.maxpool``: identity, so the stem does not downsample again.
        3. ``model.fc``: map the existing feature width to ``num_classes``.
    """
    model.conv1=nn.Conv2d(
        in_channels=config.in_channels,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()  # type: ignore[assignment]
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    return model


def build_resnet18(config: ResNet18Config) -> ResNet:
    """Construct a torchvision ResNet-18 and apply the CIFAR adaptation."""
    config.validate()
    model = resnet18(
        weights=None,
        zero_init_residual=config.zero_init_residual,
    )
    return adapt_resnet18_for_cifar(model, config)
