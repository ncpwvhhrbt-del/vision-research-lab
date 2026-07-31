"""A compact, shape-safe CNN baseline for CIFAR-sized images.

Expected tensor flow:
    input:       [B,   3, 32, 32]
    block 1:     [B,  32, 16, 16]
    block 2:     [B,  64,  8,  8]
    block 3:     [B, 128,  4,  4]
    global pool: [B, 128,  1,  1]
    flatten:     [B, 128]
    logits:      [B,  10]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import yaml


@dataclass(frozen=True)
class SimpleCNNConfig:
    """Architecture choices required to reproduce the SimpleCNN baseline."""

    in_channels: int = 3
    num_classes: int = 10
    channels: tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.20

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimpleCNNConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw: Mapping[str, Any] = yaml.safe_load(file)

        if raw.get("model") != "simple_cnn":
            raise ValueError("Expected model: simple_cnn.")

        config = cls(
            in_channels=int(raw.get("in_channels", 3)),
            num_classes=int(raw.get("num_classes", 10)),
            channels=tuple(int(value) for value in raw.get("channels", [32, 64, 128])),
            dropout=float(raw.get("dropout", 0.20)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if len(self.channels) != 3 or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must contain three positive integers.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class ConvBlock(nn.Sequential):
    """Two convolutions followed by one 2x spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class SimpleCNN(nn.Module):
    """Three-stage CNN that returns unnormalized class scores (logits)."""

    def __init__(self, config: SimpleCNNConfig) -> None:
        super().__init__()
        config.validate()
        channel_1, channel_2, channel_3 = config.channels

        self.block1 = ConvBlock(config.in_channels, channel_1)
        self.block2 = ConvBlock(channel_1, channel_2)
        self.block3 = ConvBlock(channel_2, channel_3)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(channel_3, config.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map an image batch ``[B, 3, 32, 32]`` to logits ``[B, 10]``.

        Responsibility:
            Pass tensors through three convolution blocks, global pooling,
            flattening, and the classifier.
        Input:
            ``images`` with shape ``[batch, channels, height, width]``.
        Output:
            Raw logits with shape ``[batch, num_classes]``. Do not apply
            softmax here because CrossEntropyLoss expects logits.
        Key calls:
            ``self.block1``, ``self.block2``, ``self.block3``,
            ``self.global_pool``, ``torch.flatten``, ``self.classifier``.
        """
        features = self.block1(images)
        features = self.block2(features)
        features = self.block3(features)
        features = self.global_pool(features)
        features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features)
        return logits


def build_simple_cnn(config: SimpleCNNConfig) -> SimpleCNN:
    """Public construction function used later by experiment runners."""
    return SimpleCNN(config)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that will be updated by an optimizer."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
