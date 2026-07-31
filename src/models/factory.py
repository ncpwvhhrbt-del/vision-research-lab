"""Select exactly one model from an experiment's model YAML file.

The training pipeline should depend on this function instead of importing a
specific architecture. This keeps training code unchanged when experiments
switch between SimpleCNN and ResNet-18.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from torch import nn
import yaml

from .resnet import ResNet18Config, build_resnet18
from .simple_cnn import SimpleCNNConfig, build_simple_cnn


SUPPORTED_MODELS = ("simple_cnn", "resnet18")


def read_model_name(path: str | Path) -> str:
    """Read and validate only the model selector from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw: Mapping[str, Any] = yaml.safe_load(file)

    model_name = str(raw.get("model", "")).lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}; choose one of {SUPPORTED_MODELS}."
        )
    return model_name


def build_model_from_yaml(path: str | Path) -> nn.Module:
    """Build the one architecture selected by ``model:`` in YAML.

    Responsibility:
        Dispatch configuration parsing and construction to the selected model.
    Input:
        Path to one model YAML file.
    Output:
        One ``nn.Module`` instance, never both models.
    Key calls:
        ``read_model_name``, each config's ``from_yaml``, and each builder.
    """
    model_name = read_model_name(path)
    if model_name == "simple_cnn":
        config = SimpleCNNConfig.from_yaml(path)
        return build_simple_cnn(config)
    if model_name == "resnet18":
        config = ResNet18Config.from_yaml(path)
        return build_resnet18(config)
    raise AssertionError(f"Validated model selector reached an impossible branch: {model_name!r}.")
