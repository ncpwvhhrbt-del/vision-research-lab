"""Optimizer construction separated from the epoch training loop."""

from __future__ import annotations

from torch import nn
from torch.optim import Optimizer
from torch.optim import AdamW,SGD
from .config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Build the optimizer selected in the experiment configuration.

    Responsibility:
        Give all trainable model parameters to either SGD or AdamW using the
        configured hyperparameters.
    Inputs:
        Model and validated training configuration.
    Output:
        One initialized PyTorch optimizer.
    Key calls:
        ``model.parameters()``, ``torch.optim.SGD``, ``torch.optim.AdamW``.
    """
    config.validate()
    if config.optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise AssertionError("TrainingConfig already validates optimizer_name.")
