"""Checkpoint persistence kept outside algorithmic training loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .metrics import EpochMetrics


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
    experiment_name: str,
) -> Path:
    """Save enough state to inspect or resume a baseline experiment."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "experiment": experiment_name,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_metrics": {
            "loss": train_metrics.loss,
            "accuracy": train_metrics.accuracy,
            "samples": train_metrics.samples,
        },
        "val_metrics": {
            "loss": val_metrics.loss,
            "accuracy": val_metrics.accuracy,
            "samples": val_metrics.samples,
        },
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model state and optionally optimizer state."""
    checkpoint: dict[str, Any] = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
