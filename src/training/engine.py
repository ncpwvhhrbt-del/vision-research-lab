"""Architecture-independent loops for one training or evaluation epoch."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import ClassificationMeter, EpochMetrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Update model parameters once for every batch in ``loader``.

    Required reasoning order for each batch:
        move data -> clear old gradients -> forward -> loss -> backward
        -> optimizer step -> metric update

    The function must call ``model.train()`` once before iteration and return
    the meter's final ``EpochMetrics``.
    """
    model.train()
    meter=ClassificationMeter()
    for images,targets in loader:
        images=images.to(device)
        targets=targets.to(device)
        optimizer.zero_grad()
        logits=model(images)
        loss=criterion(logits,targets)
        loss.backward()
        optimizer.step()
        meter.update(loss,logits,targets)
    return meter.compute()

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate without gradients or parameter updates.

    Required reasoning order:
        evaluation mode -> move data -> forward -> loss -> metric update

    Do not call ``backward`` or an optimizer anywhere in this function.
    """
    model.eval()
    meter=ClassificationMeter()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        meter.update(loss,logits,targets)
    return meter.compute()
