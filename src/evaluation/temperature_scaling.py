"""Post-hoc temperature scaling for classification calibration."""

from __future__ import annotations

import torch
from torch import nn


def scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by a positive scalar temperature."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return logits / temperature


def fit_temperature(
    validation_logits: torch.Tensor,
    validation_targets: torch.Tensor,
) -> float:
    """Fit one scalar temperature on validation logits.
    The returned temperature must be learned from validation data only.
    """
    if validation_logits.ndim!=2 or validation_targets.ndim!=1:
        raise ValueError("logits must be [N, C], targets must be [N].")
    if validation_logits.size(0)!=validation_targets.numel():
        raise ValueError("Number of logits and targets must match.")
    if validation_logits.size(0)==0:
        raise ValueError("Validation inputs cannot be empty.")
    if torch.any(validation_targets < 0) or torch.any(validation_targets>=validation_logits.size(1)):
        raise ValueError("validation_targets contain an invalid class index.")

    logits=validation_logits.detach().float().clone()
    targets=validation_targets.detach().long().clone()
    log_temperature=nn.Parameter(torch.zeros(1))
    optimizer=torch.optim.LBFGS([log_temperature],lr=0.1,max_iter=50,line_search_fn="strong_wolfe")
    criterion=nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        temperature=torch.exp(log_temperature)
        loss=criterion(logits/temperature,targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).item())

def calibrated_probabilities(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Convert logits to probabilities after temperature scaling."""
    return torch.softmax(scale_logits(logits, temperature), dim=1)
