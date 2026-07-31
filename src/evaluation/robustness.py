"""Evaluation helpers for controlled input perturbations."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def add_gaussian_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise in the current image representation.

    The CIFAR loader supplies normalized tensors, so ``std`` is measured in
    normalized tensor units rather than raw uint8 pixel values.
    """
    if std<0:
        raise ValueError("std must be non-negative")
    noise=torch.randn_like(images)
    return images+std*noise


@torch.inference_mode()
def evaluate_gaussian_noise(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_std: float,
) -> float:
    """Measure classification accuracy after adding Gaussian noise."""
    model.eval()
    correct = 0
    samples = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        perturbed_images = add_gaussian_noise(images, noise_std)
        predictions = model(perturbed_images).argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        samples += int(targets.numel())
    if samples == 0:
        raise ValueError("Cannot evaluate robustness on an empty loader.")
    return correct / samples
