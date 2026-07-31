"""Confidence calibration metrics for classification models."""

from __future__ import annotations

import torch


def calibration_bin_stats(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    *,
    num_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Return reliability-diagram statistics for each non-empty bin."""
    if confidences.ndim!=1 or correctness.ndim!=1:
        raise ValueError('inputs must be one-dimensional')
    if confidences.numel()!=correctness.numel():
        raise ValueError("Inputs must have the same number of samples.")
    if confidences.numel()==0:
        raise ValueError("Inputs cannot be empty")
    if num_bins <=0:
        raise ValueError("num_bins must be positive")
    confidences=confidences.detach().float().cpu()
    correctness=correctness.detach().bool().cpu()
    if torch.any(confidences<0) or torch.any(confidences>1):
        raise ValueError("confidences must be in [0, 1].")
    boundaries=torch.linspace(0.0,1.0,num_bins+1)
    stats=[]
    for index in range(num_bins):
        left=boundaries[index]
        right=boundaries[index+1]
        if index==num_bins-1:
            mask=(confidences>=left)&(confidences<=right)
        else:
            mask=(confidences>=left)&(confidences<right)
        if not torch.any(mask):
            continue
        bin_accuracy=correctness[mask].float().mean().item()
        bin_confidence=confidences[mask].mean().item()
        count = int(mask.sum().item())
        stats.append({"bin":index,
                    "count":count,
                    "confidence":bin_confidence,
                    "accuracy":bin_accuracy,
                })
    return stats



def expected_calibration_error(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    *,
    num_bins: int = 10,
) -> float:
    """Compute equal-width Expected Calibration Error (ECE).

    ``confidences`` contains the predicted probability of each sample's
    predicted class. ``correctness`` is a boolean tensor indicating whether
    that prediction was correct.
    """
    if confidences.ndim!=1 or correctness.ndim!=1:
        raise ValueError('inputs must be one-dimensional')
    if confidences.numel()!=correctness.numel():
        raise ValueError("Inputs must have the same number of samples.")
    if confidences.numel()==0:
        raise ValueError("Inputs cannot be empty")
    if num_bins <=0:
        raise ValueError("num_bins must be positive")
    confidences=confidences.detach().float().cpu()
    correctness=correctness.detach().bool().cpu()
    if torch.any(confidences<0) or torch.any(confidences>1):
        raise ValueError("confidences must be in [0, 1].")
    boundaries=torch.linspace(0.0,1.0,num_bins+1)
    total=confidences.numel()
    ece=0.0
    for index in range(num_bins):
        left=boundaries[index]
        right=boundaries[index+1]
        if index==num_bins-1:
            mask=(confidences>=left)&(confidences<=right)
        else:
            mask=(confidences>=left)&(confidences<right)
        if not torch.any(mask):
            continue
        bin_accuracy=correctness[mask].float().mean().item()
        bin_confidence=confidences[mask].mean().item()
        ece += (mask.sum().item() / total) * abs(bin_accuracy - bin_confidence)
    return ece
