"""Probability-based scoring metrics for multiclass classification."""

from __future__ import annotations

import torch


def multiclass_brier_score(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute the mean multiclass Brier score.

    ``probabilities`` has shape ``[N, C]`` and each row should represent a
    probability distribution. ``targets`` has shape ``[N]``.
    """
    if probabilities.ndim!=2 or targets.ndim!=1:
        raise ValueError("probabilities must be [N, C], targets must be [N].")
    if probabilities.size(0)!=targets.numel():
        raise ValueError("Number of probabilities and targets must match.")
    if probabilities.size(0)==0:
        raise ValueError("Inputs cannot be empty.")
    if torch.any(targets < 0) or torch.any(targets >= probabilities.size(1)):
        raise ValueError("targets contain an invalid class index.")
    if torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("probabilities must be in [0, 1].")
    targets=targets.long()
    one_hot=torch.zeros_like(probabilities)
    one_hot.scatter_(1,targets.unsqueeze(1),1.0)
    squared_error=(probabilities-one_hot).pow(2).sum(dim=1)
    return squared_error.mean().item()
