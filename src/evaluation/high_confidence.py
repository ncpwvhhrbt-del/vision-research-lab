"""Identify confident but incorrect classification predictions."""

from __future__ import annotations


import torch


def rank_high_confidence_errors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: tuple[str, ...],
    *,
    top_k: int = 20,
) -> list[dict[str, float | int | str]]:
    """Return the highest-confidence incorrect predictions.

    The returned ``index`` refers to the row index in the supplied logits and
    targets tensors, not necessarily the original dataset index.
    """
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("logits must be [N, C], targets must be [N].")
    if logits.size(0)!=targets.numel():
        raise ValueError("Number of logits and targets must match.")
    if logits.size(1) != len(class_names):
        raise ValueError("class_names length must match number of classes.")
    if logits.size(0)==0:
        raise ValueError("Input cannot be empty")
    if top_k<=0:
        raise ValueError("top_k must be positive")
    probabilities=torch.softmax(logits.detach().float().cpu(),dim=1)
    targets=targets.detach().long().cpu()
    confidences,predictions=probabilities.max(dim=1)
    wrong=predictions!=targets
    indices=torch.where(wrong)[0].tolist()
    indices.sort(key=lambda index: (-float(confidences[index]), index))
    return [
        {"index":int(index),
         "true_class":class_names[int(targets[index])],
         "predicted_class":class_names[int(predictions[index])],
         "confidence":float(confidences[index]),
        }
        for index in indices[:top_k]
    ]
