"""Streaming classification metrics with no dependency on model architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated values returned by one complete loader pass."""

    loss: float
    accuracy: float
    samples: int


class ClassificationMeter:
    """Accumulate sample-weighted loss and top-1 accuracy across batches."""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.correct = 0
        self.samples = 0

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update totals from one batch.

        Responsibility:
            Convert mean batch loss to summed sample loss, count correct top-1
            predictions, and count samples.
        Inputs:
            Scalar mean ``loss``, logits ``[B, C]``, targets ``[B]``.
        Output:
            None; mutate only this meter's three numeric totals.
        Key calls:
            ``targets.size(0)``, ``logits.argmax(dim=1)``, ``loss.item()``.
        """
        batch_size=targets.size(0)
        predictions=logits.argmax(dim=1)
        self.loss_sum+=loss.item()*batch_size
        self.correct+=(predictions==targets).sum().item()
        self.samples+=batch_size
    def compute(self) -> EpochMetrics:
        """Convert accumulated totals into epoch averages.

        Raise a clear error if no samples were observed; otherwise return loss
        and accuracy normalized by the total sample count.
        """

        if self.samples==0:
            raise RuntimeError("No samples")
        else:
            return EpochMetrics(loss=self.loss_sum/self.samples,accuracy=self.correct/self.samples,samples=self.samples)
