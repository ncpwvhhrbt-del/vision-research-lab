"""Architecture-independent classification evaluation.

Confusion-matrix convention:
    rows    = true classes
    columns = predicted classes

For example, ``matrix[2, 5]`` counts true class 2 samples predicted as class 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassificationReport:
    """Structured result consumed by exporters and future research modules."""

    class_names: tuple[str, ...]
    confusion_matrix: torch.Tensor
    overall_accuracy: float
    per_class_accuracy: tuple[float | None, ...]
    samples: int


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Accumulate one batch into a CPU integer confusion matrix.

    Responsibility:
        Convert logits to predicted class indices and count every
        ``(true_class, predicted_class)`` pair without a Python sample loop.
    Inputs:
        Matrix ``[C, C]``, logits ``[B, C]``, targets ``[B]``.
    Output:
        None; update ``confusion_matrix`` in place.
    Key calls:
        ``argmax``, encoded index ``target * C + prediction``, ``bincount``,
        and ``reshape``.
    """

    num_classes=confusion_matrix.size(0)
    predictions=logits.argmax(dim=1).cpu()
    targets=targets.cpu()
    encoded_pairs = targets * num_classes + predictions
    batch_counts=torch.bincount(encoded_pairs,minlength=num_classes*num_classes).reshape(num_classes,num_classes)
    confusion_matrix+=batch_counts



def report_from_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: tuple[str, ...],
) -> ClassificationReport:
    """Calculate overall and per-class accuracy from a confusion matrix.

    A class with zero true samples must receive ``None`` rather than causing a
    division-by-zero result.
    """
    if confusion_matrix.ndim != 2 or confusion_matrix.size(0) != confusion_matrix.size(1):
        raise ValueError("confusion_matrix must be square [C, C].")
    if confusion_matrix.size(0) != len(class_names):
        raise ValueError("class_names length must match confusion_matrix size.")
    class_totals=confusion_matrix.sum(dim=1)
    class_correct=confusion_matrix.diag()
    samples=int(class_totals.sum().item())
    correct=int(class_correct.sum().item())
    overall_accuracy=correct/samples if samples else 0.0
    per_class_accuracy_list=[]
    for correct_counts,total in zip(class_correct,class_totals):
        if total.item()==0:
            per_class_accuracy_list.append(None)
        else :
            per_class_accuracy_list.append(correct_counts.item()/total.item())
    per_class_accuracy=tuple(per_class_accuracy_list)
    return ClassificationReport(class_names=class_names,
                                confusion_matrix=confusion_matrix.clone(),
                                overall_accuracy=overall_accuracy,
                                per_class_accuracy=per_class_accuracy,
                                samples=samples)



@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: tuple[str, ...],
) -> ClassificationReport:
    """Run the full test loader and return a classification report.

    Required order:
        eval mode -> empty CPU matrix -> batch loop -> move images only
        -> forward -> update matrix -> construct report

    Targets may remain on CPU because the confusion matrix is intentionally
    accumulated on CPU and no loss is computed here.
    """
    model.eval()
    confusion_matrix=torch.zeros((len(class_names),len(class_names)),dtype=torch.int64)
    for images,targets in loader:
        images=images.to(device)
        logits=model(images)
        update_confusion_matrix(confusion_matrix,logits,targets)
    return report_from_confusion_matrix(confusion_matrix,class_names)
