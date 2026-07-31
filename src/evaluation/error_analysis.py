"""Utilities for turning a confusion matrix into ranked error cases."""

from __future__ import annotations

import torch
import numpy as np

def rank_confusions(
    confusion_matrix: torch.Tensor,
    class_names: tuple[str, ...],
    *,
    top_k: int = 10,
) -> list[dict[str, int | str]]:
    """Return the most frequent off-diagonal class confusions.

    Responsibility:
        Ignore correct predictions on the diagonal and rank each
        ``true_class -> predicted_class`` error by its count.

    Input:
        ``confusion_matrix`` has shape ``[C, C]`` with rows as true classes
        and columns as predicted classes. ``class_names`` has length ``C``.

    Output:
        A list of dictionaries with ``true_class``, ``predicted_class`` and
        ``count``, sorted from most frequent to least frequent.

    Key design question:
        Should ties be deterministic? Yes. Use true index and predicted index
        as a stable secondary ordering so reports are reproducible.
    """
    if confusion_matrix.ndim!=2 or confusion_matrix.size(0)!=confusion_matrix.size(1):
        raise ValueError("confusion_matrix must be square [C,C]")
    if confusion_matrix.size(0)!= len(class_names):
        raise ValueError("class_names length must match confusion_matrix size.")
    if top_k <=0:
        raise ValueError("top_k must be positive.")
    matrix=confusion_matrix.detach().cpu()
    matrix_np=matrix.numpy().copy()
    np.fill_diagonal(matrix_np,0)
    rows,cols=np.where(matrix_np>0)
    if len(rows)==0:
        return[]
    error_list=[]
    for i,j in zip(rows,cols):
        count=matrix_np[i,j]
        error_list.append((count,i,j))
    error_list.sort(key=lambda x: (-x[0], x[1], x[2]))
    error_list=error_list[:top_k]
    results=[]
    for count,i,j in error_list:
        results.append({"true_class":class_names[i],
                        "predicted_class":class_names[j],
                        "count":int(count)
        })
    return results
