"""Unified classification evaluation and report export."""

from .classification import (
    ClassificationReport,
    evaluate_classifier,
    report_from_confusion_matrix,
    update_confusion_matrix,
)
from .export import save_report_json

__all__ = [
    "ClassificationReport",
    "update_confusion_matrix",
    "report_from_confusion_matrix",
    "evaluate_classifier",
    "save_report_json",
]
