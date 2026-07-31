"""Serialize classification reports into GitHub-friendly artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .classification import ClassificationReport


def save_report_json(
    report: ClassificationReport,
    path: str | Path,
) -> Path:
    """Write a human-readable report without serializing PyTorch internals."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": report.samples,
        "overall_accuracy": report.overall_accuracy,
        "classes": [
            {
                "name": name,
                "accuracy": accuracy,
            }
            for name, accuracy in zip(
                report.class_names,
                report.per_class_accuracy,
            )
        ],
        "confusion_matrix": report.confusion_matrix.tolist(),
    }
    with destination.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return destination
