"""Rank the most frequent class confusions from an evaluation report.

Example:
    python -m scripts.analyze_errors \
        --report outputs/augmentation_sweeps_formal/crop_flip_seed_2024/test_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.evaluation.error_analysis import rank_confusions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_report(path: Path) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Load only the portable fields required for error ranking."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    class_names = tuple(item["name"] for item in payload["classes"])
    confusion_matrix = torch.tensor(payload["confusion_matrix"], dtype=torch.int64)
    return confusion_matrix, class_names


def main() -> None:
    args = parse_args()
    if not args.report.is_file():
        raise FileNotFoundError(f"Report not found: {args.report}")

    confusion_matrix, class_names = load_report(args.report)
    confusions = rank_confusions(
        confusion_matrix,
        class_names,
        top_k=args.top_k,
    )

    output_path = args.output or args.report.with_name("error_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(confusions, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"report={args.report}")
    print(f"top_k={len(confusions)}")
    for rank, item in enumerate(confusions, start=1):
        print(
            f"{rank:02d}. {item['true_class']} -> "
            f"{item['predicted_class']}: {item['count']}"
        )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
