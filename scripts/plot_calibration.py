"""Plot a reliability diagram from a calibration report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.report.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    bins = payload["bins"]
    confidence = [item["confidence"] for item in bins]
    accuracy = [item["accuracy"] for item in bins]
    counts = [item["count"] for item in bins]

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(confidence, accuracy, "o-", label="Model")
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"Reliability diagram (ECE={payload['ece']:.4f})")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"bins={len(bins)} samples={sum(counts)}")
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()
