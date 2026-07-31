"""Plot training & validation loss/accuracy curves from an experiment's
history.jsonl. Extracts the append-only JSON Lines record produced by the
training engine and renders a two-panel figure saved under reports/figures.

Example:
    python -m scripts.plot_training_curve \
        --history outputs/cosine_crop_flip/history.jsonl \
        --output reports/figures/cosine_crop_flip_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def load_history(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
                if key not in record:
                    raise KeyError(f"history record missing {key!r}: {record}")
            records.append(record)
    if not records:
        raise ValueError(f"No history records found in {path}")
    return records


def main() -> None:
    args = parse_args()
    if not args.history.is_file():
        raise FileNotFoundError(f"history not found: {args.history}")

    records = load_history(args.history)
    epochs = [int(record["epoch"]) for record in records]
    train_loss = [float(record["train_loss"]) for record in records]
    val_loss = [float(record["val_loss"]) for record in records]
    train_acc = [float(record["train_accuracy"]) for record in records]
    val_acc = [float(record["val_accuracy"]) for record in records]
    lr = [float(record.get("learning_rate", float("nan"))) for record in records]

    title = args.title or args.history.parent.name

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    figure.suptitle(f"Training curves — {title}", fontsize=13, fontweight="bold")

    loss_ax = axes[0]
    loss_ax.plot(epochs, train_loss, "o-", color="#1f77b4", label="Train loss", lw=1.6, ms=3)
    loss_ax.plot(epochs, val_loss, "s--", color="#d62728", label="Val loss", lw=1.6, ms=3)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss")
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    acc_ax = axes[1]
    acc_ax.plot(epochs, train_acc, "o-", color="#1f77b4", label="Train accuracy", lw=1.6, ms=3)
    acc_ax.plot(epochs, val_acc, "s--", color="#d62728", label="Val accuracy", lw=1.6, ms=3)
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Accuracy")
    acc_ax.grid(alpha=0.3)
    acc_ax.legend()

    axes[0].set_xticks(epochs[:: max(1, len(epochs) // 10)])
    axes[1].set_xticks(epochs[:: max(1, len(epochs) // 10)])

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)

    last = records[-1]
    peak_val = max(val_acc)
    peak_epoch = val_acc.index(peak_val) + 1
    print(
        f"epochs={len(records)} final_val_acc={val_acc[-1]:.4f} "
        f"best_val_acc={peak_val:.4f}@epoch{peak_epoch} "
        f"final_lr={lr[-1]:.6f}"
    )
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()