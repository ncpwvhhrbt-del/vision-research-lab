"""Run one configured training experiment.

Example:
    python -m scripts.train --config configs/experiment/baseline_simple_cnn.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import islice
from pathlib import Path

from torch import nn

from src.data.cifar import CIFARDataConfig, build_dataloaders
from src.models.factory import build_model_from_yaml
from src.training.checkpoint import save_checkpoint
from src.training.engine import evaluate, train_one_epoch
from src.training.experiment import ExperimentSpec
from src.training.history import append_history
from src.training.optim import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.runtime import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/baseline_simple_cnn.yaml"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional short-run override; the YAML remains the formal default.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ExperimentSpec.from_yaml(args.config)
    if args.seed is not None or args.output_dir is not None:
        spec = replace(
            spec,
            seed=args.seed if args.seed is not None else spec.seed,
            output_dir=args.output_dir if args.output_dir is not None else spec.output_dir,
        )
    seed_everything(spec.seed)
    device = resolve_device(spec.training.device)

    data_config = CIFARDataConfig.from_yaml(spec.data_config)
    data = build_dataloaders(data_config)
    model = build_model_from_yaml(spec.model_config).to(device)
    optimizer = build_optimizer(model, spec.training)
    criterion = nn.CrossEntropyLoss()

    epochs = args.epochs if args.epochs is not None else spec.training.epochs
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    scheduler = build_scheduler(optimizer, spec.training, epochs)

    best_accuracy = -1.0
    print(f"experiment={spec.name} device={device} epochs={epochs}")
    for epoch in range(1, epochs + 1):
        train_loader = (
            islice(data.train_loader, args.max_train_batches)
            if args.max_train_batches is not None
            else data.train_loader
        )
        val_loader = (
            islice(data.val_loader, args.max_val_batches)
            if args.max_val_batches is not None
            else data.val_loader
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_history(spec.output_dir / "history.jsonl", record)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f}"
        )

        if val_metrics.accuracy > best_accuracy:
            best_accuracy = val_metrics.accuracy
            save_checkpoint(
                spec.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                experiment_name=spec.name,
            )

        if scheduler is not None:
            scheduler.step()

    print(f"best_val_accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
