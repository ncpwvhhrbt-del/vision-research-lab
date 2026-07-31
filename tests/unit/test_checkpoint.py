"""Checkpoint round-trip test independent of CIFAR."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from torch import nn
from torch.optim import SGD

from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.metrics import EpochMetrics


class CheckpointTests(unittest.TestCase):
    def test_round_trip_restores_metadata(self) -> None:
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.1)
        metrics = EpochMetrics(loss=1.0, accuracy=0.5, samples=4)

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=3,
                train_metrics=metrics,
                val_metrics=metrics,
                experiment_name="unit-test",
            )
            checkpoint = load_checkpoint(path, model=model, optimizer=optimizer)

        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["experiment"], "unit-test")


if __name__ == "__main__":
    unittest.main()
