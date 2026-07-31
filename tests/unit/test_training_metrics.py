"""Behavioral specification for epoch metric accumulation."""

import unittest

import torch

from src.training.metrics import ClassificationMeter


class ClassificationMeterTests(unittest.TestCase):
    def test_loss_is_weighted_across_unequal_batches(self) -> None:
        meter = ClassificationMeter()
        meter.update(
            loss=torch.tensor(1.0),
            logits=torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            targets=torch.tensor([0, 1]),
        )
        meter.update(
            loss=torch.tensor(3.0),
            logits=torch.tensor([[0.0, 3.0]]),
            targets=torch.tensor([0]),
        )

        metrics = meter.compute()

        self.assertAlmostEqual(metrics.loss, 5.0 / 3.0)
        self.assertAlmostEqual(metrics.accuracy, 2.0 / 3.0)
        self.assertEqual(metrics.samples, 3)

    def test_compute_rejects_empty_meter(self) -> None:
        with self.assertRaises(RuntimeError):
            ClassificationMeter().compute()


if __name__ == "__main__":
    unittest.main()
